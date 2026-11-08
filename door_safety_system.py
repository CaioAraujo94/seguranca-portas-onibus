import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------------

FRAME_W, FRAME_H = 640, 480

FRAMES_PARA_CONFIRMAR_LIVRE = 5
FRAMES_PARA_MODO_INCERTO = 8

# Intertravamento porta/veiculo (simulado)
FRAMES_PARA_FECHAR_PORTA = 12       # duracao do fechamento da porta
FRAMES_PARA_LIBERAR_VEICULO = 6     # frames LIVRE seguidos, com porta fechada, para liberar
COMANDOS_FECHAR_MOTORISTA = {70, 105, 190, 240, 262}   # frames em que o motorista aperta "fechar" (simulacao)

# Alarmes sonoros
FPS_SIMULACAO = 20.0
ALARME_FECHAMENTO_S = 2.0           # alarme antes de a porta comecar a fechar
ALARME_ABERTURA_S = 1.5             # alarme antes de a porta comecar a abrir
ALARME_IGNICAO_S = 5.0              # alarme ao ligar o veiculo com o freio de porta inoperante

# Falha do freio de porta ("anjo da guarda")
MENSAGENS_FREIO_PORTA = ("Freio de porta inoperante", "Visitar a oficina")
SEGUNDOS_ALTERNANCIA_MENSAGEM = 1.0
MENSAGEM_FREIO_PORTA_DURACAO_S = None   # None = mensagens ficam enquanto a falha existir

# Objeto/pessoa de dificil identificacao (modo video, HOG)
LIMIAR_CONFIANCA_PESSOA_HOG = 0.5   # abaixo disso a deteccao e "incerta" (calibrar com videos reais)


class EstadoPorta(Enum):
    LIVRE = "LIVRE"
    OCUPADA = "OCUPADA"
    CONTATO_BORDA = "CONTATO_BORDA"
    INCERTA = "INCERTA"
    OBJETO_INCERTO = "OBJETO_INCERTO"


@dataclass
class ZonaPorta:
    nome: str
    zona_geral: tuple
    zona_borda: tuple

    def ponto_na_zona(self, ponto, zona):
        x, y = ponto
        x1, y1, x2, y2 = zona
        return x1 <= x <= x2 and y1 <= y <= y2

    def bbox_toca_zona(self, bbox, zona):
        bx1, by1, bx2, by2 = bbox
        zx1, zy1, zx2, zy2 = zona
        return not (bx2 < zx1 or bx1 > zx2 or by2 < zy1 or by1 > zy2)


# ----------------------------------------------------------------------------
# RASTREADOR SIMPLES (centroid tracker)
# ----------------------------------------------------------------------------

class RastreadorCentroide:
    def __init__(self, dist_max=80, sumiu_max=10):
        self.prox_id = 0
        self.objetos = {}
        self.bboxes = {}
        self.sumido_ha = {}
        self.dist_max = dist_max
        self.sumiu_max = sumiu_max

    def _centroide(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def atualizar(self, deteccoes_bbox):
        centroides_novos = [self._centroide(b) for b in deteccoes_bbox]

        if not self.objetos:
            for c, b in zip(centroides_novos, deteccoes_bbox):
                self._registrar(c, b)
            return dict(self.bboxes)

        ids_existentes = list(self.objetos.keys())
        centroides_existentes = [self.objetos[i] for i in ids_existentes]

        usados_novos = set()
        usados_existentes = set()

        pares = []
        for i, ce in enumerate(centroides_existentes):
            for j, cn in enumerate(centroides_novos):
                d = math.dist(ce, cn)
                pares.append((d, i, j))
        pares.sort(key=lambda p: p[0])

        for d, i, j in pares:
            if i in usados_existentes or j in usados_novos:
                continue
            if d > self.dist_max:
                continue
            obj_id = ids_existentes[i]
            self.objetos[obj_id] = centroides_novos[j]
            self.bboxes[obj_id] = deteccoes_bbox[j]
            self.sumido_ha[obj_id] = 0
            usados_existentes.add(i)
            usados_novos.add(j)

        for i, obj_id in enumerate(ids_existentes):
            if i not in usados_existentes:
                self.sumido_ha[obj_id] = self.sumido_ha.get(obj_id, 0) + 1
                if self.sumido_ha[obj_id] > self.sumiu_max:
                    del self.objetos[obj_id]
                    del self.bboxes[obj_id]
                    del self.sumido_ha[obj_id]

        for j, (c, b) in enumerate(zip(centroides_novos, deteccoes_bbox)):
            if j not in usados_novos:
                self._registrar(c, b)

        return dict(self.bboxes)

    def _registrar(self, centroide, bbox):
        self.objetos[self.prox_id] = centroide
        self.bboxes[self.prox_id] = bbox
        self.sumido_ha[self.prox_id] = 0
        self.prox_id += 1


# ----------------------------------------------------------------------------
# EMISSOR DE EVENTOS
# ----------------------------------------------------------------------------

class EmissorEventosTablet:
    def __init__(self, caminho_log):
        self.caminho_log = caminho_log
        self._arquivo = open(caminho_log, "w", encoding="utf-8")

    def emitir(self, tipo, porta, extra=None):
        evento = {
            "timestamp": round(time.time(), 3),
            "porta": porta,
            "tipo": tipo,
        }
        if extra:
            evento.update(extra)
        self._arquivo.write(json.dumps(evento, ensure_ascii=False) + "\n")
        self._arquivo.flush()
        return evento

    def fechar(self):
        self._arquivo.close()


# ----------------------------------------------------------------------------
# MONITOR PRINCIPAL DE UMA PORTA
# ----------------------------------------------------------------------------

class MonitorPorta:
    def __init__(self, zona: ZonaPorta, emissor: EmissorEventosTablet):
        self.zona = zona
        self.emissor = emissor
        self.tracker = RastreadorCentroide()
        self.frames_livre_seguidos = 0
        self.frames_sem_deteccao_valida = 0
        self.estado_anterior = EstadoPorta.LIVRE
        self.ids_ja_avisados_borda = set()

    def processar_frame(self, deteccoes_bbox, deteccao_confiavel=True, objetos_incertos=None):
        objetos = self.tracker.atualizar(deteccoes_bbox)

        if not deteccao_confiavel:
            self.frames_sem_deteccao_valida += 1
        else:
            self.frames_sem_deteccao_valida = 0

        if self.frames_sem_deteccao_valida >= FRAMES_PARA_MODO_INCERTO:
            estado = EstadoPorta.INCERTA
            self.frames_livre_seguidos = 0
            self._transicionar(estado, porta_travada=True)
            return estado, objetos

        contato_borda = False
        ocupada_geral = False

        for obj_id, bbox in objetos.items():
            centro = self.tracker._centroide(bbox)

            if self.zona.bbox_toca_zona(bbox, self.zona.zona_borda):
                contato_borda = True
                if obj_id not in self.ids_ja_avisados_borda:
                    self.ids_ja_avisados_borda.add(obj_id)
                    self.emissor.emitir(
                        "CONTATO_BORDA_DETECTADO",
                        self.zona.nome,
                        {"pessoa_id": obj_id, "alerta": "NAO FECHAR - risco de prensamento"},
                    )
            elif self.zona.bbox_toca_zona(bbox, self.zona.zona_geral):
                ocupada_geral = True

        # Objetos/pessoas de dificil identificacao: travam o fechamento ate
        # serem retirados da zona (em geral pelos proprios passageiros).
        objetos_incertos = objetos_incertos or []
        if any(self.zona.bbox_toca_zona(b, self.zona.zona_borda) for b in objetos_incertos):
            contato_borda = True
        objeto_incerto_na_zona = any(
            self.zona.bbox_toca_zona(b, self.zona.zona_geral) for b in objetos_incertos
        )

        if contato_borda:
            estado = EstadoPorta.CONTATO_BORDA
            self.frames_livre_seguidos = 0
        elif objeto_incerto_na_zona:
            estado = EstadoPorta.OBJETO_INCERTO
            self.frames_livre_seguidos = 0
        elif ocupada_geral:
            estado = EstadoPorta.OCUPADA
            self.frames_livre_seguidos = 0
        else:
            self.frames_livre_seguidos += 1
            if self.frames_livre_seguidos >= FRAMES_PARA_CONFIRMAR_LIVRE:
                estado = EstadoPorta.LIVRE
            else:
                estado = EstadoPorta.OCUPADA

        porta_travada = estado != EstadoPorta.LIVRE
        self._transicionar(estado, porta_travada)
        return estado, objetos

    def _transicionar(self, novo_estado, porta_travada):
        if novo_estado != self.estado_anterior:
            self.emissor.emitir(
                "MUDANCA_ESTADO",
                self.zona.nome,
                {
                    "de": self.estado_anterior.value,
                    "para": novo_estado.value,
                    "porta_travada": porta_travada,
                },
            )
            self.estado_anterior = novo_estado


# ----------------------------------------------------------------------------
# INTERTRAVAMENTO DE PORTA E VEICULO (simulado)
# ----------------------------------------------------------------------------
# Regras:
#  1. O comando de fechar so e aceito com o monitor em LIVRE. Em qualquer outro
#     estado (inclusive INCERTA e OBJETO_INCERTO) a porta fica travada aberta,
#     mesmo que o motorista aperte o botao. Objeto de dificil identificacao
#     trava o fechamento ate ser retirado da zona.
#  2. Alarme sonoro antes de fechar (2 s) e antes de abrir (1,5 s): a porta so
#     se move depois do alarme. Se surgir risco durante o alarme de
#     fechamento, o fechamento e cancelado.
#  3. Se alguem aparecer na zona durante o fechamento, a porta reabre na hora
#     (antiprensamento), com o alarme de abertura soando junto.
#  4. Abertura de emergencia (passageiro preso) nao espera o alarme.
#  5. Com a porta diferente de FECHADA: freio retido e acelerador bloqueado.
#  6. O veiculo so e liberado com a porta FECHADA e a camera confirmando LIVRE
#     por FRAMES_PARA_LIBERAR_VEICULO frames seguidos.
#  7. Ao ligar a ignicao com o freio de porta ("anjo da guarda") inoperante:
#     mensagens vermelhas alternadas no mesmo lugar + alarme de 5 s.
#
# Obs.: aqui e apenas a logica simulada. Em um veiculo real o intertravamento
# precisa ficar em hardware/CLP certificado, independente desta camera.

class MecanicaPorta(Enum):
    ABERTA = "ABERTA"
    AVISO_FECHAR = "AVISO FECHAR"
    FECHANDO = "FECHANDO"
    FECHADA = "FECHADA"
    AVISO_ABRIR = "AVISO ABRIR"
    REABRINDO = "REABRINDO"


MOTIVO_NEGADO = {
    EstadoPorta.CONTATO_BORDA: "contato na borda",
    EstadoPorta.OCUPADA: "passageiro na zona",
    EstadoPorta.INCERTA: "deteccao incerta",
    EstadoPorta.OBJETO_INCERTO: "objeto nao identificado",
}


class AlarmeSonoro:
    def __init__(self, nome_porta, emissor: EmissorEventosTablet, fps=FPS_SIMULACAO, tocar_som=None):
        self.nome = nome_porta
        self.emissor = emissor
        self.fps = fps
        self.tocar_som = tocar_som   # opcional: funcao(tipo, duracao_s) que toca o som de verdade
        self.tipo = None
        self.frames_restantes = 0

    @property
    def ativo(self):
        return self.frames_restantes > 0

    def iniciar(self, tipo, duracao_s):
        self.tipo = tipo
        self.frames_restantes = int(round(duracao_s * self.fps))
        self.emissor.emitir("ALARME_SONORO", self.nome, {"alarme": tipo, "duracao_s": duracao_s})
        if self.tocar_som:
            self.tocar_som(tipo, duracao_s)

    def cancelar(self):
        if self.ativo:
            self.emissor.emitir("ALARME_SONORO_CANCELADO", self.nome, {"alarme": self.tipo})
        self.tipo = None
        self.frames_restantes = 0

    def atualizar(self):
        if self.frames_restantes > 0:
            self.frames_restantes -= 1
            if self.frames_restantes == 0:
                self.tipo = None


class ControladorPortaVeiculo:
    def __init__(self, nome_porta, emissor: EmissorEventosTablet,
                 freio_porta_operante=True, tocar_som=None):
        self.nome = nome_porta
        self.emissor = emissor
        self.porta = MecanicaPorta.ABERTA
        self.progresso = 0.0              # 0.0 = aberta, 1.0 = fechada
        self.frames_aviso = 0             # contagem do alarme antes de fechar/abrir
        self.veiculo_liberado = False     # porta aberta no inicio: veiculo bloqueado
        self.frames_confirmando = 0
        self.alarme = AlarmeSonoro(nome_porta, emissor, tocar_som=tocar_som)
        self.freio_porta_operante = freio_porta_operante
        self.ignicao_ligada = False
        self.frames_desde_ignicao = 0
        self.emissor.emitir(
            "VEICULO_BLOQUEADO", self.nome,
            {"motivo": "PORTA_ABERTA", "freio_retido": True, "acelerador_bloqueado": True},
        )

    @property
    def freio_retido(self):
        return not self.veiculo_liberado

    @property
    def acelerador_bloqueado(self):
        return not self.veiculo_liberado

    # ---------------- ignicao / freio de porta ----------------

    def ligar_ignicao(self):
        self.ignicao_ligada = True
        self.frames_desde_ignicao = 0
        self.emissor.emitir("IGNICAO_LIGADA", self.nome,
                            {"freio_porta_operante": self.freio_porta_operante})
        if not self.freio_porta_operante:
            self.emissor.emitir("FREIO_PORTA_INOPERANTE", self.nome,
                                {"mensagens": list(MENSAGENS_FREIO_PORTA)})
            self.alarme.iniciar("FREIO DE PORTA", ALARME_IGNICAO_S)

    def mensagem_falha_freio(self):
        """Mensagem vermelha da vez (alterna no mesmo lugar) ou None."""
        if not self.ignicao_ligada or self.freio_porta_operante:
            return None
        if (MENSAGEM_FREIO_PORTA_DURACAO_S is not None
                and self.frames_desde_ignicao > MENSAGEM_FREIO_PORTA_DURACAO_S * FPS_SIMULACAO):
            return None
        frames_alternancia = max(1, int(round(SEGUNDOS_ALTERNANCIA_MENSAGEM * FPS_SIMULACAO)))
        # frames_desde_ignicao ja conta o frame atual, entao a primeira mensagem comeca em 0
        frames_passados = max(0, self.frames_desde_ignicao - 1)
        indice = (frames_passados // frames_alternancia) % len(MENSAGENS_FREIO_PORTA)
        return MENSAGENS_FREIO_PORTA[indice]

    # ---------------- comandos do motorista ----------------

    def comando_fechar(self, estado_monitor):
        """Motorista apertou 'fechar'. Retorna (aceito, motivo) ou None se ignorado."""
        if self.porta != MecanicaPorta.ABERTA:
            return None
        if estado_monitor != EstadoPorta.LIVRE:
            self.emissor.emitir(
                "COMANDO_FECHAR_NEGADO", self.nome,
                {"estado_monitor": estado_monitor.value, "porta_travada_aberta": True},
            )
            return False, MOTIVO_NEGADO[estado_monitor]
        self.porta = MecanicaPorta.AVISO_FECHAR
        self.frames_aviso = int(round(ALARME_FECHAMENTO_S * FPS_SIMULACAO))
        self.alarme.iniciar("FECHAMENTO", ALARME_FECHAMENTO_S)
        self.emissor.emitir("COMANDO_FECHAR_ACEITO", self.nome)
        return True, ""

    def comando_abrir(self, emergencia=False):
        """Abrir e sempre permitido. Em emergencia (passageiro preso) nao espera o alarme."""
        if self.porta == MecanicaPorta.AVISO_FECHAR:
            self.porta = MecanicaPorta.ABERTA
            self.frames_aviso = 0
            self.alarme.cancelar()
            self.emissor.emitir("FECHAMENTO_CANCELADO", self.nome, {"motivo": "COMANDO_ABRIR"})
        elif self.porta in (MecanicaPorta.FECHANDO, MecanicaPorta.FECHADA):
            if emergencia or self.porta == MecanicaPorta.FECHANDO:
                self.porta = MecanicaPorta.REABRINDO
                self.alarme.iniciar("ABERTURA", ALARME_ABERTURA_S)
                self.emissor.emitir("ABERTURA_IMEDIATA", self.nome,
                                    {"motivo": "EMERGENCIA" if emergencia else "COMANDO_ABRIR"})
            else:
                self.porta = MecanicaPorta.AVISO_ABRIR
                self.frames_aviso = int(round(ALARME_ABERTURA_S * FPS_SIMULACAO))
                self.alarme.iniciar("ABERTURA", ALARME_ABERTURA_S)
                self.emissor.emitir("COMANDO_ABRIR_ACEITO", self.nome)

    # ---------------- atualizacao por frame ----------------

    def atualizar(self, estado_monitor):
        """Chamar uma vez por frame, depois do monitor."""
        self.alarme.atualizar()
        if self.ignicao_ligada:
            self.frames_desde_ignicao += 1

        if self.porta == MecanicaPorta.AVISO_FECHAR:
            if estado_monitor != EstadoPorta.LIVRE:
                self.porta = MecanicaPorta.ABERTA
                self.frames_aviso = 0
                self.alarme.cancelar()
                self.emissor.emitir("FECHAMENTO_CANCELADO", self.nome,
                                    {"estado_monitor": estado_monitor.value})
            else:
                self.frames_aviso -= 1
                if self.frames_aviso <= 0:
                    self.porta = MecanicaPorta.FECHANDO
        elif self.porta == MecanicaPorta.AVISO_ABRIR:
            self.frames_aviso -= 1
            if self.frames_aviso <= 0:
                self.porta = MecanicaPorta.REABRINDO
        elif self.porta == MecanicaPorta.FECHANDO:
            if estado_monitor != EstadoPorta.LIVRE:
                self.porta = MecanicaPorta.REABRINDO
                self.alarme.iniciar("ABERTURA", ALARME_ABERTURA_S)
                self.emissor.emitir(
                    "REVERSAO_ANTIPRENSAMENTO", self.nome,
                    {"estado_monitor": estado_monitor.value, "progresso": round(self.progresso, 2)},
                )
            else:
                self.progresso = min(1.0, self.progresso + 1.0 / FRAMES_PARA_FECHAR_PORTA)
                if self.progresso >= 1.0:
                    self.porta = MecanicaPorta.FECHADA
                    self.emissor.emitir("PORTA_FECHADA", self.nome)
        elif self.porta == MecanicaPorta.REABRINDO:
            self.progresso = max(0.0, self.progresso - 2.0 / FRAMES_PARA_FECHAR_PORTA)
            if self.progresso <= 0.0:
                self.porta = MecanicaPorta.ABERTA
                self.emissor.emitir("PORTA_ABERTA", self.nome)

        self._atualizar_veiculo(estado_monitor)

    def _atualizar_veiculo(self, estado_monitor):
        if self.porta != MecanicaPorta.FECHADA:
            self.frames_confirmando = 0
            if self.veiculo_liberado:
                self.veiculo_liberado = False
                self.emissor.emitir(
                    "VEICULO_BLOQUEADO", self.nome,
                    {"motivo": "PORTA_" + self.porta.name,
                     "freio_retido": True, "acelerador_bloqueado": True},
                )
            return

        if self.veiculo_liberado:
            return

        if estado_monitor == EstadoPorta.LIVRE:
            self.frames_confirmando += 1
            if self.frames_confirmando >= FRAMES_PARA_LIBERAR_VEICULO:
                self.veiculo_liberado = True
                self.emissor.emitir(
                    "VEICULO_LIBERADO", self.nome,
                    {"freio_retido": False, "acelerador_bloqueado": False},
                )
        else:
            self.frames_confirmando = 0


# ----------------------------------------------------------------------------
# DETECTOR PARA VÍDEO REAL
# ----------------------------------------------------------------------------

def criar_detector_hog():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detectar_pessoas_hog(hog, frame):
    """Retorna (pessoas_confiaveis, objetos_incertos).

    Deteccoes com pontuacao baixa do SVM sao tratadas como objeto/pessoa de
    dificil identificacao e travam o fechamento da porta.
    """
    retangulos, pesos = hog.detectMultiScale(
        frame, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    pesos = np.asarray(pesos, dtype=float).reshape(-1)
    confiaveis, incertos = [], []
    for (x, y, w, h), peso in zip(retangulos, pesos):
        bbox = (int(x), int(y), int(x + w), int(y + h))
        if peso >= LIMIAR_CONFIANCA_PESSOA_HOG:
            confiaveis.append(bbox)
        else:
            incertos.append(bbox)
    return confiaveis, incertos


# ----------------------------------------------------------------------------
# DESENHO / OVERLAY
# ----------------------------------------------------------------------------

COR_ESTADO = {
    EstadoPorta.LIVRE: (210, 100, 0),
    EstadoPorta.OCUPADA: (0, 165, 255),
    EstadoPorta.CONTATO_BORDA: (0, 0, 255),
    EstadoPorta.INCERTA: (0, 0, 200),
    EstadoPorta.OBJETO_INCERTO: (160, 40, 160),
}

TEXTO_ESTADO = {
    EstadoPorta.LIVRE: "PORTA SEGURA PARA O FECHAMENTO",
    EstadoPorta.OCUPADA: "AGUARDANDO - PASSAGEIRO NA ZONA",
    EstadoPorta.CONTATO_BORDA: "PERIGO - CONTATO NA BORDA! NAO FECHAR",
    EstadoPorta.INCERTA: "DETECCAO INCERTA - FAIL-SAFE ATIVO",
    EstadoPorta.OBJETO_INCERTO: "OBJETO NAO IDENTIFICADO - RETIRE DA ZONA",
}


def desenhar_overlay(frame, zona: ZonaPorta, objetos, estado: EstadoPorta, info_extra="", objetos_incertos=()):
    saida = frame.copy()

    x1, y1, x2, y2 = zona.zona_geral
    bx1, by1, bx2, by2 = zona.zona_borda

    # 1. QUADRANTE AZUL: Interior do Ônibus (desenhado do início da tela até a porta)
    cv2.rectangle(saida, (0, y1), (x1, y2), (180, 80, 40), -1) # Fundo azul (BGR)
    cv2.putText(saida, "ONIBUS", (8, y1 + int((y2 - y1) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. ZONA GERAL (Amarela)
    cv2.rectangle(saida, (x1, y1), (x2, y2), (0, 255, 255), 1)
    # Movido para a direita, ao lado da linha da borda, para não encavalar
    cv2.putText(saida, "zona de seguranca", (bx2 + 8, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # 3. ZONA DE BORDA (Vermelha)
    cv2.rectangle(saida, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
    # Movido um pouco mais para cima para dar respiro
    cv2.putText(saida, "borda critica", (bx1, y1 - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Identificadores de Pessoas
    for obj_id, bbox in objetos.items():
        px1, py1, px2, py2 = bbox
        cv2.rectangle(saida, (px1, py1), (px2, py2), (255, 255, 255), 2)
        cv2.putText(saida, f"ID {obj_id}", (px1, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Objetos de dificil identificacao
    for bbox in objetos_incertos:
        px1, py1, px2, py2 = bbox
        cv2.rectangle(saida, (px1, py1), (px2, py2), (200, 60, 200), 2)
        cv2.putText(saida, "?", (px1, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 60, 200), 1, cv2.LINE_AA)

    # Barra de Status Superior
    cor = COR_ESTADO[estado]
    cv2.rectangle(saida, (0, 0), (FRAME_W, 36), cor, -1)
    cv2.putText(saida, TEXTO_ESTADO[estado], (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    if info_extra:
        cv2.putText(saida, info_extra, (10, FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return saida


COR_PORTA = {
    MecanicaPorta.ABERTA: (0, 140, 255),
    MecanicaPorta.AVISO_FECHAR: (0, 100, 220),
    MecanicaPorta.FECHANDO: (0, 165, 255),
    MecanicaPorta.FECHADA: (0, 150, 0),
    MecanicaPorta.AVISO_ABRIR: (0, 100, 220),
    MecanicaPorta.REABRINDO: (0, 0, 200),
}


def desenhar_porta_mecanica(frame, zona: ZonaPorta, controlador: ControladorPortaVeiculo):
    """Painel cinza que fecha por cima da zona de borda."""
    if controlador.progresso <= 0.0:
        return
    bx1, by1, bx2, by2 = zona.zona_borda
    largura = int((bx2 - bx1) * controlador.progresso)
    cv2.rectangle(frame, (bx1, by1), (bx1 + largura, by2), (170, 170, 170), -1)


def desenhar_painel_veiculo(frame, controlador: ControladorPortaVeiculo, mensagem_comando=None):
    caixas = [
        (f"PORTA: {controlador.porta.value}", COR_PORTA[controlador.porta]),
        ("FREIO: RETIDO" if controlador.freio_retido else "FREIO: LIBERADO",
         (0, 0, 200) if controlador.freio_retido else (0, 150, 0)),
        ("ACELERADOR: BLOQUEADO" if controlador.acelerador_bloqueado else "ACELERADOR: LIBERADO",
         (0, 0, 200) if controlador.acelerador_bloqueado else (0, 150, 0)),
    ]
    for k, (texto, cor) in enumerate(caixas):
        x = 10 + k * 212
        cv2.rectangle(frame, (x, 408), (x + 202, 434), cor, -1)
        cv2.putText(frame, texto, (x + 8, 426), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if mensagem_comando:
        texto, cor = mensagem_comando
        cv2.putText(frame, texto, (10, 452), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, cor, 1, cv2.LINE_AA)

    if controlador.alarme.ativo:
        cv2.putText(frame, f"ALARME: {controlador.alarme.tipo}", (430, 452),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)


def desenhar_aviso_falha_freio(frame, mensagem):
    """Mensagem vermelha logo abaixo da barra de status; as duas mensagens alternam no mesmo lugar."""
    if not mensagem:
        return
    cv2.rectangle(frame, (0, 40), (FRAME_W, 76), (0, 0, 200), -1)
    cv2.rectangle(frame, (0, 40), (FRAME_W, 76), (255, 255, 255), 1)
    (largura, _), _ = cv2.getTextSize(mensagem, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, mensagem, ((FRAME_W - largura) // 2, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# MODO 1: SIMULAÇÃO SINTÉTICA (Aprimorada com Profundidade 2.5D)
# ----------------------------------------------------------------------------

@dataclass
class Passageiro:
    x: float
    y: float
    vx: float
    vy: float
    raio: int
    cor: tuple
    rotulo: str
    vai_encostar_borda: bool = False
    ativo: bool = True

    def bbox(self):
        return (int(self.x - self.raio), int(self.y - self.raio),
                int(self.x + self.raio), int(self.y + self.raio))

    def mover(self):
        self.x += self.vx
        self.y += self.vy


def gerar_cenario_simulado():
    # Coordenadas Y separadas e novas cores para melhor contraste
    passageiros = [
        Passageiro(x=40, y=210, vx=3.0, vy=0, raio=26,
                   cor=(180, 200, 50), rotulo="Passageiro comum"),
        Passageiro(x=120, y=250, vx=0.15, vy=0, raio=25,
                   cor=(50, 50, 220), rotulo="Segurando a porta",
                   vai_encostar_borda=True),
        Passageiro(x=550, y=290, vx=-6.0, vy=0, raio=27,
                   cor=(220, 100, 255), rotulo="Evasor"),
    ]
    return passageiros


def rodar_simulacao(caminho_video_saida, caminho_log, freio_porta_operante=True):
    zona = ZonaPorta(
        nome="Porta traseira",
        # ZONA GERAL REDUZIDA - O espaço monitorado fora do ônibus é bem menor agora
        zona_geral=(70, 120, 350, 400),
        zona_borda=(70, 120, 130, 400),
    )
    emissor = EmissorEventosTablet(caminho_log)
    monitor = MonitorPorta(zona, emissor)
    controlador = ControladorPortaVeiculo(zona.nome, emissor, freio_porta_operante=freio_porta_operante)
    mensagem_comando, mensagem_ate = None, -1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(caminho_video_saida, fourcc, FPS_SIMULACAO, (FRAME_W, FRAME_H))

    passageiros = gerar_cenario_simulado()
    # Aumentado para 400 frames para compensar a zona menor e a entrada tardia do evasor
    total_frames = 340   # inclui os 2 s de alarme antes de fechar a porta
    frame_incerto_inicio, frame_incerto_fim = 80, 120

    for i in range(total_frames):
        frame = np.full((FRAME_H, FRAME_W, 3), (40, 40, 40), dtype=np.uint8)
        cv2.rectangle(frame, (0, 400), (FRAME_W, FRAME_H), (25, 25, 25), -1)

        # 1. ATUALIZAR POSIÇÕES (Update)
        deteccoes = []
        passageiros_visiveis = []

        for p in passageiros:
            # Evasor entra apenas quando a porta estiver livre de fato (após o frame 280)
            if p.rotulo.startswith("Evasor") and i < 160:
                continue
            if not p.ativo:
                continue

            if p.vai_encostar_borda:
                if i < 200:
                    p.x = 132 + math.sin(i / 6) * 4
                else:
                    p.x += 4.5
                    if p.x > FRAME_W + 50:
                        p.ativo = False
            else:
                p.mover()
                if p.x < -50 or p.x > FRAME_W + 50:
                    p.ativo = False

            if p.ativo:
                deteccoes.append(p.bbox())
                passageiros_visiveis.append(p)

        # Processamento da lógica da IA (Bounding boxes)
        deteccao_confiavel = not (frame_incerto_inicio <= i <= frame_incerto_fim)
        info_extra = "SIMULACAO - camera embacada" if not deteccao_confiavel else ""
        estado, objetos = monitor.processar_frame(deteccoes, deteccao_confiavel)

        # Ignicao ligada no primeiro frame (testa o aviso do freio de porta)
        if i == 0:
            controlador.ligar_ignicao()

        # Motorista aperta "fechar": so funciona se a camera confirmar LIVRE
        if i in COMANDOS_FECHAR_MOTORISTA:
            resultado = controlador.comando_fechar(estado)
            if resultado is not None:
                aceito, motivo = resultado
                if aceito:
                    mensagem_comando = ("COMANDO FECHAR: ACEITO - alarme antes de fechar", (0, 220, 0))
                else:
                    mensagem_comando = (f"COMANDO FECHAR: NEGADO - {motivo}", (80, 80, 255))
                mensagem_ate = i + 25
        controlador.atualizar(estado)

        # Desenha a zona e o status ANTES dos passageiros para não cobri-los
        frame = desenhar_overlay(frame, zona, objetos, estado, info_extra)
        desenhar_porta_mecanica(frame, zona, controlador)
        desenhar_painel_veiculo(frame, controlador, mensagem_comando if i <= mensagem_ate else None)
        desenhar_aviso_falha_freio(frame, controlador.mensagem_falha_freio())

        # 2. DESENHAR PASSAGEIROS COM PROFUNDIDADE (Z-Sorting / Painter's Algorithm)
        # Ordena pelo Y: quem tem Y menor (mais ao fundo) é desenhado primeiro.
        passageiros_visiveis.sort(key=lambda p: p.y)

        for p in passageiros_visiveis:
            px, py = int(p.x), int(p.y)

            # A. Sombra no chão (Elipse escura deslocada para baixo)
            cv2.ellipse(frame, (px, py + p.raio - 5), (p.raio + 4, p.raio - 6), 0, 0, 360, (20, 20, 20), -1)

            # B. Ombro/Corpo (Círculo principal)
            cv2.circle(frame, (px, py), p.raio, p.cor, -1)
            cv2.circle(frame, (px, py), p.raio, (255, 255, 255), 1) # Contorno

            # C. Cabeça (Círculo menor deslocado levemente para cima, com cor mais clara)
            cor_cabeca = (min(255, p.cor[0] + 60), min(255, p.cor[1] + 60), min(255, p.cor[2] + 60))
            cv2.circle(frame, (px, py - 6), int(p.raio * 0.55), cor_cabeca, -1)
            cv2.circle(frame, (px, py - 6), int(p.raio * 0.55), (200, 200, 200), 1)

            # Rótulo de texto flutuando acima da cabeça
            cv2.putText(frame, p.rotulo, (px - p.raio, py - p.raio - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, p.cor, 1, cv2.LINE_AA)

        cv2.putText(frame, f"frame {i+1}/{total_frames}", (FRAME_W - 150, FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        video.write(frame)

    video.release()
    emissor.fechar()


# ----------------------------------------------------------------------------
# MODO 2: PROCESSAR VÍDEO REAL COM HOG
# ----------------------------------------------------------------------------

def rodar_em_video_real(caminho_entrada, caminho_video_saida, caminho_log):
    cap = cv2.VideoCapture(caminho_entrada)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir o vídeo: {caminho_entrada}")

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_W
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    zona = ZonaPorta(
        nome="Porta (video real)",
        zona_geral=(int(largura * 0.05), int(altura * 0.15),
                    int(largura * 0.95), int(altura * 0.95)),
        zona_borda=(int(largura * 0.05), int(altura * 0.15),
                    int(largura * 0.20), int(altura * 0.95)),
    )
    emissor = EmissorEventosTablet(caminho_log)
    monitor = MonitorPorta(zona, emissor)
    hog = criar_detector_hog()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(caminho_video_saida, fourcc, fps, (largura, altura))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        bboxes, incertos = detectar_pessoas_hog(hog, frame)
        estado, objetos = monitor.processar_frame(
            bboxes, deteccao_confiavel=True, objetos_incertos=incertos
        )
        saida = desenhar_overlay(frame, zona, objetos, estado, objetos_incertos=incertos)
        video.write(saida)

    cap.release()
    video.release()
    emissor.fechar()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Protótipo de sistema de segurança de portas de ônibus com IA"
    )
    sub = parser.add_subparsers(dest="modo", required=True)

    p_sim = sub.add_parser("simulate", help="Gera um vídeo de demonstração sintético")
    p_sim.add_argument("--saida", default="demo_portas_onibus.mp4")
    p_sim.add_argument("--log", default="eventos_portas.jsonl")
    p_sim.add_argument("--freio-porta-rompido", action="store_true",
                       help="Simula freio de porta (anjo da guarda) inoperante ao ligar o veiculo")

    p_vid = sub.add_parser("video", help="Processa um vídeo real com detector HOG")
    p_vid.add_argument("--input", required=True)
    p_vid.add_argument("--saida", default="resultado_processado.mp4")
    p_vid.add_argument("--log", default="eventos_portas.jsonl")

    args = parser.parse_args()

    if args.modo == "simulate":
        print("Gerando simulação sintética com profundidade 2.5D...")
        rodar_simulacao(args.saida, args.log, freio_porta_operante=not args.freio_porta_rompido)
        print(f"Vídeo salvo em: {args.saida}")
        print(f"Log de eventos salvo em: {args.log}")
    elif args.modo == "video":
        print(f"Processando vídeo real: {args.input}")
        rodar_em_video_real(args.input, args.saida, args.log)
        print(f"Vídeo processado salvo em: {args.saida}")
        print(f"Log de eventos salvo em: {args.log}")


if __name__ == "__main__":
    main()
