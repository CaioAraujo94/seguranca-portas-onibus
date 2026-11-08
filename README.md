# Sistema de Segurança de Portas de Ônibus

Protótipo em **Python + OpenCV** que monitora a zona da porta de um ônibus com visão computacional e decide se é seguro fechá-la. Se houver risco, a porta **fica travada aberta mesmo que o motorista aperte o botão**, e o veículo continua com freio retido e acelerador bloqueado até a porta fechar e a câmera confirmar que não há ninguém na zona.

!\[Demonstração do sistema](docs/demo.gif)

> ⚠️ \*\*Projeto de estudo e prototipagem.\*\* Não foi certificado nem testado em veículos reais e não deve ser usado como sistema de segurança. A parte do veículo (freio, acelerador, alarme) é lógica simulada.

##  Motivação

A ideia nasceu da rotina de quem dirige ônibus: na hora de fechar a porta, o motorista depende do espelho para saber se alguém ficou na borda. O projeto testa como uma câmera pode ajudar nessa decisão, sempre assumindo o pior quando houver dúvida (**fail-safe**).

##  Como funciona

### Zonas monitoradas

* **Zona de segurança**: área ao redor da porta (retângulo amarelo).
* **Borda crítica**: faixa junto ao fechamento da porta, onde há risco de prensamento (retângulo vermelho).

Cada pessoa detectada recebe um ID e é acompanhada entre os frames por um rastreador de centroides simples.

### Estados da porta (barra no topo)

|Estado|Barra|Significado|
|-|-|-|
|`CONTATO\_BORDA`|vermelha|Alguém tocando a borda crítica: **não fechar**|
|`OCUPADA`|laranja|Passageiro na zona de segurança|
|`OBJETO\_INCERTO`|roxa|Objeto ou pessoa de difícil identificação na zona|
|`INCERTA`|vermelha escura|Detecção não confiável (câmera suja, embaçada etc.): **fail-safe ativo**|
|`LIVRE`|azul|"Porta segura para o fechamento"|

O estado só vira `LIVRE` depois de alguns frames seguidos com a zona vazia (`FRAMES\_PARA\_CONFIRMAR\_LIVRE`), para evitar piscadas.

### Intertravamento de porta e veículo

1. **Fechar só com `LIVRE`.** Em qualquer outro estado (inclusive `INCERTA` e `OBJETO\_INCERTO`) o comando é negado e a porta continua aberta.
2. **Alarme antes de mover a porta:** 2 s antes de fechar e 1,5 s antes de abrir. Se surgir risco durante o alarme de fechamento, o fechamento é cancelado.
3. **Antiprensamento:** se alguém entrar na zona durante o fechamento, a porta reabre na hora, com o alarme de abertura soando junto.
4. **Abertura de emergência** (passageiro preso) não espera o alarme.
5. **Porta diferente de `FECHADA`** → freio retido e acelerador bloqueado.
6. **Liberação do veículo** só com a porta `FECHADA` e a câmera confirmando `LIVRE` por alguns frames seguidos.
7. **Freio de porta ("anjo da guarda") inoperante:** ao ligar a ignição, aparecem em vermelho, alternadas no mesmo lugar, as mensagens *"Freio de porta inoperante"* e *"Visitar a oficina"*, com alarme de 5 s.

### O que a demonstração mostra

|Tempo|O que acontece|
|-|-|
|0 – 4 s|Passageiro segurando a porta na borda crítica; os pedidos de fechar do motorista são **negados**|
|4 – 6 s|Câmera "embaçada": detecção incerta, fail-safe ativo, pedido negado|
|6 – 12 s|Um segundo passageiro corre para dentro do ônibus enquanto o primeiro ainda está na zona; pedidos negados|
|\~12,8 s|Zona vazia: barra azul "porta segura para o fechamento"|
|\~13 s|Motorista aperta fechar: **aceito**, alarme de 2 s|
|\~15 s|Porta fecha e, com a câmera confirmando, freio e acelerador são **liberados**|

##  Tecnologias

Python 3.8+ · OpenCV · NumPy

##  Como rodar

```bash
git clone https://github.com/<CaioAraujo94>/<seguranca-portas-onibus>.git
cd <seguranca-portas-onibus>
pip install -r requirements.txt
```

**Modo simulação** (gera um vídeo sintético e o log de eventos):

```bash
python door\_safety\_system.py simulate --saida demo.mp4 --log eventos.jsonl
```

Para simular o freio de porta inoperante ao ligar o veículo:

```bash
python door\_safety\_system.py simulate --freio-porta-rompido
```

**Modo vídeo** (experimental: detecta pessoas em um vídeo real com HOG + SVM):

```bash
python door\_safety\_system.py video --input meu\_video.mp4 --saida resultado.mp4 --log eventos.jsonl
```

> O vídeo gerado usa o codec `mp4v`, que pode não abrir em alguns players. Se isso acontecer: `ffmpeg -i demo.mp4 -c:v libx264 -pix\_fmt yuv420p saida.mp4`.

##  Log de eventos

Cada decisão é gravada em um arquivo `.jsonl` (um JSON por linha), útil como evidência depois:

```json
{"timestamp": 1730000000.0, "porta": "Porta traseira", "tipo": "COMANDO\_FECHAR\_NEGADO", "estado\_monitor": "CONTATO\_BORDA", "porta\_travada\_aberta": true}
{"timestamp": 1730000012.9, "porta": "Porta traseira", "tipo": "ALARME\_SONORO", "alarme": "FECHAMENTO", "duracao\_s": 2.0}
{"timestamp": 1730000015.9, "porta": "Porta traseira", "tipo": "VEICULO\_LIBERADO", "freio\_retido": false, "acelerador\_bloqueado": false}
```

Alguns tipos de evento: `MUDANCA\_ESTADO`, `CONTATO\_BORDA\_DETECTADO`, `COMANDO\_FECHAR\_ACEITO`, `COMANDO\_FECHAR\_NEGADO`, `FECHAMENTO\_CANCELADO`, `REVERSAO\_ANTIPRENSAMENTO`, `PORTA\_FECHADA`, `VEICULO\_BLOQUEADO`, `VEICULO\_LIBERADO`, `ALARME\_SONORO`, `FREIO\_PORTA\_INOPERANTE`.

##  Parâmetros principais

No início de `door\_safety\_system.py`:

|Constante|Padrão|O que controla|
|-|-|-|
|`FRAMES\_PARA\_CONFIRMAR\_LIVRE`|5|Frames com a zona vazia para voltar a `LIVRE`|
|`FRAMES\_PARA\_MODO\_INCERTO`|8|Frames de detecção ruim até entrar em fail-safe|
|`ALARME\_FECHAMENTO\_S` / `ALARME\_ABERTURA\_S`|2,0 / 1,5|Duração dos alarmes antes de mover a porta|
|`ALARME\_IGNICAO\_S`|5,0|Alarme ao ligar com o freio de porta inoperante|
|`FRAMES\_PARA\_LIBERAR\_VEICULO`|6|Confirmação da câmera para liberar freio e acelerador|
|`LIMIAR\_CONFIANCA\_PESSOA\_HOG`|0,5|Abaixo disso, a detecção vira "objeto incerto" (precisa de calibração)|

##  Limitações

* **Lógica simulada.** Num veículo real, o intertravamento teria de ficar em hardware certificado, independente da câmera.
* **Áudio simulado:** os alarmes aparecem no log e na tela. A classe `AlarmeSonoro` aceita uma função `tocar\_som` para tocar som de verdade.
* **HOG + SVM** é uma abordagem clássica: perde precisão com pouca luz, oclusão e ângulos difíceis, e só enxerga pessoas. Carrinho de bebê, cadeira de rodas e mochila só travam o fechamento se forem detectados como "incertos".
* **No modo `video`**, a porta mecânica e o intertravamento não são simulados e o fail-safe de câmera (`INCERTA`) não dispara, porque nada mede a saúde da câmera. As zonas são proporções fixas do quadro.
* A parte do HOG precisa do **OpenCV 4.x** (por isso o `requirements.txt` limita a versão).
* Testado apenas com a simulação sintética.

##  Próximos passos

* \[ ] **Watchdog:** bloquear a porta se a câmera parar de enviar atualizações
* \[ ] **Saúde da câmera:** medir nitidez, brilho e frames repetidos para acionar o fail-safe no modo `video`
* \[ ] Detector mais moderno (ex.: YOLO) para pessoas e objetos, comparando falsos positivos e negativos
* \[ ] Alarme sonoro real e liberação gradual do freio (subidas)
* \[ ] Separar o código em módulos (detecção, zonas, decisão, intertravamento) e adicionar testes automatizados

##  Autor

CaioAraujo · www.linkedin.com/in/caio-henrique-mendes-araujo-1443ba18b

