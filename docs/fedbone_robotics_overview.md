# Overview: FedBone aplicado a robotica

## Ideia central do artigo

O FedBone propoe um framework de Federated Multi-Task Learning heterogeneo para treinar um modelo geral grande sem exigir que cada cliente execute esse modelo localmente. A arquitetura divide o aprendizado em duas partes:

- **Cliente/edge:** calcula o patch embedding, executa o modulo de adaptacao da tarefa e aplica o head especifico da tarefa.
- **Servidor/cloud:** executa o general model, isto e, o backbone grande compartilhado entre tarefas.

Esse desenho combina split learning e federated learning. O cliente envia embeddings intermediarios ao servidor, recebe features gerais, calcula a perda da sua tarefa e devolve gradientes intermediarios. O servidor agrega os gradientes do general model e atualiza o conhecimento compartilhado.

## Por que isso combina com robotica

Em robotica distribuida, cada robo pode operar em um ambiente diferente, com sensores, objetivos e distribuicoes de dados diferentes. Isso cria dois problemas parecidos com os tratados no artigo:

- **Restricao de recurso no robo:** o robo pode nao ter GPU/memoria suficiente para treinar um backbone grande.
- **Heterogeneidade de tarefas:** diferentes robos podem fazer grasping, navegacao, percepcao de cena, estimativa de profundidade, deteccao de falhas ou classificacao de estado.

No contexto deste projeto, o FedBone pode ser interpretado assim:

- O robo executa a parte leve do modelo, preservando dados locais e reduzindo custo computacional embarcado.
- A nuvem executa o modelo geral, que aprende representacoes compartilhadas entre robos e tarefas.
- A agregacao por projecao de gradientes reduz interferencia quando tarefas ou ambientes empurram o modelo para direcoes conflitantes.

## Como o artigo mede o FedBone

O artigo avalia o FedBone em tarefas heterogeneas de visao e saude. As metricas principais sao:

- **Segmentacao semantica:** mIoU.
- **Estimativa de profundidade:** RMSE.
- **Estimativa de normais:** mErr.
- **Deteccao de bordas:** odsF.
- **Saliencia:** maxF.
- **Classificacao oftalmologica:** acuracia por tarefa.

Essas metricas medem qualidade preditiva, mas nao capturam completamente o impacto operacional em robotica.

## Metricas adicionais para robotica

A apresentacao propoe complementar as metricas do artigo com medidas mais ligadas a execucao fisica, eficiencia federada e cloud robotics:

- **Task Success Rate (TSR):** mede se a tarefa robotica foi concluida com sucesso.
- **Erro de posicionamento em mm:** mede precisao em grasping e pick-and-place.
- **Rounds ate convergencia:** mede eficiencia do treinamento federado.
- **Latencia de inferencia end-to-end:** mede viabilidade de execucao com parte do modelo na nuvem.
- **RMSE cross-client/cross-cluster:** mede transferencia entre robos, ambientes ou clusters.
- **RMSE e TSR por cluster:** medem personalizacao dos modelos por grupos de clientes.
- **Bytes transmitidos por round:** mede custo de comunicacao.

## Estado atual do repositorio

O projeto usa exclusivamente o RoboMimic low-dimensional para imitation learning
de tarefas de manipulacao. Ele contem:

- FedAvg baseline.
- Clustered FL.
- FedBone com general model LSTM no servidor.
- Task adaptation no cliente.
- GP Aggregation para conflitos de gradiente.

No protocolo offline, TSR e a proporcao de acoes preditas cujo erro vetorial
fica abaixo de um limiar. No RoboSuite, o avaliador online registra o sucesso
real da tarefa reportado pelo ambiente.

## Proximos passos tecnicos

1. Executar os tres metodos no mesmo conjunto de arquivos e sementes.
2. Avaliar online os checkpoints de baseline no RoboSuite.
3. Reportar media e desvio entre varias sementes.
4. Comparar custo de comunicacao e latencia de inferencia entre os metodos.
