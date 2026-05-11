# FedBone-Robotics

Implementação de Federated Learning multi-tarefa para sistemas robóticos em nuvem. O projeto combina **Clustered FL** e **FedBone** (split learning com projeção de gradientes) usando o dataset UCI HAR como proxy de dados sensoriais robóticos.

---

## Motivação

Frotas robóticas operam em ambientes distribuídos, com dados heterogêneos e recursos computacionais limitados. Federated Learning permite treinar modelos colaborativos sem centralizar os dados, preservando privacidade e reduzindo custos de comunicação. Este projeto explora esse cenário simulando robôs como clientes FL com distribuição não-IID dos dados.

---

## Abordagens Implementadas

**FedAvg (baseline)**
Agregação federada padrão. O servidor inicializa um modelo global, distribui aos clientes, cada cliente treina localmente e retorna os pesos, que são agregados via média ponderada.

**Clustered FL**
Clientes são agrupados por similaridade de modelo (K-Means ou hierárquico) antes da agregação. Cada cluster mantém seu próprio modelo global, permitindo especialização por tipo de tarefa. O reagrupamento pode ocorrer em intervalos configuráveis durante o treinamento.

**FedBone**
Arquitetura de split learning combinada com agregação por projeção de gradientes (GP Aggregation). O modelo é dividido entre cliente e servidor:

- Cliente (robô): executa o Patch Embedding, o módulo de Task Adaptation e o Task Head
- Servidor (nuvem): executa o General Model (LSTM bidirecional de grande escala)

O fluxo por round é:
1. O cliente computa embeddings locais e os envia ao servidor
2. O servidor extrai features com o General Model e as retorna ao cliente
3. O cliente completa o forward pass, computa a loss e envia os gradientes ao servidor
4. O servidor agrega os gradientes via GP Aggregation e atualiza o General Model

**GP Aggregation** resolve conflitos entre gradientes de tarefas heterogêneas: escala os gradientes com atenção ao histórico agregado, projeta gradientes conflitantes no plano normal do gradiente oposto e realiza média ponderada dos gradientes resultantes.

---

## Estrutura do Projeto

```
FedBone-robotics/
├── config/
│   └── config.py               # Hiperparâmetros e configurações globais
├── data/
│   ├── download_har.py         # Download do UCI HAR Dataset
│   ├── dataset_loader.py       # Split não-IID via distribuição Dirichlet
│   └── multitask_loader.py     # Criação de múltiplas tarefas a partir do HAR
├── models/
│   ├── model.py                # Modelo LSTM baseline
│   └── fedbone_model.py        # Arquitetura FedBone (cliente e servidor)
├── federated/
│   ├── baseline_fl.py          # FedAvg via Flower
│   ├── clustered_fl.py         # Clustered FL
│   ├── fedbone_fl.py           # Loop de treinamento FedBone
│   └── gp_aggregation.py       # GP Aggregation
├── runner/
│   ├── run_experiments.py      # Executa FedAvg e Clustered FL
│   └── run_fedbone.py          # Executa FedBone
└── utils/
    ├── utils.py                # Métricas e helpers gerais
    ├── visualization.py        # Plots para FedAvg e Clustered FL
    └── fedbone_visualization.py # Plots para FedBone
```

---

## Instalação

```bash
git clone https://github.com/seu-usuario/FedBone-robotics.git
cd FedBone-robotics
pip install -r requirements.txt
python data/download_har.py
```

---

## Uso

Executar FedAvg e Clustered FL:
```bash
python runner/run_experiments.py
```

Executar FedBone:
```bash
python runner/run_fedbone.py
```

Gerar visualizações:
```python
from utils.fedbone_visualization import create_all_fedbone_plots
create_all_fedbone_plots("results/")

from utils.visualization import create_all_plots
create_all_plots("results/")
```

---

## Configuração

Os principais parâmetros estão em `config/config.py`:

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `NUM_ROBOTS` | 10 | Número de clientes |
| `NUM_ROUNDS` | 50 | Rounds de comunicação |
| `CLIENTS_PER_ROUND` | 5 | Clientes selecionados por round |
| `LOCAL_EPOCHS` | 5 | Épocas locais por round |
| `ALPHA` | 0.5 | Heterogeneidade Dirichlet (menor = mais heterogêneo) |
| `NUM_CLUSTERS` | 3 | Clusters para Clustered FL |
| `EMBED_DIM` | 64 | Dimensão do embedding no cliente |
| `GENERAL_HIDDEN` | 128 | Hidden size do General Model no servidor |
| `USE_GP_AGGREGATION` | True | Habilitar GP Aggregation no FedBone |

---

## Referências

- Zaland et al., *Federated Learning for Large-Scale Cloud Robotic Manipulation: Opportunities and Challenges*, arXiv:2507.17903, 2025
- Chen et al., *FedBone: Towards Large-Scale Federated Multi-Task Learning*, arXiv:2306.17465, 2023
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS, 2017

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
