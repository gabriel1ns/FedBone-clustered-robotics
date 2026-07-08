# FedBone-Robotics

Implementação de Federated Learning multi-tarefa para sistemas robóticos em nuvem. O projeto compara **FedAvg**, **Clustered FL** e **FedBone** (split learning com projeção de gradientes) no RoboMimic, usando imitation learning multi-tarefa.

---

## Motivação

Frotas robóticas operam em ambientes distribuídos, com dados heterogêneos e recursos computacionais limitados. Federated Learning permite treinar modelos colaborativos sem centralizar os dados, preservando privacidade e reduzindo custos de comunicação. Este projeto explora esse cenário simulando robôs como clientes FL com distribuição não-IID dos dados.

---

## Abordagens Implementadas

**FedAvg (baseline)**
Agregação federada padrão. O servidor inicializa um modelo global, distribui aos clientes, cada cliente treina localmente e retorna os pesos, que são agregados via média ponderada.

**Clustered FL**
Dentro de cada tarefa, clientes são agrupados pela distribuição local de observações e ações (K-Means ou hierárquico). Cada cluster mantém seu próprio modelo, permitindo especialização por perfil de demonstração. O reagrupamento pode ocorrer em intervalos configuráveis.

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

## Overview da Pesquisa

O documento [docs/fedbone_robotics_overview.md](docs/fedbone_robotics_overview.md) resume o artigo FedBone e adapta a proposta para o contexto de robótica em nuvem, incluindo as métricas do artigo e as métricas adicionais necessárias para avaliar sistemas robóticos.

RoboMimic é usado como dataset robótico offline: o loader lê demonstrações HDF5, transforma observações low-dimensional em entradas e ações demonstradas em alvos de regressão.
Veja [docs/robomimic_integration.md](docs/robomimic_integration.md) para o formato esperado e configuração.

---

## Métricas Robóticas

Além de acurácia e F1-score, o projeto registra ou disponibiliza utilitários para:

- Task Success Rate (TSR)
- erro de posicionamento em mm
- rounds até convergência
- latência de inferência end-to-end
- acurácia cross-cluster
- acurácia por cluster
- bytes transmitidos por round

As funções reutilizáveis ficam em `utils/robotics_metrics.py`. Na avaliação offline do RoboMimic, TSR mede a proporção de ações cujo erro vetorial fica abaixo do limiar configurado. Na avaliação online, usa o sucesso real reportado pelo ambiente.

As métricas de visão multi-tarefa usadas no artigo, como mIoU, RMSE, mErr, odsF e maxF, ficam em `utils/vision_mtl_metrics.py` para preparar a adaptação futura ao NYUv2/PASCAL.

---

## Estrutura do Projeto

```
FedBone-robotics/
├── config/
│   └── config.py               # Hiperparâmetros e configurações globais
├── data/
│   ├── datasets.py             # Dataset e DataLoader compartilhados
│   └── robomimic_loader.py     # Loader HDF5 RoboMimic
├── models/
│   ├── model.py                # Modelo LSTM baseline
│   └── fedbone_model.py        # Arquitetura FedBone (cliente e servidor)
├── federated/
│   ├── robomimic_baselines.py # FedAvg e Clustered FL para regressão de ações
│   ├── fedbone_fl.py           # Loop de treinamento FedBone
│   └── gp_aggregation.py       # GP Aggregation
├── runner/
│   ├── run_experiments.py      # Executa FedAvg e Clustered FL
│   └── run_fedbone.py          # Executa FedBone
└── utils/
    ├── utils.py                # Métricas e helpers gerais
    ├── robotics_metrics.py     # Métricas específicas para robótica e cloud robotics
    ├── vision_mtl_metrics.py   # Métricas NYUv2/PASCAL usadas no artigo FedBone
    ├── visualization.py        # Plots para FedAvg e Clustered FL
    └── fedbone_visualization.py # Plots para FedBone
```

---

## Instalação

```bash
git clone https://github.com/seu-usuario/FedBone-robotics.git
cd FedBone-robotics
pip install -r requirements.txt
.\scripts\download_robomimic_lowdim.ps1 -Scope expanded-ph
```

---

## Uso

Executar FedAvg e Clustered FL nas mesmas tarefas RoboMimic:
```bash
python runner/run_experiments.py
```

Executar FedBone:
```bash
python runner/run_fedbone.py
```

Coloque datasets `low_dim.hdf5` em `data/robomimic/`. Para aumentar o escopo experimental, use `.\scripts\download_robomimic_lowdim.ps1 -Scope expanded-ph`, que adiciona Lift, Can, Square, Tool Hang e Transport.

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
| `CLIENTS_PER_ROUND` | 10 | Clientes selecionados por round |
| `LOCAL_EPOCHS` | 2 | Épocas locais por round |
| `NUM_CLUSTERS` | 3 | Clusters para Clustered FL |
| `EMBED_DIM` | 64 | Dimensão do embedding no cliente |
| `GENERAL_HIDDEN` | 128 | Hidden size do General Model no servidor |
| `USE_GP_AGGREGATION` | True | Habilitar GP Aggregation no FedBone |
| `RUN_FEDBONE_ABLATIONS` | True | Rodar FedBone sem GP vs com GP no mesmo protocolo |
| `ROBOMIMIC_DATA_DIR` | data/robomimic | Diretório com arquivos HDF5 RoboMimic |

---

## Referências

- Zaland et al., *Federated Learning for Large-Scale Cloud Robotic Manipulation: Opportunities and Challenges*, arXiv:2507.17903, 2025
- Chen et al., *FedBone: Towards Large-Scale Federated Multi-Task Learning*, arXiv:2306.17465, 2023
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS, 2017

---

## Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
