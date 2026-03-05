Motif-Graphlet Graph Neural Networks
Competency Mapping with Motif-Graphlet Graph Neural Networks: Engagement-Based Educational Assessment for the Generative AI Challenge

License: MIT Python 3.9+ PyTorch

Overview
This repository contains the source code for the MGI-GNN (Motif-Graphlet Integrated Graph Neural Network) framework, which shifts educational assessment focus from submitted artefacts to the structural pattern of student engagement.

Generative AI now answers over 65% of university exam questions correctly and produces submissions that go undetected 94% of the time. This framework addresses that challenge by evaluating how students interact with learning materials (graph topology) rather than what they submit (products).

Architecture
Educational Data Sources
        │
        ▼
┌─────────────────────────────┐
│  Heterogeneous Graph G      │
│  (Student-Course-Concept)   │
└───────────┬─────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌──────────┐ ┌───────────┐
│  Motif   │ │ Graphlet  │
│ Adjacency│ │  Degree   │
│ Matrix M │ │ Vectors   │
│ (13 types│ │ (73-dim)  │
│  Eq. 1)  │ │           │
└────┬─────┘ └─────┬─────┘
     │             │
     ▼             ▼
┌──────────┐ ┌───────────┐
│  Motif   │ │ Graphlet  │
│  GCN     │ │ Feedfwd   │
│ Encoder  │ │ Encoder   │
└────┬─────┘ └─────┬─────┘
     │             │
     └──────┬──────┘
            ▼
   ┌────────────────┐
   │ Gated Attention│
   │    Fusion      │
   └───────┬────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│Competency│ │ Cluster │
│Prediction│ │Detection│
└─────────┘ └─────────┘
Key Results
Model	MOOCCubeX AUC	OULAD AUC	MOOCCubeX NMI	OULAD NMI
MLP	0.724 ± 0.008	0.847 ± 0.009	0.312	0.287
GCN	0.781 ± 0.006	0.891 ± 0.007	0.446	0.423
GKT	0.803 ± 0.005	0.903 ± 0.006	0.491	0.462
GIKT	0.819 ± 0.007	0.912 ± 0.005	0.518	0.489
LightGCN	0.792 ± 0.006	0.898 ± 0.007	0.467	0.441
MGI-GNN	0.864 ± 0.004	0.943 ± 0.003	0.624	0.558
Graph-structural engagement features alone (AUC 0.921) outperformed traditional click-based metrics (AUC 0.876) for pass/fail prediction.

Project Structure
├── configs/
│   └── default.yaml              # Experiment configuration
├── src/
│   ├── data/
│   │   ├── mooccubex.py          # MOOCCubeX dataset loader
│   │   ├── oulad.py              # OULAD dataset loader
│   │   └── graph_builder.py      # Heterogeneous graph construction
│   ├── features/
│   │   ├── motif_extractor.py    # Motif adjacency matrix (13 types)
│   │   └── graphlet_extractor.py # Graphlet degree vectors (73-dim)
│   ├── models/
│   │   ├── mgi_gnn.py            # MGI-GNN architecture (Eq. 1)
│   │   └── baselines.py          # MLP, GCN, GKT, GIKT, LightGCN
│   └── evaluation/
│       └── metrics.py            # AUC, NMI, silhouette, t-tests
├── scripts/
│   ├── train.py                  # Single model training pipeline
│   └── run_experiment.py         # Full comparison experiment
├── requirements.txt
├── LICENSE
└── README.md
Installation
git clone https://github.com/wmurikah/Motif-Graphlet-Graph-Neural-Networks.git
cd Motif-Graphlet-Graph-Neural-Networks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
PyTorch Geometric
Install PyTorch Geometric separately following the official guide:

pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
Datasets
MOOCCubeX
Download from the official repository:

mkdir -p data/mooccubex
# Place entity, relation, and interaction JSON files in data/mooccubex/
OULAD
Download from Open University:

mkdir -p data/oulad
# Place CSV files (studentInfo.csv, studentVle.csv, etc.) in data/oulad/
Usage
Train MGI-GNN
# On MOOCCubeX (default)
python scripts/train.py --config configs/default.yaml

# On OULAD
python scripts/train.py --config configs/default.yaml --dataset oulad

# With specific device
python scripts/train.py --config configs/default.yaml --device cpu
Run ablation study
# Remove motif encoder
python scripts/train.py --config configs/default.yaml --ablation no_motif

# Remove graphlet encoder
python scripts/train.py --config configs/default.yaml --ablation no_graphlet

# Replace gated attention with concatenation
python scripts/train.py --config configs/default.yaml --ablation no_gated_attention

# Standard GCN (no higher-order features)
python scripts/train.py --config configs/default.yaml --ablation no_higher_order
Run full comparison experiment
# Runs all 6 models + 4 ablation variants, produces Tables 3-5
python scripts/run_experiment.py --config configs/default.yaml --dataset mooccubex
python scripts/run_experiment.py --config configs/default.yaml --dataset oulad
Configuration
All hyperparameters are in configs/default.yaml. Key settings:

Parameter	MOOCCubeX	OULAD
Embedding dim	128	64
GNN layers	3	2
Learning rate	0.005	0.01
Epochs	100	100
Early stopping	15	15
OULAD overrides are applied automatically when --dataset oulad is specified.

Citation
If you use this code, please cite:

@article{nyaga2025mgignn,
  title={Competency Mapping with Motif-Graphlet Graph Neural Networks:
         Engagement-Based Educational Assessment for the Generative AI Challenge},
  author={Nyaga, Kelvin Munene and Musyoka, Faith Mueni and Mugo, David Muchangi
          and Murikah, Wilberforce},
  year={2025}
}
License
This project is licensed under the MIT License. See LICENSE for details
