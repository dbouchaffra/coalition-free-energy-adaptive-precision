# Coalition-Free-Energy-Adaptive-Precision
Variational framework for multi-agent cooperation using coalition free energy and adaptive precision control. Implements GT-FEP, inverted-U credit assignment, and APC algorithm.
# Coalition Free Energy and Adaptive Precision in Multi-Agent Cooperation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of the paper  
**"Coalition Free Energy and Adaptive Precision in Multi‑Agent Cooperation"**  
(D. Bouchaffra, F. Ykhlef, M. Lebbah, H. Azzag).

We introduce the **Game‑Theoretic Free Energy Principle (GT‑FEP)** – a variational framework that models coalition formation via a Gibbs distribution over interacting agents. The framework yields a falsifiable prediction: an agent’s credit assignment (Shapley value) follows an **inverted‑U** curve with respect to its sensory precision β. We then propose **Adaptive Precision Control (APC)**, an online algorithm that dynamically tunes each agent’s observation noise to stay near the optimal peak. Experiments on real‑world Swiss roundabout trajectories, a multi‑agent control task, and the Vicsek flocking model validate the theory and demonstrate the effectiveness of APC.

## Key Features

- **GT‑FEP implementation** – coalition Gibbs distribution, Shapley value estimation, Harsanyi dividend decomposition.
- **Inverted‑U analysis** – fit quadratic models to credit assignment vs. β.
- **Adaptive Precision Control (APC)** – online β adaptation using local gradient estimates.
- **Synergy‑aware credit assignment** – using Harsanyi dividends for interpretable synergy/redundancy.
- **Multiple environments** – Swiss roundabout prediction, MARL collision avoidance, Vicsek flocking.

## Repository Structure
.
├── src/
│ ├── gt_fep.py # Core GT-FEP classes (coalition Gibbs, Shapley)
│ ├── apc.py # Adaptive Precision Control algorithm
│ ├── harsanyi.py # Harsanyi dividend computation via Möbius inversion
│ ├── models.py # Linear predictor and MARL Q‑learning agents
│ └── utils.py # Data loading, metrics, plotting
├── experiments/
│ ├── run_roundabout.py # Swiss roundabout prediction experiments
│ ├── run_marl.py # Multi‑agent control experiments
│ └── run_vicsek.py # Vicsek flocking experiments
├── data/ # Place your CSV datasets here (see below)
├── results/ # Output logs and figures
├── notebooks/ # Jupyter notebooks for analysis
├── requirements.txt
├── LICENSE
└── README.md

text

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/coalition-free-energy-adaptive-precision.git
   cd coalition-free-energy-adaptive-precision
Create a virtual environment (optional but recommended)

bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
Install dependencies

bash
pip install -r requirements.txt
Datasets
Swiss roundabout trajectories
Download the two CSV files (D1_AM2_F1.csv and D2_PM1_L1.csv) from [Zenodo link – insert DOI here]. Place them in the data/ folder.

Vicsek model – generated on the fly (no external data required).

Usage
Reproduce main figures (inverted‑U, APC learning curves)
bash
# Run inverted‑U sweep on first roundabout dataset
python experiments/run_roundabout.py --dataset D1_AM2_F1 --mode invertedU

# Run APC on second roundabout
python experiments/run_roundabout.py --dataset D2_PM1_L1 --mode apc

# Run multi‑agent control experiment with APC
python experiments/run_marl.py --method apc

# Run Vicsek flocking with APC
python experiments/run_vicsek.py --adaptive True
Compute Harsanyi dividends for synergy analysis
bash
python src/harsanyi.py --data data/D1_AM2_F1.csv --precision 4.13
Train with a fixed precision (for comparison)
bash
python experiments/run_roundabout.py --dataset D1_AM2_F1 --mode fixed --beta 2.0
Requirements
The main dependencies are:

Python >= 3.8

numpy

scipy

pandas

matplotlib

scikit-learn

torch (optional, for MARL extensions)

A full list is provided in requirements.txt.

Citation
If you use this code in your research, please cite our paper:

bibtex
@article{bouchaffra2025coalition,
  title={Coalition Free Energy and Adaptive Precision in Multi-Agent Cooperation},
  author={Bouchaffra, Djamel and Ykhlef, Faycal and Lebbah, Mustapha and Azzag, Hanane},
  journal={arXiv preprint arXiv:2605.26278},
  year={2025}
}
License
This project is licensed under the MIT License – see the LICENSE file for details.

Contact
Djamel Bouchaffra – djamel.bouchaffra@uvsq.fr
