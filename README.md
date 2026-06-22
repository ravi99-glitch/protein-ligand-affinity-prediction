# Protein–Small Molecule Binding Affinity Prediction (PA2)

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C.svg)
![scikit-learn](https://img.shields.io/badge/scikit-learn-1.5-F7931E.svg)
![RDKit](https://img.shields.io/badge/RDKit-2024.03-green.svg)

Predicting the binding affinity between proteins and small molecules is a critical bottleneck in early-stage computational drug discovery. 

This project investigates whether modern machine learning models truly learn underlying molecular interaction principles or merely memorize their training data. By evaluating sequence-based models and structure-based Graph Neural Networks (GNNs) on strict out-of-distribution (OOD) datasets, this repository provides a rigorous framework for testing model generalization.

## Core Features
* **Advanced Feature Engineering:** Encodes protein sequences and ligand SMILES using classical physicochemical descriptors and state-of-the-art pre-trained LLM embeddings (ESM-2, MolFormer).
* **Sequence-Based Baselines:** Implements robust ensemble models (XGBoost, Histogram-Based Gradient Boosting) to establish high-performing baselines.
* **Structure-Based Learning (GNNs):** Integrates 3D spatial geometry modeling using the GEMS architecture to capture spatial interaction patterns.
* **Rigorous Evaluation:** Utilizes the PDBbind database and CASF benchmarks, implementing a custom "CleanSplit" to systematically prevent data leakage and target overlap.

## Key Findings
* **The Memorization Trap:** Sequence-based models (even those augmented with LLM embeddings) performed strongly on known data but struggled significantly on OOD datasets, acting primarily as efficient interpolators rather than generalizable learners.
* **The Geometric Advantage:** The structure-based GNN (GEMS) showed a complementary advantage, achieving positive predictive performance on several unseen protein families where sequence models failed, demonstrating that geometric deep learning raises the ceiling for generalization.

## Tech Stack
* **Languages & Frameworks:** Python, PyTorch, scikit-learn, XGBoost
* **Bioinformatics & Cheminformatics:** BioPython, RDKit, ESM-2, MolFormer, GEMS
* **Data Processing:** NumPy, Pandas, Matplotlib, Seaborn

## 🛠️ Installation & Setup

Clone the repository and set up the Conda environment:

```bash
git clone [https://github.com/ravi99-glitch/PA2.git](https://github.com/ravi99-glitch/PA2.git)
cd PA2
conda env create -f pa2.yml
conda activate pa2_env
