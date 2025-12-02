# Protein Small Molecule Binding Affinity Prediction
This project lies at the intersection of computational drug discovery and machine learning. It investigates models that predict the binding affinity between proteins and small molecules (e.g., aspirin binding to cyclooxygenase).
My aim is to determine whether these models truly learn molecular interaction principles or merely memorize their training data.

---
## Features
- Encodes protein sequences and ligand SMILES for ML models
- Implements sequence-based baselines using classical ML and LLM embeddings
- Integrates structure-based learning with graph neural networks (GNNs)
- Evaluates model generalization vs. memorization
- Includes robust train/test splits to detect data leakage

## Goals
While many affinity prediction models exist, few generalize well to new protein–ligand pairs.
This project aims to:
- Quantify how much models learn vs. memorize
- Establish solid baselines using sequence-based approaches
- Evaluate the added value of structure-based neural networks

## Requirements
The project runs in a Conda environment defined in `pa2.yml`

**Main dependencies:**
Python 3.9
NumPy, Pandas, Matplotlib, Seaborn
scikit-learn, XGBoost
BioPython, RDKit

## Project Work

This project was developed as part of an applied research initiative in computational drug discovery.
It explores the intersection of bioinformatics, machine learning, and structural biology to build interpretable, generalizable protein–ligand prediction models.

Author: Ravidu Nakandalage
Institution: Zurich University of Applied Sciences (ZHAW), 2025


