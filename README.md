
# Heterogeneous Graph Neural Network for Drug Repurposing

This repository implements a Heterogeneous Graph Neural Network (GNN) for drug repurposing by predicting interactions between compounds and proteins. The model leverages graph-based embeddings to identify potential drug-protein interactions, which can be useful in identifying new uses for existing drugs.
## Overview

This project aims to assist in drug repurposing by predicting interactions between drugs (compounds) and proteins. The model represents drugs and proteins as graphs and uses a GNN to learn their interactions.
## Data Preparation

- Compound Embeddings: Generated from SMILES strings using a script (compound.py), which converts each compound into a graph and extracts its embedding.
- Protein Embeddings: Generated from protein sequences using a script (protein.py), which converts each sequence into a graph and extracts its embedding.
## Model

- Encoder: A GNN using GIN layers processes the graph-based embeddings of compounds and proteins.
- Decoder: A MLP edge decoder predicts the likelihood of interactions between drug-protein pairs.
- Loss Function: Binary Cross-Entropy with logits.
## Installation

1- Clone the repository\
2- Install dependencies by running 
``` bash
pip install -r requirements.txt
```
3- Run the Model
``` bash
python Main.py
```
4- If you wan t to change embeddings configs, run
``` bash
python EmbGen_Protein.py <method>
```
(method : deepwalk or node2vec)