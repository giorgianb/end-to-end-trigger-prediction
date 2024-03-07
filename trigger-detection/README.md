# Explainable End-To-End Pipeline for Trigger Detection

## Overview
This repository hosts the implementation for the explainable end-to-end pipeline for trigger detection in experimental physics, featuring Bipartite Graph Networks with Set Transformers, as detailed in our ICDE 2023 paper. Designed for high-energy physics experiments, our pipeline efficiently processes, classifies trigger events, and enhances data acquisition systems' decision-making.

## Structure
- **Dataloaders**: Modules for data loading, preprocessing, and sensitivity analysis.
- **Models**: Neural networks including Set Transformers, ParticleNet, and various attention mechanisms tailored for trigger detection.
- **Utils**: Utilities supporting loss computation and logging.
- **Configs**: YAML configurations dictating experiment setups and model parameters.
- **Main Scripts**: Entrypoints for executing model training, evaluation, and data processing tasks.

## Quick Start
1. **Environment Setup**: 
```
pip install PyYAML scikit-learn matplotlib icecream numpy pandas torch torch_geometric tqdm wandb
```
2. **Running a Script**: Execute any script from `main_scripts/` with:
```
      python main_scripts/<script_name>.py
```
This automatically loads the associated config from `configs/`. The dataset and model parameters can be modified in the appropiate config file.

## Customization
Modify `configs/` YAML files to specify dataset specifications and model parameters,.

## Data and Models
In this repository, we utilize Set Transformer and GarNet as baseline models for our experiments. These models serve as foundational architectures to benchmark our novel approach against.
