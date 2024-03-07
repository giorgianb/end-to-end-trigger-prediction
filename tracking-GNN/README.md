# Tracking Pipeline README

## Overview
This repository contains the tracking pipeline designed for processing and analyzing experimental physics data. The pipeline focuses on training models for track reconstruction using graph neural networks (GNNs) and performing inference to build the tracks dataset.

## Structure
- **./train.py**: The main script for training models.
- **./trainers**: Trainer classes for different GNN architectures.
- **./datasets**: Dataset definitions and loaders for handling physics data.
- **./utils**: Utility functions for data processing, checks, and other common tasks.
- **./build_tracks.py**: Script for building tracks from model inference results.
- **./models**: Neural network models, including various GNN architectures designed for track reconstruction.
- **./configs**: Configuration files specifying model parameters and training settings.
- **./inference-pytorch.py**: Script for running model inference.

## Environment Setup
Ensure Python 3.x and necessary packages (PyTorch, numpy, pandas, PyYAML, tqdm) are installed. Use the following command to install dependencies:
```
pip install torch numpy pandas PyYAML tqdm
```

## Training
To train a model, run:
```
python train.py
```
Training configurations are specified within `train.py` and can be adjusted via `./configs/*.yaml` files.

## Inference
After training, run:
```
python inference-pytorch.py
```
to perform inference. This script uses trained models to predict tracks. Modify the `output_dirs` and `model_results_folder` in `inference-pytorch.py` in order to specify the model with which to perform inference and where to store the inference results.

## Track Building
Use `build_tracks.py` to compile inference results into a comprehensive tracks dataset, ready for training the trigger prediction model. Modify the `input_dir` variable to one of the output directories from `inference-pytorch.py`, and modify the `output_dir` variable to where you want the track dataset saved.

## Customization
Modify configuration files in `./configs/` to tailor the pipeline to specific datasets or experimental setups.
