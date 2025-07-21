# NASA Beyond the Algorithm Challenge: Flood Mapping with SNN & Attention

This repository provides code and documentation for a neuromorphic approach to satellite flood mapping, developed for the NASA Beyond the Algorithm Challenge. The solution leverages spiking neural networks (SNNs) with attention mechanisms, inspired by recent advances in unconventional and brain-inspired computing for Earth science and disaster resilience.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Training](#training)
  - [Testing and Visualization](#testing-and-visualization)
- [Model Architectures](#model-architectures)
- [Citations](#citations)

## Overview

This project implements:
- A **Spiking U-Net with Spiking Self-Attention** (`model_snn.py`) using PyTorch and SpikingJelly and inspired by Spike2Former.
- A **standard U-Net for satellite flood mapping** as a baseline (`model_ann.py`).
- Scripts for training and evaluating both models on optical and SAR flood data (`train_flood_snn.py`, `train_flood_ann.py`).

Designed for rapid, energy-efficient, and explainable flood segmentation from multi-band satellite imagery, this approach fits NASA’s aim of leveraging innovative computational methods for actionable Earth science.

## Features

- **Spiking Neural Networks (SNNs):** Enables brain-inspired, low-power computation.
- **Spiking Self-Attention:** Enhances context awareness for complex Earth data.
- **Modular PyTorch code:** Easy customization or extension.
- **Baseline ANN model:** For comparison and ablation.
- **Visualization pipeline:** For qualitative evaluation of results.

## Technology Stack

| Component      | Technology                | Purpose                                      |
|----------------|--------------------------|----------------------------------------------|
| Data Handling  | Python, NumPy, scikit-image | Loading, preprocessing, and normalization     |
| Deep Learning  | PyTorch                   | Model definition and training                |
| SNNs           | [SpikingJelly](https://github.com/fuzhenn/spikingjelly) | Spiking neuron layers and simulation         |
| Visualization  | Matplotlib                | Result visualization                         |
| Dataset        | Satellite flood imagery (multi-band, geoTIFF/PNG) | Input for semantic segmentation              |

## Directory Structure

```plain
flood_data/
  └── train/
      ├── images/     # .tif satellite images (e.g., 12 bands)
      └── labels/     # .png segmentation masks (0: background, 1: flood)

snn_attention/
  ├── model_snn.py        # Spiking U-Net with Self-Attention
  ├── train_flood_snn.py  # Training/testing script for SNN
  ├── model_ann.py        # Standard U-Net baseline (ANN)
  ├── train_flood_ann.py  # Training/testing script for ANN
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone 
cd snn_attention
```

### 2. Python Environment

Requires Python 3.8+. We recommend using Anaconda or `venv`.

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # Or with Anaconda: conda activate 
```

### 3. Install Dependencies

```bash
pip install torch torchvision
pip install spikingjelly
pip install matplotlib numpy scikit-image
```

Ensure you have access to a CUDA-capable GPU or Apple M1/M2 GPU for best performance (CPU-only mode is supported but slow for SNNs).

### 4. Data Preparation

- Satellite `.tif` images (with multiple spectral bands) located in `flood_data/train/images/`.
- Corresponding binary mask files (`.png`) in `flood_data/train/labels/`.

## Usage

### Training

For Spiking U-Net:

```bash
python train_flood_snn.py --data-dir ../flood_data --batch-size 4 --epochs 50 --lr 1e-4 --timesteps 4 --save-dir snn_checkpoints --data-seed 0
```

For ANN baseline (U-Net):

```bash
python train_flood_ann.py --data-dir ../flood_data --batch-size 16 --epochs 50 --lr 1e-4 --save-dir ann_checkpoints --data-seed 0
```

**Key Arguments:**
- `--data-dir`: Path to data (default: `../flood_data`)
- `--batch-size`: Batch size
- `--epochs`: Number of epochs
- `--lr`: Learning rate
- `--timesteps`: Number of SNN simulation steps (SNN only)
- `--save-dir`: Checkpoint directory
- `--data-seed`: Data split seed for reproducibility, random split if not set

### Testing and Visualization

After training, to generate segmentation predictions and visualizations:

```bash
python train_flood_snn.py --test-only --model-path snn_checkpoints/snn_unet_best.pth --output-dir snn_predictions --num-visualizations 10
```

Result images (original, ground truth, probability, and binary prediction) are saved to `snn_predictions/`.

## Model Architectures

### SpikingUNet with Self-Attention (SNN)

- **Encoder/decoder:** Modeled after U-Net.
- **SpikingDoubleConv:** Standard convolution + BatchNorm + LIF spiking neuron.
- **SpikingSelfAttention:** Multi-head attention applied at deep layers, with spiking activation. Inspired by Spike2Former.
- **Temporal Simulation:** Each input is presented as a sequence of `T` timesteps.
- **Output:** Probability mask per pixel via sigmoid.

### ANNUNet (Baseline)

- **Standard U-Net encoder-decoder:** Convolutions, ReLU activation.
- **No temporal or spiking dynamics.**
- **Output:** Probability mask per pixel via sigmoid.

## Citations

- [1] "NASA Beyond the Algorithm Challenge", https://www.nasa-beyond-challenge.org
