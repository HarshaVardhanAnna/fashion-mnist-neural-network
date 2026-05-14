# fashion-mnist-neural-network
A numpy-only feedforward neural network trained on Fashion-MNIST, with 6 optimizers and wandb experiment tracking.

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
wandb login
```

## Usage

### Q1 — Visualise one sample per class
```bash
python q1_visualize.py
```