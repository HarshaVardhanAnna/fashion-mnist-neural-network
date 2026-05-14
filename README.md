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

### Q2/Q3 — Train with a custom configuration (no wandb)
```bash
python train.py --epochs 10 --num_hidden_layers 5 --hidden_size 128 --activation relu --optimizer nadam --learning_rate 0.001 --batch_size 32 --weight_decay 0 --weight_init xavier --loss cross_entropy --save_path best_model
```

### Q4/Q5 — Run Bayesian hyperparameter sweep (50 runs)
```bash
python sweep.py
# Sweep URL will be printed; open it to watch runs in real time
```

### Q6 — Fetch sweep results and generate analysis plots
```bash
python q6_analysis.py fashion-mnist-nn <your_sweep_id>
# sweep_id is in the URL: .../sweeps/<sweep_id>
```

### Q7 — Confusion matrix for the best model
```bash
# Load saved model (fast)
python q7_confusion.py --model_path best_model.npz --activation relu

# Or retrain from scratch
python q7_confusion.py --epochs 10 --optimizer nadam --activation relu
```