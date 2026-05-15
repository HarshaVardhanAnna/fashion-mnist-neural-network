# fashion-mnist-neural-network
A NumPy-only feedforward neural network trained on Fashion-MNIST, implemented completely from scratch without TensorFlow/PyTorch training APIs.

The project includes:
- manual forward propagation
- manual backpropagation
- multiple optimizers
- Bayesian hyperparameter sweeps using Weights & Biases
- confusion matrix visualization
- loss-function comparison
- experiment tracking and analysis

---

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
python train.py --epochs 10 --num_hidden_layers 5 --hidden_size 128 --activation tanh --optimizer nadam --learning_rate 0.0001 --batch_size 32 --weight_decay 0 --weight_init xavier --loss cross_entropy --save_path best_model
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
python q7_confusion_matrix.py --model_path best_model.npz

# Or retrain from scratch
python q7_confusion_matrix.py --epochs 10 --optimizer tanh --activation relu
```

### Q8 — Loss function comparison
```bash
python q8_loss_comparison.py
```

## Generated Outputs

- q1_samples.png
- q6_parallel_coords.png
- q6_breakdowns.png
- q7_confusion_matrix.png
- q8_loss_comparison.png