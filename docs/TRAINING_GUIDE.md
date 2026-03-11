# Training Models for ElasticNet Attack - Beginner's Guide

This document explains how the model training system works and how you can train your own image classifier. No deep learning experience required!

## Table of Contents

1. [What is a Neural Network Model?](#what-is-a-neural-network-model)
2. [What is MNIST?](#what-is-mnist)
3. [How the Training Script Works](#how-the-training-script-works)
4. [Understanding the Code](#understanding-the-code)
5. [Running the Training](#running-the-training)
6. [Troubleshooting](#troubleshooting)
7. [Customizing Your Model](#customizing-your-model)

## What is a Neural Network Model?

Think of a neural network like a student learning to recognize objects:

- **Training Phase**: The student studies examples (images + correct answers)
- **Learning**: The student's brain adjusts to recognize patterns
- **Testing Phase**: The student takes a test on new examples they haven't seen
- **Accuracy**: We measure how many test questions they answer correctly

A neural network works similarly:
- **Input**: An image (28×28 pixels for MNIST)
- **Hidden Layers**: Mathematical transformations that detect features
- **Output**: A prediction (which digit 0-9 it thinks the image shows)

## What is MNIST?

MNIST is a famous dataset of **handwritten digits** (0-9). It's like a textbook of examples:

- **60,000 training examples**: Handwritten digits with correct labels
- **10,000 test examples**: New handwritten digits to test on
- **Image size**: 28×28 pixels (very small and simple)
- **Goal**: Train a model to recognize what digit each image shows

The dataset is freely available and widely used for teaching machine learning.

## How the Training Script Works

Here's the high-level flow:

```
┌─────────────────────────────────────────────┐
│ 1. Configure Environment                    │
│    - Set up GPU or CPU                      │
│    - Ensure reproducible results            │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 2. Load MNIST Dataset                       │
│    - Download (if needed)                   │
│    - Organize into training/test batches    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 3. Create or Load Model                     │
│    - If model exists: Load from disk        │
│    - If new: Initialize fresh model         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 4. Train Model (if needed)                  │
│    - Feed training examples                 │
│    - Adjust weights based on errors         │
│    - Repeat for 5 epochs                    │
│    - Evaluate on test data each epoch       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 5. Save Model to Disk                       │
│    - Save weights and structure             │
│    - Can be loaded later                    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 6. Evaluate & Report Accuracy               │
│    - Test on unseen examples                │
│    - Report percentage correct              │
└─────────────────────────────────────────────┘
```

## Understanding the Code

### 1. Imports - Tools We Need

```python
import torch                                    # PyTorch: deep learning library
from pathlib import Path                        # Handle file paths easily
from htb_ai_library.data import get_mnist_loaders      # Load MNIST data
from htb_ai_library.models import MNISTClassifierWithDropout  # Model architecture
```

**What each does:**
- `torch`: The deep learning framework that runs neural networks
- `Path`: Makes it easy to work with file paths across different computers
- `get_mnist_loaders`: Downloads and organizes MNIST data
- `MNISTClassifierWithDropout`: A pre-built neural network designed for MNIST

### 2. Environment Configuration

```python
def configure_environment(seed: int = 1337) -> torch.device:
    """Set up reproducibility and device configuration."""
    use_htb_style()
    set_reproducibility(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
```

**What this does:**
- **`set_reproducibility(1337)`**: Uses a "seed" so results are the same every run
  - Like setting a random number generator to always start at the same point
  - Ensures if you run the code twice, you get identical results
- **`device`**: Checks if GPU (cuda) is available
  - GPU = Fast (use if available)
  - CPU = Slower but works everywhere
  - The function automatically picks the best option

**Why reproducibility matters:**
Imagine a friend wants to verify your results. Without reproducibility, their training might give different numbers. With it, they'll get exactly the same results.

### 3. Loading Data

```python
train_loader, test_loader = get_mnist_loaders(batch_size=128)
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")
```

**What happens:**
- **`get_mnist_loaders(batch_size=128)`**: Downloads MNIST and organizes it
  - `batch_size=128`: Process 128 images at a time
  - Larger batches = faster but uses more memory
  - Smaller batches = slower but uses less memory
  - 128 is a good middle ground
- **`train_loader`**: Iterator over training data (60,000 images)
- **`test_loader`**: Iterator over test data (10,000 images)

Think of it like preparing a classroom:
- Training data = Practice problems students solve to learn
- Test data = Final exam questions to measure learning

### 4. Model Creation or Loading

```python
model = MNISTClassifierWithDropout(num_classes=10).to(device)

if model_path.exists():
    print(f"Loading existing model from {model_path}")
    model = load_model(model, model_path, device)
else:
    print("Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=5, device=device)
    save_model(model, model_path)
```

**Two scenarios:**

**Scenario A: First Time Running**
1. Initialize a fresh neural network
2. Network is untrained (random weights)
3. Run training loop to learn patterns
4. Save the trained model

**Scenario B: Already Trained**
1. Load previously trained model from disk
2. Skip training (saves time!)
3. Use it immediately

**Key parameter: `num_classes=10`**
- MNIST has 10 possible outputs (digits 0-9)
- Model learns to output a probability for each digit

### 5. Training Process

```python
model = train_model(model, train_loader, test_loader, epochs=5, device=device)
```

**What "training" means:**

```
For each epoch (5 times):
    For each batch of 128 images:
        1. Show batch to model
        2. Model makes prediction
        3. Compare to correct answer
        4. Calculate error (loss)
        5. Adjust model weights slightly to reduce error
    Evaluate on test set to check progress
```

**What is an epoch?**
- One complete pass through all training data
- 60,000 images ÷ 128 batch size ≈ 469 batches per epoch
- 5 epochs = 469 × 5 ≈ 2,345 total batch updates

**Why 5 epochs?**
- Too few epochs (1-2): Model hasn't learned enough
- Too many epochs (50+): Takes forever, may overfit
- 5 is a practical balance for MNIST

**Progress visualization:**
During training, you'll see output like:
```
Epoch 1/5 Progress: 100% [==========] Loss: 0.0234
Epoch 2/5 Progress: 100% [==========] Loss: 0.0156
...
```

Lower loss = better learning

## Running the Training

### Quick Start

```bash
cd elasticnet
python3 train_model.py
```

### What Happens

1. **First run** (no model exists):
   ```
   Using device: cuda (or cpu)
   Loading MNIST dataset...
   Training samples: 60000
   Test samples: 10000
   
   Initializing MNIST classifier...
   Training new model...
   [Training progresses with epochs...]
   
   Saving trained model to output/mnist_target.pth
   Test accuracy: 98.50%
   ```

2. **Subsequent runs** (model exists):
   ```
   Using device: cuda (or cpu)
   Loading MNIST dataset...
   Test samples: 10000
   
   Loading existing model from output/mnist_target.pth
   
   Test accuracy: 98.50%
   Model saved to: output/mnist_target.pth
   ```

### Understanding Accuracy

```
Test accuracy: 98.50%
```

This means:
- Out of 10,000 test images
- Model correctly identified 9,850 digits
- Only 150 mistakes
- 98.5% is very good for MNIST!

Expected performance:
- Random guessing: 10% (1 in 10)
- Untrained model: 10-20%
- After training: 97-99%

## Troubleshooting

### Issue: "CUDA out of memory"

**Problem**: GPU memory is full

**Solutions**:
```python
# Option 1: Reduce batch size in train_model.py
train_loader, test_loader = get_mnist_loaders(batch_size=64)  # Was 128

# Option 2: Use CPU instead
device = torch.device("cpu")

# Option 3: Clear GPU cache
torch.cuda.empty_cache()
```

### Issue: Training is very slow

**Problem**: Using CPU instead of GPU

**Check your device output:**
```
Using device: cpu  ← Slow
Using device: cuda ← Fast
```

**Solutions**:
- Use a computer with GPU
- Install GPU support (CUDA/cuDNN)
- Reduce batch size if you have limited memory

### Issue: "FileNotFoundError: Can't find mnist_target.pth"

**Problem**: Model hasn't been trained yet

**Solution**:
```bash
# Run training first
python3 train_model.py

# Then run attack
python3 run_attack.py
```

### Issue: Low accuracy (< 90%)

**Problem**: Model didn't train properly

**Solutions**:
```python
# Delete old model to retrain
rm output/mnist_target.pth

# Retrain with more epochs
# (Edit train_model.py and change epochs parameter)

# Run training again
python3 train_model.py
```

## Customizing Your Model

### Changing Number of Epochs

```python
# In train_model.py, change this line:
model = train_model(model, train_loader, test_loader, epochs=5, device=device)

# To train for 10 epochs:
model = train_model(model, train_loader, test_loader, epochs=10, device=device)
```

**Trade-offs:**
- More epochs = Better accuracy but takes longer
- Fewer epochs = Faster but lower accuracy
- Try 5, 10, or 20 depending on your patience

### Changing Batch Size

```python
# In train_model.py, change this line:
train_loader, test_loader = get_mnist_loaders(batch_size=128)

# Smaller batch (slower but less memory):
train_loader, test_loader = get_mnist_loaders(batch_size=32)

# Larger batch (faster but more memory):
train_loader, test_loader = get_mnist_loaders(batch_size=256)
```

### Using a Different Architecture

To use a completely different model, you'd need to:

1. Create a new model class:
```python
class MyCustomModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Define your network layers here
        
    def forward(self, x):
        # Define how data flows through network
        return output
```

2. Update train_model.py:
```python
from my_models import MyCustomModel  # Import your model

model = MyCustomModel(num_classes=10).to(device)
```

## Training Your Own Dataset

To train on a different dataset (not MNIST), you'd need to:

### Step 1: Create a Data Loader

```python
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)
```

### Step 2: Adjust Model Architecture

The `MNISTClassifierWithDropout` expects 28×28 grayscale images. For other sizes:

```python
# MNIST: 28×28 grayscale (1 channel)
# CIFAR-10: 32×32 color (3 channels)
# Create appropriate model for your data
```

### Step 3: Update Training Script

Replace the MNIST-specific calls with your data.

## Summary

The training script:
1. ✓ Sets up environment (GPU/CPU)
2. ✓ Loads MNIST data
3. ✓ Creates or loads a neural network
4. ✓ Trains the network (if needed)
5. ✓ Saves the trained model
6. ✓ Reports accuracy

**You can use this as a template to train models on other datasets!**

The key insight: **A trained model is just a collection of numbers (weights) that have been adjusted to solve a specific task.** Once trained, these weights can be loaded and reused without retraining.

## Key Takeaways

- **Model**: A function that learns patterns from data
- **Training**: The process of adjusting the model's internal weights
- **Accuracy**: How often the model makes correct predictions
- **Epoch**: One complete pass through all training data
- **Batch**: A small group of examples processed together
- **GPU/CPU**: Hardware choices (GPU is faster)
- **Reproducibility**: Getting the same results every time

Good luck training your own models! 🚀
