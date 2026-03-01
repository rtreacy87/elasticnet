# ElasticNet Attack (EAD)

A PyTorch implementation of the ElasticNet Decision-based Adversarial Attack (EAD), a powerful white-box attack that generates adversarial examples with minimal distortion by combining L1 and L2 distance metrics.

## Overview

The ElasticNet Attack is an optimization-based adversarial attack that crafts imperceptible perturbations to fool deep neural networks. Unlike attacks that focus solely on L2 or L∞ distance metrics, EAD uses an **elastic-net regularization** (combination of L1 and L2 norms) to produce sparse perturbations that are both small in magnitude and concentrated in fewer pixels.

This implementation targets MNIST digit classification but can be adapted to other image classification tasks.

## Attack Methodology

### Core Objective

The attack solves the following optimization problem:

```
minimize: ||δ||₂² + β||δ||₁ + c · L_adv(x + δ, y)
subject to: x + δ ∈ [0,1]
```

Where:
- `δ` is the perturbation added to the original image `x`
- `||δ||₂²` is the squared L2 distance (smoothness)
- `||δ||₁` is the L1 distance (sparsity)
- `β` controls the trade-off between L1 and L2 regularization
- `c` balances adversarial loss vs distortion
- `L_adv` is the adversarial loss encouraging misclassification

### Algorithm Components

#### 1. **FISTA Optimization**
The attack uses the **Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)**, which efficiently handles the non-smooth L1 penalty:

- **Gradient Step**: Minimize the smooth part (adversarial loss + L2 distance)
- **Proximal Operator**: Apply soft-thresholding for L1 sparsity
- **Nesterov Acceleration**: Momentum updates for faster convergence

```python
momentum = k / (k + 3)  # where k is iteration number
y_momentum = adv + momentum * (adv - adv_prev)
```

#### 2. **Shrinkage-Thresholding**
The proximal operator for L1 regularization:

```python
if perturbation > threshold:
    perturbation -= threshold
elif perturbation < -threshold:
    perturbation += threshold
else:
    perturbation = 0  # Sparse solution
```

This encourages most perturbations to be exactly zero, creating sparse adversarial examples.

#### 3. **Binary Search on Trade-off Constant**
The constant `c` balances misclassification vs distortion:

- **Too small**: Attack fails to fool the model
- **Too large**: Unnecessary large distortions
- **Binary search**: Iteratively finds optimal `c` for minimal distortion

```python
if attack succeeds:
    upper_bound = c  # Try smaller c
else:
    lower_bound = c  # Need larger c
c = (lower_bound + upper_bound) / 2
```

#### 4. **Carlini & Wagner (C&W) Loss**
The adversarial loss uses a margin-based formulation:

```python
# For untargeted attacks:
loss = max(0, f(x)_true - max(f(x)_other) + κ)
```

Where:
- `f(x)_true` is the logit for the true class
- `max(f(x)_other)` is the highest logit among other classes  
- `κ` (confidence) ensures the attack is robust

## Implementation Details

### Key Parameters

```python
config = {
    "beta": 0.01,              # L1 weight (higher = sparser perturbations)
    "confidence": 0,            # Margin for misclassification
    "learning_rate": 0.01,      # FISTA step size
    "max_iterations": 1000,     # Iterations per binary search step
    "binary_search_steps": 5,   # Number of binary search rounds
    "initial_const": 0.001,     # Starting trade-off constant
    "clip_min": 0.0,           # Min pixel value
    "clip_max": 1.0,           # Max pixel value
}
```

### Attack Pipeline

1. **Select correctly classified samples** from the test set
2. **Initialize binary search bounds** for trade-off constant `c`
3. **For each binary search step:**
   - Initialize adversarial images from originals
   - Run FISTA optimization for max_iterations:
     - Compute loss at momentum point
     - Take gradient step
     - Apply shrinkage-thresholding
     - Update momentum
   - Check attack success
   - Update best adversarial examples (minimal distortion)
   - Adjust binary search bounds
4. **Evaluate final adversarial examples**

### Distance Metrics

The attack tracks three distance metrics:

- **L1 Distance**: `Σ|x_adv - x_orig|` - sum of absolute differences (sparsity)
- **L2 Distance**: `Σ(x_adv - x_orig)²` - sum of squared differences (smoothness)
- **Elastic Distance**: `L2 + β·L1` - combined metric
- **L∞ Distance**: `max|x_adv - x_orig|` - maximum per-pixel change

## Files Structure

```
elasticnet/
├── main.py              # Main attack script
├── src/
│   └── en_func.py       # Core attack functions
├── output/              # Generated models and visualizations
│   ├── mnist_target.pth
│   ├── ead_attack_process.png
│   ├── ead_distortion_analysis.png
│   └── ead_successful_adversarials.png
└── data/                # MNIST dataset (auto-downloaded)
```

## Usage

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib
```

Ensure the HTB AI Library is available with utilities:
- `htb_ai_library.utils` - reproducibility, model I/O, styling
- `htb_ai_library.data` - MNIST data loaders
- `htb_ai_library.models` - MNIST classifier
- `htb_ai_library.training` - training and evaluation functions

### Running the Attack

```bash
python main.py
```

The script will:
1. Train or load an MNIST classifier
2. Perform the ElasticNet attack on 20 correctly classified samples
3. Generate visualizations showing:
   - Original images, perturbations, and adversarial examples
   - Distortion analysis (L1, L2, L∞, Elastic distances)
   - Successful adversarial examples

### Output

```
ElasticNet Attack Results:
============================================================
Success Rate: 95.00% (19/20)
Average L1 Distortion: 12.3456
Average Squared L2 Distortion: 2.3456
Average L∞ Distortion: 0.1234
Average Elastic Distortion: 2.4691
============================================================
```

## Key Advantages

1. **Sparsity**: L1 regularization creates perturbations concentrated in fewer pixels
2. **Imperceptibility**: Combined L1+L2 keeps perturbations small and smooth
3. **Transferability**: Successful attacks often transfer to other models
4. **Flexibility**: Can be targeted or untargeted
5. **Efficiency**: FISTA with momentum converges faster than standard gradient descent

## Attack Variants

### Untargeted Attack (Default)
Forces misclassification to any wrong class:
```python
loss = max(0, logit_true - max(logit_others) + κ)
```

### Targeted Attack
Forces specific target class prediction:
```python
loss = max(0, max(logit_others) - logit_target + κ)
targeted = True  # in function calls
```

## Defense Considerations

This attack demonstrates vulnerabilities in neural networks. Potential defenses include:

- **Adversarial Training**: Train on adversarially perturbed examples
- **Input Preprocessing**: Denoisers, quantization, JPEG compression
- **Detection**: Statistical tests for adversarial inputs
- **Certified Defenses**: Provable robustness guarantees
- **Ensemble Methods**: Multiple models with different architectures

## References

Based on the paper:
- **"EAD: Elastic-Net Attacks to Deep Neural Networks"** by Pin-Yu Chen, Yash Sharma, Huan Zhang, Jinfeng Yi, and Cho-Jui Hsieh (AAAI 2018)

Related attacks:
- **C&W Attack**: Uses L2 or L∞ distance only
- **FGSM**: Fast gradient sign method (L∞)
- **PGD**: Projected gradient descent
- **DeepFool**: Minimal perturbation to decision boundary

## License

This implementation is for educational and research purposes related to adversarial machine learning and security testing.

## Troubleshooting

### Import Errors
Ensure all dependencies are installed and the HTB AI Library is in your Python path.

### Memory Issues
Reduce batch size (`num_samples`) or number of iterations if running out of memory.

### Slow Execution
- Decrease `max_iterations` (may reduce attack success)
- Reduce `binary_search_steps` 
- Use GPU acceleration if available

### Low Success Rate
- Increase `initial_const` or `confidence`
- Increase `max_iterations` or `binary_search_steps`
- Adjust `learning_rate` (typically 0.001-0.1)

## Contact

Part of the HackTheBox AI Security training materials.
