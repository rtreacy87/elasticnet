# ElasticNet Attack Section Guide (Beginner to Intermediate)

This guide explains how the attack stage works in this project, using plain language and practical steps.

## What this section does

The attack stage takes a **trained classifier** and generates **adversarial examples** (inputs that look normal to humans but cause wrong predictions).

In this project, the attack stage is implemented in:
- [run_attack.py](../run_attack.py)
- [src/attack.py](../src/attack.py)
- [src/en_func.py](../src/en_func.py)

The main output is:
- `output/attack_result.pt`

---

## Quick start

From the [elasticnet](..) folder:

1. Train/load model first:
   - `python3 train_model.py`
2. Run attack:
   - `python3 run_attack.py`

If successful, you will get `output/attack_result.pt`.

---

## High-level attack flow

The attack pipeline does this:

1. **Load trained model** (`output/mnist_target.pth`)
2. **Pick correctly classified test images**
3. **Run ElasticNet optimization** to find perturbations
4. **Measure success and distortion**
5. **Save tensors and metrics** to `output/attack_result.pt`

---

## Key files and responsibilities

### [run_attack.py](../run_attack.py)
Orchestrator script for the attack stage.

- Configures device (CPU/GPU)
- Loads model and MNIST test data
- Creates `AttackConfig`
- Calls `ElasticNetAttack.run()`
- Prints attack summary
- Saves results with `torch.save(...)`

### [src/attack.py](../src/attack.py)
Core attack logic in a clean class structure.

- `AttackConfig`: attack hyperparameters
- `AttackResult`: output data container
- `ElasticNetAttack`: class that runs the full attack loop
- `select_correctly_classified_samples()`: picks stable samples for fair attack evaluation

### [src/en_func.py](../src/en_func.py)
Math/optimization helper functions.

- `compute_adversarial_loss()`
- `compute_total_loss()`
- `fista_step()`
- `compute_distances()`
- `update_binary_search_bounds()`
- `check_attack_success()`

---

## Why only correctly classified samples?

Before attacking, the script selects samples the model already gets right.

Reason: if the model is already wrong on an image, changing it is not a meaningful attack success.

So this line of thinking is used:
- baseline prediction is correct → now adversarially flip it

---

## Attack objective (simple explanation)

The attack balances two goals:

1. **Fool the model**
2. **Keep perturbation small**

It combines:
- $L_2$ distortion (smooth magnitude control)
- $L_1$ distortion (sparsity encouragement)
- adversarial classification loss

That is the "ElasticNet" idea: mix of $L_1$ and $L_2$.

---

## Important concepts used in this implementation

### 1) `AttackConfig`
Default parameters in [src/attack.py](../src/attack.py):

- `beta=0.01`: weight of $L_1$ term
- `confidence=0.0`: margin for adversarial objective
- `learning_rate=0.01`: update step size
- `max_iterations=1000`: optimization steps per binary-search round
- `binary_search_steps=5`: rounds to tune trade-off constant
- `initial_const=0.001`: start value for attack trade-off constant
- `clip_min=0.0`, `clip_max=1.0`: valid image range

### 2) Binary search over attack constant
The attack tunes the constant $c$ that balances "fool model" vs "small perturbation".

- if attack succeeds, try smaller $c$
- if attack fails, increase $c$

This helps find less-distorted adversarial examples.

### 3) FISTA optimization
Each round uses `fista_step()` repeatedly.

FISTA combines:
- gradient-based update
- shrinkage/thresholding for $L_1$
- momentum for faster convergence

---

## What gets saved in `attack_result.pt`

The attack script stores these keys:

- `original_images`
- `adversarial_images`
- `true_labels`
- `adv_predictions`
- `success_mask`
- `success_rate`
- `l1_dist`
- `l2_dist`
- `linf_dist`
- `elastic_dist`

These are used later by [generate_plots.py](../generate_plots.py).

---

## Reading the output metrics

When `run_attack.py` finishes, it prints:

- **Success Rate**: percent of samples successfully attacked
- **Average L1 Distortion**: sparsity-related perturbation size
- **Average Squared L2 Distortion**: overall perturbation energy
- **Average L∞ Distortion**: max single-pixel perturbation
- **Average Elastic Distortion**: combined metric used by attack style

General interpretation:
- higher success is better for attacker
- lower distortion is better stealth

---

## Typical failure cases and fixes

### Error: trained model not found
Cause: model checkpoint missing.

Fix:
1. run `python3 train_model.py`
2. rerun `python3 run_attack.py`

### Attack success is very low
Try:
- increase `max_iterations` (e.g. 1000 → 1500)
- increase `binary_search_steps` (e.g. 5 → 7)
- slightly increase `initial_const`
- tune `learning_rate` (too high can overshoot, too low can stall)

### Attack is too slow
Try:
- use GPU
- reduce `max_iterations`
- reduce number of samples (`num_samples` in [run_attack.py](../run_attack.py))

---

## Safe parameter tuning strategy

Change one parameter at a time, record results, then compare.

Suggested order:
1. `max_iterations`
2. `learning_rate`
3. `initial_const`
4. `beta`
5. `binary_search_steps`

---

## Minimal checklist to run your own attack stage

- [ ] `output/mnist_target.pth` exists
- [ ] run `python3 run_attack.py`
- [ ] confirm `output/attack_result.pt` was created
- [ ] inspect success/distortion summary
- [ ] run plotting stage if needed: `python3 generate_plots.py`

---

## Final takeaway

The attack section is an optimization pipeline that searches for the smallest possible perturbation that still causes misclassification. It is modular by design:

- orchestration in [run_attack.py](../run_attack.py)
- attack engine in [src/attack.py](../src/attack.py)
- optimization math in [src/en_func.py](../src/en_func.py)

This separation makes it easier to learn, tune, and extend.
