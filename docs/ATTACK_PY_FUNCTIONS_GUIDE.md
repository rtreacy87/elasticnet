# Guide to the Functions in src/attack.py

This guide explains the code in [src/attack.py](../src/attack.py) in a beginner-friendly way.

The goal is simple:
- understand what each function or method does
- learn what input it expects
- learn what output it returns
- be able to run each part in isolation

This file focuses only on [src/attack.py](../src/attack.py).

---

## What lives inside src/attack.py?

The file contains:

1. `AttackConfig` — stores attack settings
2. `AttackResult` — stores attack outputs
3. `ElasticNetAttack` — class that runs the attack
4. `select_correctly_classified_samples()` — helper function for choosing attack inputs

Inside `ElasticNetAttack`, there are these methods:

- `run()`
- `_to_one_hot()`
- `_initialize_binary_search()`
- `_update_best_examples()`
- `_build_result()`

The methods starting with `_` are intended as **internal helper methods**. You can still run them for learning purposes.

---

## Before you begin

Run commands from the [elasticnet](.. ) folder.

Typical setup:

```python
import torch
from src.attack import AttackConfig, AttackResult, ElasticNetAttack, select_correctly_classified_samples
```

If you want to test methods that require a model, create a simple dummy model first:

```python
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return torch.randn(batch_size, 10)

model = DummyModel()
device = torch.device("cpu")
```

This dummy model is useful because it returns fake logits for 10 classes.

---

## How to interpret printed model output

When you print the loaded model, you may see output like this:

```python
MNISTClassifierWithDropout(
    (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (dropout1): Dropout2d(p=0.25, inplace=False)
    (dropout2): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=12544, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```

This is not an error message. It is a readable summary of the model architecture.

Think of it as a blueprint showing the layers the image passes through.

### The overall meaning

This model is a **convolutional neural network (CNN)** built for MNIST digit classification.

It takes in:

- a grayscale image
- with shape `(1, 28, 28)`

and produces:

- 10 output scores
- one score for each digit from `0` to `9`

### How to read each line

#### `MNISTClassifierWithDropout(...)`

This is the class name of the model.

It tells you:

- the model is designed for MNIST
- it includes dropout layers
- it is intended for classification

---

#### `(conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))`

This is the **first convolution layer**.

How to interpret it:

- `Conv2d`
    - this layer scans across the image using small filters
    - it is good at detecting local patterns like edges or strokes
- `1`
    - this is `in_channels`, which means how many channels each input image has
    - for MNIST, each image is grayscale, so every pixel has one intensity value
    - that gives an image shape of `(1, 28, 28)` where:
        - `1` = channels
        - `28` = height
        - `28` = width
    - another way to think about it: the model receives one "stack" (or plane) of numbers per image

Common alternatives to `1`:

- `3` channels
    - standard RGB color images
    - examples: CIFAR-10, ImageNet photos
    - shape example: `(3, 32, 32)` or `(3, 224, 224)`
- `4` channels
    - RGBA images (RGB + alpha/transparency)
    - less common for training unless transparency is important
- `N > 3` channels
    - multi-spectral or scientific imagery
    - medical imaging slices, satellite bands, sensor stacks

Key rule:

- the first number in `Conv2d(in_channels, out_channels, ...)` must match the number of channels in your input tensors
- if your data has 3 channels, `conv1` must start with `Conv2d(3, ...)`, not `Conv2d(1, ...)`

Beginner clarification:

- It is reasonable to think of each channel as a separate input variable (or feature plane).
- But the value range is not always `0` to `255` **inside the model**.

More precise explanation:

1. Raw image files are often stored as integers in `0..255`.
2. During preprocessing, data is usually converted to floating point and often normalized.
3. In this project, the attack config uses `clip_min=0.0` and `clip_max=1.0`, which indicates inputs are expected in the range `0..1` during model/attack processing.

So for MNIST here:

- one channel (grayscale)
- pixel values are typically floats in `0..1` when fed to the model

For RGB images:

- three channels (`R`, `G`, `B`)
- each channel starts from raw `0..255` in many file formats
- after preprocessing, each channel is commonly scaled to `0..1` (or standardized to other ranges)

Quick mental model:

- channel count = "how many value maps per image"
- value range = "how values are encoded after preprocessing"

Both matter, and they are different concepts.
- `32`
    - the layer creates 32 feature maps
    - you can think of these as 32 different pattern detectors

What is a feature map (simple definition):

- A feature map is a 2D grid of numbers produced by one convolution filter.
- High values in that grid mean "this pattern was found strongly here."
- Low values mean "this pattern was weak or not present here."

Concrete example:

- Suppose one learned filter becomes an "edge detector" for diagonal strokes.
- The filter itself is `3 × 3` (because `kernel_size=(3, 3)`).
- But the **output map is not 3 × 3**. The filter slides across the whole image and writes one response at each location.
- After applying that filter to a digit image, it outputs a map (for example, shape `28 × 28`).
- In locations where the digit has a strong diagonal line, values may be larger.
- In background areas, values are usually smaller.

Why this can be confusing:

- `kernel_size` tells you the size of the moving window (`3 × 3`).
- A feature map tells you the collected responses from that window at many positions.

So:

- kernel = local patch size
- feature map = full grid of responses after scanning

In this specific layer (`stride=1`, `padding=1`):

- input is `28 × 28`
- a `3 × 3` kernel scans every location
- padding keeps border sizes stable
- output stays `28 × 28` per filter

Quick formula (applies separately to height and width):

$$
    ext{out} = \left\lfloor\frac{\text{in} + 2p - k}{s}\right\rfloor + 1
$$

Where each symbol means:

- $\text{in}$ = input size along one axis (height or width)
- $p$ = padding size on one side
- $k$ = kernel size along that axis
- $s$ = stride (how far the kernel moves each step)
- $\text{out}$ = output size along that axis

For this layer in your model (`28 × 28`, `k=3`, `p=1`, `s=1`):

$$
    ext{out} = \left\lfloor\frac{28 + 2(1) - 3}{1}\right\rfloor + 1
= \left\lfloor 27 \right\rfloor + 1
= 28
$$

So each filter produces a `28 × 28` map, and with 32 filters you get `32 × 28 × 28`.

Tiny toy map example (not real MNIST values, just to build intuition):

```text
[[0.01, 0.05, 0.80, 0.92],
 [0.00, 0.03, 0.75, 0.88],
 [0.00, 0.01, 0.10, 0.20],
 [0.00, 0.00, 0.02, 0.04]]
```

Applying the formula to the toy size (`4 × 4`):

If we treat `4 × 4` as an **input** and use `k=3`, `s=1`:

1. With padding $p=1$ (same-style):

$$
	ext{out} = \left\lfloor\frac{4 + 2(1) - 3}{1}\right\rfloor + 1
= \left\lfloor 3 \right\rfloor + 1
= 4
$$

Output would be `4 × 4`.

2. With padding $p=0$ (valid/no padding):

$$
	ext{out} = \left\lfloor\frac{4 + 0 - 3}{1}\right\rfloor + 1
= \left\lfloor 1 \right\rfloor + 1
= 2
$$

Output would be `2 × 2`.

So the toy grid shown here can be interpreted as either:

- an output map of size `4 × 4`, or
- an input that would produce `4 × 4` output when `p=1`, and `2 × 2` output when `p=0`.

How to read that toy map:

- top-right area has high responses (`0.80`, `0.92`, `0.88`), so the filter found its pattern there
- bottom-left area is near zero, so the pattern is mostly absent there

In this layer, there are **32 filters**, so you get **32 such maps** per image.
Each map usually focuses on a different type of pattern (edges, curves, stroke fragments, etc.).

Shape intuition:

- if input image shape is `(1, 28, 28)` and `conv1` has 32 filters with same-size padding,
- output shape is typically `(32, 28, 28)`
- that means 32 feature maps, each with height 28 and width 28
- `kernel_size=(3, 3)`
    - each filter looks at a $3 \times 3$ patch at a time
- `stride=(1, 1)`
    - the filter moves 1 pixel at a time
- `padding=(1, 1)`
    - padding helps preserve spatial size after convolution

Beginner interpretation:

This layer looks at the raw handwritten digit and starts extracting simple visual features.

---

#### `(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))`

This is the **second convolution layer**.

How to interpret it:

- input is now `32` channels
    - that means it receives the 32 feature maps from `conv1`
- output is `64` channels
    - now the model learns 64 more advanced feature maps

Beginner interpretation:

The first convolution layer finds simple patterns.
The second layer combines those into richer patterns, like digit shapes or stroke combinations.

---

#### `(dropout1): Dropout2d(p=0.25, inplace=False)`

This is the first dropout layer.

How to interpret it:

- `Dropout2d`
    - randomly turns off some feature maps during training
- `p=0.25`
    - about 25% of features are dropped during training
- `inplace=False`
    - PyTorch creates a new output tensor instead of modifying the old one directly

Beginner interpretation:

This helps prevent the model from memorizing training examples too closely.
It encourages the model to learn more general patterns.

---

#### `(dropout2): Dropout(p=0.5, inplace=False)`

This is another dropout layer, usually used later in the network.

How to interpret it:

- `Dropout`
    - randomly removes some neuron activations during training
- `p=0.5`
    - about 50% are dropped during training

Beginner interpretation:

This is a stronger regularization step than `dropout1`.
It helps reduce overfitting in the fully connected part of the network.

---

#### `(fc1): Linear(in_features=12544, out_features=128, bias=True)`

This is the first fully connected layer.

How to interpret it:

- `Linear`
    - every input value can connect to every output value
- `in_features=12544`
    - the convolution output has been flattened into a long vector of length 12544
- `out_features=128`
    - the model compresses that information into 128 learned features
- `bias=True`
    - this layer also learns a bias term, which is standard

Beginner interpretation:

After learning spatial features with convolution layers, the model now summarizes them into a smaller decision-oriented representation.

---

#### `(fc2): Linear(in_features=128, out_features=10, bias=True)`

This is the final output layer.

How to interpret it:

- input size is `128`
- output size is `10`
    - one output for each digit class: `0` through `9`

Important detail:

These 10 outputs are usually **logits**, not probabilities.
That means they are raw scores.

To get the predicted class, the code usually does something like:

```python
predictions = outputs.argmax(dim=1)
```

Beginner interpretation:

This is the model's final vote across the 10 digit classes.
The largest score becomes the prediction.

---

### How the full model works step by step

A simple mental model is:

1. `conv1`
     - detect simple visual patterns
2. `conv2`
     - build more complex features from earlier patterns
3. `dropout1`
     - reduce overfitting during training
4. flatten data into a long vector
5. `fc1`
     - combine extracted features into a compact representation
6. `dropout2`
     - further reduce overfitting
7. `fc2`
     - produce 10 class scores

### What a beginner should pay attention to

When reading printed model output, focus on these questions:

1. **What kind of layer is this?**
     - convolution, dropout, or linear?
2. **What size data goes in?**
     - channels or feature count
3. **What size data comes out?**
     - more channels, fewer features, or class scores?
4. **What role does the layer play?**
     - feature extraction, regularization, or classification?

### Quick interpretation summary

- `conv1`
    - early feature extraction from grayscale images
- `conv2`
    - deeper feature extraction
- `dropout1`
    - regularization in convolution features
- `dropout2`
    - regularization in dense features
- `fc1`
    - converts many learned features into a compact hidden representation
- `fc2`
    - maps hidden representation to 10 class scores

### Why this matters for the attack section

The attack in [src/attack.py](../src/attack.py) does not change the model structure.
Instead, it changes the **input image** so that this model produces a different final prediction.

So when you see this architecture printed, you should think:

- this is the system being attacked
- the attack will try to manipulate the input
- the goal is to make the final output layer choose the wrong digit

---

## 1. `AttackConfig`

### What it is

`AttackConfig` is a dataclass that stores the hyperparameters for the attack.

Think of it as a configuration box for the attack.

### Default fields

- `beta`
- `confidence`
- `learning_rate`
- `max_iterations`
- `binary_search_steps`
- `initial_const`
- `clip_min`
- `clip_max`

### Why it exists

Instead of scattering hard-coded numbers across the file, all attack settings live in one place.

This makes the code:
- easier to read
- easier to tune
- easier to debug

### Run it in isolation

```python
from src.attack import AttackConfig

config = AttackConfig()
print(config)
```

### Expected result

You should see something similar to:

```python
AttackConfig(beta=0.01, confidence=0.0, learning_rate=0.01, max_iterations=1000, binary_search_steps=5, initial_const=0.001, clip_min=0.0, clip_max=1.0)
```

### Try custom values

```python
config = AttackConfig(
    beta=0.05,
    learning_rate=0.005,
    max_iterations=500,
)
print(config)
```

### What to learn from this

This object does not perform any attack by itself. It only stores settings.

---

## 2. `AttackResult`

### What it is

`AttackResult` is a dataclass that stores the final outputs of the attack.

Think of it as a results container.

### Stored values

- original images
- adversarial images
- true labels
- adversarial predictions
- success mask
- success rate
- L1 distance
- L2 distance
- L∞ distance
- elastic distance

### Why it exists

It groups the attack outputs into one clean object.

Without this, you would need many separate variables.

### Run it in isolation

```python
import torch
from src.attack import AttackResult

result = AttackResult(
    original_images=torch.zeros(2, 1, 28, 28),
    adversarial_images=torch.ones(2, 1, 28, 28),
    true_labels=torch.tensor([3, 7]),
    adv_predictions=torch.tensor([8, 1]),
    success_mask=torch.tensor([True, True]),
    success_rate=100.0,
    l1_dist=torch.tensor([10.0, 12.0]),
    l2_dist=torch.tensor([2.0, 3.0]),
    linf_dist=torch.tensor([0.2, 0.3]),
    elastic_dist=torch.tensor([2.1, 3.12]),
)

print(result.success_rate)
print(result.true_labels)
```

### What to learn from this

`AttackResult` is just a structured way to hold outputs. It does not do any computation by itself.

---

## 3. `ElasticNetAttack`

### What it is

This is the main attack class.

It is responsible for:
- receiving a model
- receiving attack settings
- running the attack loop
- returning an `AttackResult`

### Constructor: `__init__()`

```python
attacker = ElasticNetAttack(model=model, config=config, device=device)
```

### What it needs

- `model`: the neural network to attack
- `config`: an `AttackConfig` object
- `device`: CPU or GPU

### Run it in isolation

```python
import torch
from src.attack import AttackConfig, ElasticNetAttack

config = AttackConfig()
device = torch.device("cpu")
attacker = ElasticNetAttack(model=model, config=config, device=device)
print(attacker.config)
```

### What to learn from this

The class itself is just a wrapper until you call one of its methods.

---

## 4. `ElasticNetAttack._to_one_hot()`

### What it does

This method converts class labels into one-hot vectors.

Example:
- label `2` becomes `[0, 0, 1, 0, 0, ...]`
- label `7` becomes `[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]`

### Why this matters

Many loss functions work more easily when labels are represented as vectors instead of single integers.

### Run it in isolation

```python
import torch
from src.attack import AttackConfig, ElasticNetAttack

config = AttackConfig()
device = torch.device("cpu")
attacker = ElasticNetAttack(model=model, config=config, device=device)

labels = torch.tensor([1, 3, 5])
one_hot = attacker._to_one_hot(labels, num_classes=10)
print(one_hot)
print(one_hot.shape)
```

### Expected output idea

A tensor with shape `(3, 10)` where each row has exactly one `1`.

### What to learn from this

This is a preprocessing step. It prepares labels for the attack loss calculation.

---

## 5. `ElasticNetAttack._initialize_binary_search()`

### What it does

This method creates the starting values for binary search.

It returns:
- `lower_bound`
- `upper_bound`
- `const`

These are used to tune how strongly the attack tries to fool the model versus keep distortion low.

### Why binary search is used

The attack uses a trade-off constant often called `c`.

- if `c` is too small, the attack may fail
- if `c` is too large, perturbations may become unnecessarily large

Binary search helps find a better value.

### Run it in isolation

```python
config = AttackConfig(initial_const=0.001)
attacker = ElasticNetAttack(model=model, config=config, device=torch.device("cpu"))

lower_bound, upper_bound, const = attacker._initialize_binary_search(batch_size=4)

print(lower_bound)
print(upper_bound)
print(const)
```

### Expected output idea

- `lower_bound`: zeros
- `upper_bound`: very large values (`1e10`)
- `const`: all starting at `initial_const`

### What to learn from this

The attack is set up to search for a good constant independently for each sample in the batch.

---

## 6. `ElasticNetAttack._update_best_examples()`

### What it does

This method compares the current candidate adversarial examples against the best ones found so far.

If a candidate:
- succeeds, and
- has lower L2 distortion

then it replaces the old best example.

### Why it matters

The attack does not just want *any* successful adversarial example.
It wants the **best successful one** according to distortion.

### Run it in isolation

```python
import torch

attacker = ElasticNetAttack(model=model, config=AttackConfig(), device=torch.device("cpu"))

current_best_adv = torch.zeros(2, 1, 2, 2)
current_best_l2 = torch.tensor([10.0, 10.0])

candidate_adv = torch.ones(2, 1, 2, 2)
candidate_l2 = torch.tensor([3.0, 12.0])
success_mask = torch.tensor([True, True])

new_best_adv, new_best_l2 = attacker._update_best_examples(
    current_best_adv,
    current_best_l2,
    candidate_adv,
    candidate_l2,
    success_mask,
)

print(new_best_l2)
print(new_best_adv)
```

### What should happen

- first sample should update because `3.0 < 10.0`
- second sample should not update because `12.0 > 10.0`

### What to learn from this

This method is a selection step. It keeps the strongest low-distortion result found so far.

---

## 7. `ElasticNetAttack._build_result()`

### What it does

This method creates the final `AttackResult` object.

It also:
- gets predictions on adversarial images
- computes success mask
- computes distortion metrics

### Inputs

- `best_adv`
- `original_images`
- `attack_targets`
- `targeted`

### Run it in isolation

This method depends on the model and helper functions from [src/en_func.py](../src/en_func.py), so you need tensors with realistic image shapes.

```python
import torch
from src.attack import AttackConfig, ElasticNetAttack

attacker = ElasticNetAttack(model=model, config=AttackConfig(), device=torch.device("cpu"))

original_images = torch.zeros(2, 1, 28, 28)
best_adv = torch.rand(2, 1, 28, 28)
attack_targets = torch.tensor([1, 2])

result = attacker._build_result(
    best_adv=best_adv,
    original_images=original_images,
    attack_targets=attack_targets,
    targeted=False,
)

print(type(result))
print(result.success_mask)
print(result.l2_dist)
```

### What to learn from this

This method transforms raw tensors into a readable, reusable results object.

---

## 8. `ElasticNetAttack.run()`

### What it does

This is the main public method of the class.

It runs the full attack pipeline:

1. copies original images
2. converts labels to one-hot format
3. initializes binary search
4. runs multiple binary-search rounds
5. runs many FISTA iterations in each round
6. checks attack success
7. stores best adversarial examples
8. returns `AttackResult`

### Why this is the most important method

This is the method you will use most often in real code.

### Minimum example to run it in isolation

This method requires:
- a model
- image tensors
- labels
- helper functions from [src/en_func.py](../src/en_func.py)

Example:

```python
import torch
from src.attack import AttackConfig, ElasticNetAttack

class SmallDummyModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return torch.randn(batch_size, 10, requires_grad=True)

model = SmallDummyModel()
config = AttackConfig(max_iterations=2, binary_search_steps=1)
attacker = ElasticNetAttack(model=model, config=config, device=torch.device("cpu"))

attack_data = torch.rand(2, 1, 28, 28)
attack_targets = torch.tensor([1, 4])

result = attacker.run(attack_data, attack_targets, targeted=False)
print(result.success_rate)
print(result.adv_predictions)
```

### Important note

A dummy model produces random behavior, so results will not be meaningful. But this is still useful to understand the flow.

### Better real example

To run it in a realistic way:

```python
from pathlib import Path
import torch
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.utils import load_model
from src.attack import AttackConfig, ElasticNetAttack, select_correctly_classified_samples

device = torch.device("cpu")
train_loader, test_loader = get_mnist_loaders(batch_size=128)

model = MNISTClassifierWithDropout(num_classes=10).to(device)
model = load_model(model, Path("output/mnist_target.pth"), device)

attack_data, attack_targets = select_correctly_classified_samples(
    model, test_loader, num_samples=2, device=device
)

attacker = ElasticNetAttack(
    model=model,
    config=AttackConfig(max_iterations=10, binary_search_steps=1),
    device=device,
)

result = attacker.run(attack_data, attack_targets)
print(result.success_rate)
```

### What to learn from this

`run()` is the full attack engine. The other methods support it.

---

## 9. `select_correctly_classified_samples()`

### What it does

This standalone function searches through the test loader and returns a batch of samples the model already classifies correctly.

### Why this matters

Attack evaluation is only meaningful if the model was correct before the attack.

### Inputs

- `model`
- `test_loader`
- `num_samples`
- `device`

### Output

Returns:
- `attack_data`
- `attack_targets`

### Run it in isolation

```python
import torch
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.utils import load_model
from src.attack import select_correctly_classified_samples

device = torch.device("cpu")
train_loader, test_loader = get_mnist_loaders(batch_size=128)

model = MNISTClassifierWithDropout(num_classes=10).to(device)
model = load_model(model, "output/mnist_target.pth", device)

attack_data, attack_targets = select_correctly_classified_samples(
    model=model,
    test_loader=test_loader,
    num_samples=5,
    device=device,
)

print(attack_data.shape)
print(attack_targets)
```

### Expected output idea

- `attack_data.shape` should look like `(5, 1, 28, 28)`
- `attack_targets` should contain 5 labels

### What to learn from this

This is a filtering step. It makes sure the attack starts from valid baseline predictions.

---

## How the pieces connect

Here is the typical call order:

1. create `AttackConfig`
2. load model
3. call `select_correctly_classified_samples()`
4. create `ElasticNetAttack`
5. call `run()`
6. get an `AttackResult`

In short:

```python
config = AttackConfig()
attack_data, attack_targets = select_correctly_classified_samples(...)
attacker = ElasticNetAttack(model, config, device)
result = attacker.run(attack_data, attack_targets)
```

---

## Beginner-friendly practice plan

If you want to learn this file gradually, test it in this order:

1. `AttackConfig`
2. `AttackResult`
3. `ElasticNetAttack(...)`
4. `_to_one_hot()`
5. `_initialize_binary_search()`
6. `_update_best_examples()`
7. `select_correctly_classified_samples()`
8. `_build_result()`
9. `run()`

That order goes from easiest to hardest.

---

## Common beginner questions

### Why do some methods start with `_`?
That is a Python convention meaning “internal helper method.”
They are still callable, but they are mainly meant to support `run()`.

### Do I need to run helper methods directly in normal use?
No. In normal usage, you usually only need:
- `AttackConfig`
- `select_correctly_classified_samples()`
- `ElasticNetAttack.run()`

### Why does `run()` need images and labels?
Because the attack is trying to change model behavior on specific inputs.
Without images and labels, there is nothing to attack.

### Why are there tensors everywhere?
PyTorch uses tensors as its main data structure for:
- images
- labels
- model outputs
- gradients
- metrics

---

## Practical mini-lab

If you want one short end-to-end experiment from this file only, try this:

```python
from pathlib import Path
import torch
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.utils import load_model
from src.attack import AttackConfig, ElasticNetAttack, select_correctly_classified_samples

device = torch.device("cpu")
_, test_loader = get_mnist_loaders(batch_size=128)

model = MNISTClassifierWithDropout(num_classes=10).to(device)
model = load_model(model, Path("output/mnist_target.pth"), device)

attack_data, attack_targets = select_correctly_classified_samples(
    model=model,
    test_loader=test_loader,
    num_samples=2,
    device=device,
)

attacker = ElasticNetAttack(
    model=model,
    config=AttackConfig(max_iterations=5, binary_search_steps=1),
    device=device,
)

result = attacker.run(attack_data, attack_targets)

print("Success rate:", result.success_rate)
print("Predictions:", result.adv_predictions)
print("L2 distances:", result.l2_dist)
```

### What the input should be

This mini-lab expects the following inputs:

- `output/mnist_target.pth`
    - A trained model checkpoint already saved to disk
    - If this file does not exist, run `python3 train_model.py` first
- `test_loader`
    - A PyTorch data loader containing MNIST test images and labels
    - In this example it is created with `get_mnist_loaders(batch_size=128)`
- `model`
    - An instance of `MNISTClassifierWithDropout` with the saved weights loaded into it
- `attack_data`
    - A small batch of correctly classified images selected from the test set
    - In this example, `num_samples=2`, so the shape should usually be `(2, 1, 28, 28)`
- `attack_targets`
    - The true labels for those selected images
    - In this example, the shape should usually be `(2,)`
- `AttackConfig(max_iterations=5, binary_search_steps=1)`
    - A reduced attack configuration so the experiment runs quickly
    - This is intentionally smaller than the full attack setup, which makes it easier to test and understand

In plain language, you are giving the mini-lab:

1. a trained digit classifier
2. two MNIST images the classifier gets correct
3. attack settings
4. a request to try turning those clean images into adversarial ones

### What the output should look like

The code prints three values:

```python
Success rate: 0.0
Predictions: tensor([1, 4])
L2 distances: tensor([0., 0.], grad_fn=<SumBackward1>)
```

or something like:

```python
Success rate: 50.0
Predictions: tensor([8, 4])
L2 distances: tensor([1.2374, 0.0000], grad_fn=<SumBackward1>)
```

Your exact numbers will vary, but here is how to read them:

- `Success rate`
    - A percentage showing how many selected images were successfully attacked
    - With `num_samples=2`, common values are `0.0`, `50.0`, or `100.0`
- `Predictions`
    - The model's predicted labels **after** the attack
    - This tensor should have 2 values because you attacked 2 images
- `L2 distances`
    - The squared $L_2$ distortion for each adversarial example
    - This tensor should also have 2 values
    - You may see output like `tensor([0., 0.], grad_fn=<SumBackward1>)`
    - `grad_fn=<SumBackward1>` means this tensor was produced by differentiable operations (here, a sum in the L2 computation), so PyTorch is tracking how gradients should flow backward through it
    - In plain language: this value is still part of the computation graph, which is useful while optimizing adversarial examples
    - If you print `result.l2_dist.detach()` or run the computation in a no-grad context, the `grad_fn` part will disappear
    - Other common `grad_fn` names you might see in similar code include:
        - `<MeanBackward0>` (came from a mean)
        - `<AddBackward0>` (came from an addition)
        - `<SubBackward0>` (came from a subtraction)
        - `<PowBackward0>` (came from exponent/power)
        - `<SqrtBackward0>` (came from square root)
        - `<LinalgVectorNormBackward0>` (came from a norm operation)
    - If a tensor prints without `grad_fn` (or shows `grad_fn=None`), it usually means gradients are not being tracked for that tensor at that point
    - Larger values usually mean the attack changed the image more

### What a successful run usually means

If the mini-lab runs correctly, you should see:

- no import errors
- no missing model-file errors
- progress text from the attack loop
- final printed metrics for success rate, predictions, and distortion

If the attack works on at least one sample, then:

- the success rate will be above `0.0`
- at least one adversarial prediction will differ from the original true label
- at least one $L_2$ distance will likely be greater than zero

If the attack does **not** succeed in this small experiment, that is still okay.
Because the configuration uses only `max_iterations=5` and `binary_search_steps=1`, it is designed for speed and learning, not maximum attack performance.

### How to interpret the printed tensors

When you print `attack_data` and `attack_targets` directly after calling
`select_correctly_classified_samples()`, the output looks like this:

```python
print(attack_data)
print(attack_targets)
```

```
tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]]])

tensor([7, 2])
```

#### Reading `attack_data`

The outermost brackets group the two images together.

Each image is nested inside **three** levels of brackets before you reach the pixel rows:

| Bracket level | What it represents |
|---|---|
| Level 1 (outermost) | The batch — all 2 images together |
| Level 2 | One image |
| Level 3 | The single colour channel (grayscale has 1 channel) |
| Level 4 (innermost rows) | The 28 pixel rows of the image |

So the structure is `[batch][image][channel][row]`, which maps directly to the shape `(2, 1, 28, 28)`.

The `...` that PyTorch prints in the middle of each row is not a sign that values are missing — it just means the row is too wide to display in full (28 values). PyTorch hides the middle values and shows the edges only.

#### Why are there so many zeros?

MNIST images are black-and-white. Most of the pixel canvas is white background, which is represented as `0.0` when the pixel values are normalised to the range `[0, 1]`. The actual digit shape only occupies a small cluster of non-zero pixels near the centre of the image. Because PyTorch truncates the display at both ends of each row, you are usually seeing the background edges, not the digit itself.

To see the non-zero pixel values you can inspect a single row directly:

```python
# print all 28 values from row 14 (middle row) of the first image
print(attack_data[0, 0, 14, :])
```

Or check overall statistics to confirm there is real data inside:

```python
print(attack_data.min(), attack_data.max(), attack_data.mean())
# e.g. tensor(0.) tensor(0.9922) tensor(0.0309)
```

The `max` above `0.0` confirms the image contains non-background pixels even though they are not visible in the default print output.

#### Reading `attack_targets`

```
tensor([7, 2])
```

This is straightforward. There are two values because you asked for `num_samples=2`.

- The first value, `7`, is the true class label for the first image in `attack_data`. The model correctly predicted this image as a **7** before the attack.
- The second value, `2`, is the true class label for the second image. The model correctly predicted this as a **2** before the attack.

These labels are the ground truth. The attack will try to push the model into predicting something *other* than `7` and `2` after the perturbation is applied.

---

### Shape summary for beginners

Here is a quick summary of what you should expect:

- `attack_data.shape` → `(2, 1, 28, 28)`
- `attack_targets.shape` → `(2,)`
- `result.adv_predictions.shape` → `(2,)`
- `result.l2_dist.shape` → `(2,)`

That means the experiment is attacking 2 grayscale images, each of size $28 \times 28$.

This is a good first experiment because it is small and easier to reason about.

---

## Final takeaway

[src/attack.py](../src/attack.py) is the control center for the attack stage.

- `AttackConfig` stores settings
- `AttackResult` stores outputs
- `ElasticNetAttack` runs the logic
- `select_correctly_classified_samples()` prepares safe inputs for attack testing

If you understand those four pieces, you understand the structure of the file.
