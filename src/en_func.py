import torch

def compute_distances(adv_images, original_images, beta):
    """
    Compute L1, L2, and elastic-net distances.

    Returns all three distance metrics used in optimization
    and decision-making.

    Parameters:
        adv_images (torch.Tensor): Adversarial images (batch_size, C, H, W)
        original_images (torch.Tensor): Original images (batch_size, C, H, W)
        beta (float): Weight for L1 in elastic-net distance

    Returns:
        tuple: (l1_dist, l2_dist, elastic_dist) each shape (batch_size,)
    """
    l1_dist = torch.sum(torch.abs(adv_images - original_images), dim=(1, 2, 3))
    l2_dist = torch.sum((adv_images - original_images) ** 2, dim=(1, 2, 3))
    elastic_dist = l2_dist + beta * l1_dist

    return l1_dist, l2_dist, elastic_dist

def compute_adversarial_loss(logits, labels_onehot, confidence, targeted=False):
    """
    Compute margin-based adversarial loss.

    Uses C&W formulation: encourage misclassification with
    confidence margin. Loss becomes zero once margin achieved.

    Parameters:
        logits (torch.Tensor): Model outputs before softmax (batch_size, num_classes)
        labels_onehot (torch.Tensor): One-hot encoded labels (batch_size, num_classes)
        confidence (float): Confidence margin kappa
        targeted (bool): Whether this is a targeted attack

    Returns:
        torch.Tensor: Adversarial loss per example (batch_size,)
    """
    # Extract scores
    real = torch.sum(labels_onehot * logits, dim=1)
    other = torch.max((1 - labels_onehot) * logits - labels_onehot * 10000, dim=1)[0]

    # Compute margin loss
    if targeted:
        # For targeted attacks: want target to exceed other classes by the margin
        loss = torch.clamp(other - real + confidence, min=0)
    else:
        # For untargeted attacks: want real class to be exceeded
        loss = torch.clamp(real - other + confidence, min=0)

    return loss

def compute_total_loss(
    adv_images,
    original_images,
    labels_onehot,
    const,
    model,
    beta,
    confidence,
    targeted=False,
):
    """
    Combine smooth components: c * adversarial + squared L2 (L1 via proximal operator).

    The constant c balances misclassification vs distortion.
    Beta controls L1 vs L2 trade-off (sparsity vs smoothness).

    Parameters:
        adv_images (torch.Tensor): Current adversarial images
        original_images (torch.Tensor): Original clean images
        labels_onehot (torch.Tensor): One-hot encoded labels
        const (torch.Tensor): Trade-off constants c per example
        model (nn.Module): Target model
        beta (float): L1 weight in elastic-net distance
        confidence (float): Margin for misclassification
        targeted (bool): Whether this is a targeted attack

    Returns:
        tuple: (total_loss, adversarial_loss, distances)
    """
    # Get model predictions
    logits = model(adv_images)

    # Compute adversarial loss
    adversarial_loss = compute_adversarial_loss(
        logits, labels_onehot, confidence, targeted
    )

    # Compute distances
    l1_dist, l2_dist, elastic_dist = compute_distances(
        adv_images, original_images, beta
    )

    # Combine: c * adversarial_loss + L2_distance
    # Note: L1 is handled by FISTA's proximal operator, not in this gradient
    total_loss = const * adversarial_loss + l2_dist

    return total_loss, adversarial_loss, (l1_dist, l2_dist, elastic_dist)

def compute_fista_momentum(iteration):
    """
    Calculate FISTA momentum parameter for iteration k.

    Uses Nesterov acceleration: k/(k+3)
    Early iterations have small momentum, later ones accelerate.

    Parameters:
        iteration (int): Current FISTA iteration number

    Returns:
        float: Momentum coefficient in [0, 1)
    """
    return iteration / (iteration + 3.0)

def fista_step(
    adv_images,
    y_momentum,
    original_images,
    labels_onehot,
    const,
    model,
    beta,
    learning_rate,
    confidence,
    iteration,
    targeted=False,
    clip_min=0.0,
    clip_max=1.0,
):
    """
    Perform one complete FISTA iteration.

    Combines gradient computation, shrinkage-thresholding,
    and momentum update into a single optimization step.

    Parameters:
        adv_images (torch.Tensor): Current adversarial images
        y_momentum (torch.Tensor): Momentum point for gradient evaluation
        original_images (torch.Tensor): Original clean images
        labels_onehot (torch.Tensor): One-hot encoded labels
        const (torch.Tensor): Trade-off constants per example
        model (nn.Module): Target model
        beta (float): L1 weight parameter
        learning_rate (float): FISTA step size
        confidence (float): Margin for misclassification
        iteration (int): Current FISTA iteration number for momentum calculation
        targeted (bool): Whether this is a targeted attack
        clip_min (float): Minimum valid pixel value
        clip_max (float): Maximum valid pixel value

    Returns:
        tuple: (new_adv_images, new_y_momentum, loss_value, distances)
    """
    # Ensure y_momentum requires gradients for backprop
    y_momentum = y_momentum.detach().requires_grad_(True)

    # Compute loss at momentum point
    total_loss, adversarial_loss, distances = compute_total_loss(
        y_momentum,
        original_images,
        labels_onehot,
        const,
        model,
        beta,
        confidence,
        targeted,
    )

    # Compute gradient of total loss w.r.t. momentum point
    total_loss_summed = total_loss.sum()
    total_loss_summed.backward()
    grad = y_momentum.grad

    # Gradient step: move in negative gradient direction
    y_new = y_momentum - learning_rate * grad

    # Apply shrinkage-thresholding (proximal operator for L1)
    adv_new = apply_shrinkage_thresholding(
        y_new, original_images, learning_rate * beta, clip_min, clip_max
    )

    # Compute momentum coefficient
    momentum_coef = compute_fista_momentum(iteration)

    # Update momentum point for next iteration
    y_new_momentum = adv_new + momentum_coef * (adv_new - adv_images)

    return adv_new, y_new_momentum, total_loss_summed.item(), distances

def check_attack_success(adv_images, labels, model, targeted=False):
    """
    Verify whether adversarial examples achieve misclassification.

    Compares model predictions on adversarial images to target labels.

    Parameters:
        adv_images (torch.Tensor): Adversarial images to evaluate
        labels (torch.Tensor): True labels (or target labels if targeted)
        model (nn.Module): Target model
        targeted (bool): Whether this is a targeted attack

    Returns:
        torch.Tensor: Boolean mask indicating successful attacks (batch_size,)
    """
    with torch.no_grad():
        outputs = model(adv_images)
        predictions = outputs.argmax(dim=1)

        if targeted:
            # Success = prediction matches target
            success = predictions.eq(labels)
        else:
            # Success = prediction differs from true label
            success = predictions.ne(labels)

    return success

def update_binary_search_bounds(lower_bound, upper_bound, const, success_mask):
    """
    Update binary search bounds based on attack success.

    Successful attacks lower upper bound (c was sufficient).
    Failed attacks raise lower bound (c was insufficient).

    Parameters:
        lower_bound (torch.Tensor): Lower bounds on c per example
        upper_bound (torch.Tensor): Upper bounds on c per example
        const (torch.Tensor): Current c values per example
        success_mask (torch.Tensor): Boolean mask of successful attacks

    Returns:
        tuple: (new_lower_bound, new_upper_bound, new_const)
    """
    # Process each example individually
    for i in range(len(success_mask)):
        if success_mask[i]:
            # Success: try smaller c
            upper_bound[i] = min(upper_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
        else:
            # Failure: need larger c
            lower_bound[i] = max(lower_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
            else:
                const[i] *= 10  # Exponential increase for persistent failures

    return lower_bound, upper_bound, const

def apply_shrinkage_thresholding(y, original_images, threshold, clip_min=0.0, clip_max=1.0):
    # Compute the difference from original
    diff = y - original_images

    # Prepare shrunk values for positive and negative perturbations
    shrink_positive = torch.clamp(y - threshold, min=clip_min, max=clip_max)
    shrink_negative = torch.clamp(y + threshold, min=clip_min, max=clip_max)

    # Three-way decision masks
    cond_positive = (diff > threshold).float()
    cond_zero = (torch.abs(diff) <= threshold).float()
    cond_negative = (diff < -threshold).float()

    # Combine using conditions
    result = (
        cond_positive * shrink_positive
        + cond_zero * original_images
        + cond_negative * shrink_negative
    )
    return result
