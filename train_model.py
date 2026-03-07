"""
Train or load MNIST classifier model.

This script handles model initialization, training, and checkpointing.
Saves the trained model to output/mnist_target.pth
"""

import warnings
from pathlib import Path

import torch

from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.training import evaluate_accuracy, train_model
from htb_ai_library.utils import load_model, save_model, set_reproducibility
from htb_ai_library.visualization import use_htb_style

warnings.filterwarnings("ignore")


def configure_environment(seed: int = 1337) -> torch.device:
    """Set up reproducibility and device configuration."""
    use_htb_style()
    set_reproducibility(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def main() -> None:
    """Main training workflow."""
    device = configure_environment()

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "mnist_target.pth"

    # Initialize model
    print("\nInitializing MNIST classifier...")
    model = MNISTClassifierWithDropout(num_classes=10).to(device)

    # Train or load model
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        model = load_model(model, model_path, device)
    else:
        print("Training new model...")
        model = train_model(model, train_loader, test_loader, epochs=5, device=device)
        print(f"Saving trained model to {model_path}")
        save_model(model, model_path)

    # Evaluate model
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"\nTest accuracy: {accuracy:.2f}%")
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
