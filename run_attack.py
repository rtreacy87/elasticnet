"""
Run ElasticNet (EAD) attack on MNIST classifier.

This script performs the actual adversarial attack and saves results.
Requires: output/mnist_target.pth (from train_model.py)
Outputs: output/attack_result.pt (pickled AttackResult)
"""

import warnings
from pathlib import Path

import torch

from src.attack import AttackConfig, ElasticNetAttack, select_correctly_classified_samples
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.utils import load_model, set_reproducibility
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


def load_or_exit(path: Path, description: str) -> Path:
    """Check that required file exists or exit."""
    if not path.exists():
        print(f"ERROR: {description} not found at {path}")
        print("Please run train_model.py first.")
        exit(1)
    return path


def print_attack_config(config: AttackConfig) -> None:
    """Display attack configuration."""
    print("\nAttack Configuration:")
    for field_name, value in config.__dict__.items():
        print(f"  {field_name}: {value}")


def print_attack_results(result) -> None:
    """Display attack results summary."""
    batch_size = result.original_images.size(0)
    success_count = int(result.success_mask.sum().item())

    print("\n" + "=" * 60)
    print("ElasticNet Attack Results:")
    print("=" * 60)
    print(f"Success Rate: {result.success_rate:.2f}% ({success_count}/{batch_size})")
    print(f"Average L1 Distortion: {result.l1_dist.mean().item():.4f}")
    print(f"Average Squared L2 Distortion: {result.l2_dist.mean().item():.4f}")
    print(f"Average L∞ Distortion: {result.linf_dist.mean().item():.4f}")
    print(f"Average Elastic Distortion: {result.elastic_dist.mean().item():.4f}")
    print("=" * 60)


def main() -> None:
    """Main attack workflow."""
    device = configure_environment()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Verify model exists
    model_path = load_or_exit(output_dir / "mnist_target.pth", "Trained model")

    # Load data and model
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    print(f"Test samples: {len(test_loader.dataset)}")

    print("\nLoading trained model...")
    model = MNISTClassifierWithDropout(num_classes=10).to(device)
    model = load_model(model, model_path, device)
    model.eval()

    # Configure attack
    config = AttackConfig()
    print_attack_config(config)

    # Select attack samples
    print("\nSelecting correctly classified samples for attack...")
    num_samples = 20
    attack_data, attack_targets = select_correctly_classified_samples(
        model,
        test_loader,
        num_samples=num_samples,
        device=device,
    )
    print(f"Selected {attack_data.size(0)} samples")

    # Run attack
    print("\nPerforming ElasticNet attack...")
    print("This may take several minutes due to iterative optimization...\n")

    attacker = ElasticNetAttack(model=model, config=config, device=device)
    result = attacker.run(attack_data, attack_targets, targeted=False)

    print_attack_results(result)

    # Save results
    result_path = output_dir / "attack_result.pt"
    print(f"\nSaving attack results to {result_path}")
    torch.save(
        {
            "original_images": result.original_images.cpu(),
            "adversarial_images": result.adversarial_images.cpu(),
            "true_labels": result.true_labels.cpu(),
            "adv_predictions": result.adv_predictions.cpu(),
            "success_mask": result.success_mask.cpu(),
            "success_rate": result.success_rate,
            "l1_dist": result.l1_dist.cpu(),
            "l2_dist": result.l2_dist.cpu(),
            "linf_dist": result.linf_dist.cpu(),
            "elastic_dist": result.elastic_dist.cpu(),
        },
        result_path,
    )
    print(f"Results saved successfully.")


if __name__ == "__main__":
    main()
