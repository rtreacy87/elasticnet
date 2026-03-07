"""
Generate visualizations from attack results.

This script creates all plots from previously computed attack results.
Requires: output/attack_result.pt (from run_attack.py)
Outputs: PNG files in output/ directory
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from src.plotting import ElasticNetPlotter, PlotColors
from htb_ai_library.utils import (
    AQUAMARINE,
    AZURE,
    HACKER_GREY,
    HTB_GREEN,
    MALWARE_RED,
    NODE_BLACK,
    NUGGET_YELLOW,
    VIVID_PURPLE,
)
from htb_ai_library.visualization import use_htb_style

warnings.filterwarnings("ignore")


@dataclass
class AttackResult:
    """Reconstructed attack result from saved tensors."""
    original_images: torch.Tensor
    adversarial_images: torch.Tensor
    true_labels: torch.Tensor
    adv_predictions: torch.Tensor
    success_mask: torch.Tensor
    success_rate: float
    l1_dist: torch.Tensor
    l2_dist: torch.Tensor
    linf_dist: torch.Tensor
    elastic_dist: torch.Tensor


def load_or_exit(path: Path, description: str) -> Path:
    """Check that required file exists or exit."""
    if not path.exists():
        print(f"ERROR: {description} not found at {path}")
        print("Please run run_attack.py first.")
        exit(1)
    return path


def main() -> None:
    """Main visualization workflow."""
    use_htb_style()

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Verify attack results exist
    result_path = load_or_exit(
        output_dir / "attack_result.pt",
        "Attack results"
    )

    # Load attack results
    print("Loading attack results...")
    data = torch.load(result_path, weights_only=False)
    result = AttackResult(
        original_images=data["original_images"],
        adversarial_images=data["adversarial_images"],
        true_labels=data["true_labels"],
        adv_predictions=data["adv_predictions"],
        success_mask=data["success_mask"],
        success_rate=data["success_rate"],
        l1_dist=data["l1_dist"],
        l2_dist=data["l2_dist"],
        linf_dist=data["linf_dist"],
        elastic_dist=data["elastic_dist"],
    )

    # Display summary
    batch_size = result.original_images.size(0)
    success_count = int(result.success_mask.sum().item())
    print(f"\nAttack Summary:")
    print(f"  Success Rate: {result.success_rate:.2f}% ({success_count}/{batch_size})")
    print(f"  Average L1 Distortion: {result.l1_dist.mean().item():.4f}")
    print(f"  Average Squared L2 Distortion: {result.l2_dist.mean().item():.4f}")
    print(f"  Average L∞ Distortion: {result.linf_dist.mean().item():.4f}")
    print(f"  Average Elastic Distortion: {result.elastic_dist.mean().item():.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plotter = ElasticNetPlotter(
        output_dir=output_dir,
        colors=PlotColors(
            green=HTB_GREEN,
            black=NODE_BLACK,
            grey=HACKER_GREY,
            azure=AZURE,
            yellow=NUGGET_YELLOW,
            red=MALWARE_RED,
            purple=VIVID_PURPLE,
            aquamarine=AQUAMARINE,
        ),
    )
    plotter.create_all(result)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
