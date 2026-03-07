from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from src.attack import AttackResult


@dataclass(frozen=True)
class PlotColors:
    green: str
    black: str
    grey: str
    azure: str
    yellow: str
    red: str
    purple: str
    aquamarine: str


class ElasticNetPlotter:
    """Responsible only for visualization generation."""

    def __init__(self, output_dir: Path, colors: PlotColors) -> None:
        self.output_dir = output_dir
        self.colors = colors

    def create_all(self, result: AttackResult) -> None:
        self._plot_attack_process(result)
        self._plot_distortion_distributions(result)
        self._plot_success_analysis(result)
        self._plot_sparsity_analysis(result)

    def _plot_attack_process(self, result: AttackResult) -> None:
        print("Creating attack process visualization...")

        batch_size = result.original_images.size(0)
        num_display = min(10, batch_size)

        fig, axes = plt.subplots(3, 10, figsize=(20, 6))

        for i in range(num_display):
            original = result.original_images[i].detach().cpu().squeeze()
            perturbation = (
                result.adversarial_images[i] - result.original_images[i]
            ).detach().cpu().squeeze()
            adversarial = result.adversarial_images[i].detach().cpu().squeeze()

            axes[0, i].imshow(original, cmap="gray", vmin=0, vmax=1)
            axes[0, i].axis("off")
            axes[0, i].set_title(f"True: {result.true_labels[i].item()}", fontsize=10)

            axes[1, i].imshow(perturbation * 10, cmap="seismic", vmin=-1, vmax=1)
            axes[1, i].axis("off")
            axes[1, i].set_title("Perturbation", fontsize=10)

            axes[2, i].imshow(adversarial, cmap="gray", vmin=0, vmax=1)
            axes[2, i].axis("off")
            axes[2, i].set_title(
                f"Pred: {result.adv_predictions[i].item()}",
                fontsize=10,
                color=self.colors.red,
            )

        fig.text(
            0.02,
            0.80,
            "Original",
            rotation=90,
            fontsize=14,
            weight="bold",
            ha="center",
            va="center",
        )
        fig.text(
            0.02,
            0.50,
            "Perturbation\n(10× amplified)",
            rotation=90,
            fontsize=14,
            weight="bold",
            ha="center",
            va="center",
        )
        fig.text(
            0.02,
            0.20,
            "Adversarial",
            rotation=90,
            fontsize=14,
            weight="bold",
            ha="center",
            va="center",
        )

        plt.tight_layout(rect=[0.03, 0, 1, 1])
        file_path = self.output_dir / "ead_attack_process.png"
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved to {file_path}")

    def _plot_distortion_distributions(self, result: AttackResult) -> None:
        print("Creating distortion analysis...")

        l1_values = result.l1_dist.detach().cpu().numpy()
        l2_values = result.l2_dist.detach().cpu().numpy()
        linf_values = result.linf_dist.detach().cpu().numpy()
        elastic_values = result.elastic_dist.detach().cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        self._histogram(
            ax=axes[0, 0],
            values=l1_values,
            title=r"$L_1$ Distortion Distribution",
            xlabel=r"$L_1$ Distance",
            color=self.colors.azure,
        )
        self._histogram(
            ax=axes[0, 1],
            values=l2_values,
            title=r"Squared $L_2$ Distortion Distribution",
            xlabel=r"Squared $L_2$ Distance",
            color=self.colors.purple,
        )
        self._histogram(
            ax=axes[1, 0],
            values=linf_values,
            title=r"$L_\infty$ Distortion Distribution",
            xlabel=r"$L_\infty$ Distance",
            color=self.colors.yellow,
        )
        self._histogram(
            ax=axes[1, 1],
            values=elastic_values,
            title="Elastic-Net Distortion Distribution",
            xlabel="Elastic Distance",
            color=self.colors.aquamarine,
        )

        plt.tight_layout()
        file_path = self.output_dir / "ead_distortion_analysis.png"
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved to {file_path}")

    def _plot_success_analysis(self, result: AttackResult) -> None:
        print("Creating mixed-norm analysis...")

        l1_values = result.l1_dist.detach().cpu().numpy()
        l2_values = result.l2_dist.detach().cpu().numpy()
        sparsity_values = self._compute_sparsity(result).detach().cpu().numpy()

        final_success_np = result.success_mask.detach().cpu().numpy().astype(bool)
        colors = [self.colors.green if success else self.colors.red for success in final_success_np]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].set_title(r"$L_1$ vs Squared $L_2$ Distortion Relationship", fontsize=14)
        axes[0].set_xlabel(r"Squared $L_2$ Distance", fontsize=12)
        axes[0].set_ylabel(r"$L_1$ Distance", fontsize=12)
        axes[0].scatter(
            l2_values,
            l1_values,
            c=colors,
            s=100,
            alpha=0.7,
            edgecolors=self.colors.black,
            linewidth=1.5,
        )
        l2_range = np.linspace(l2_values.min(), l2_values.max(), 100)
        axes[0].plot(
            l2_range,
            np.sqrt(l2_range) * 8,
            color=self.colors.azure,
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label=r"Reference: $L_1 \propto \sqrt{L_2}$",
        )
        axes[0].grid(True, alpha=0.3, color=self.colors.grey)

        total = result.original_images.size(0)
        success_count = int(result.success_mask.sum().item())
        failure_count = total - success_count
        legend_elements = [
            Patch(
                facecolor=self.colors.green,
                edgecolor=self.colors.black,
                label=f"Success ({success_count}/{total})",
            ),
            Patch(
                facecolor=self.colors.red,
                edgecolor=self.colors.black,
                label=f"Failed ({failure_count}/{total})",
            ),
        ]
        axes[0].legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=10)

        axes[1].set_title("Sparsity vs Distortion Tradeoff", fontsize=14)
        axes[1].set_xlabel(r"Squared $L_2$ Distance", fontsize=12)
        axes[1].set_ylabel("Sparsity (%)", fontsize=12)
        axes[1].scatter(
            l2_values,
            sparsity_values,
            c=colors,
            s=100,
            alpha=0.7,
            edgecolors=self.colors.black,
            linewidth=1.5,
        )
        axes[1].axvline(
            l2_values.mean(),
            color=self.colors.purple,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean $L_2$: {l2_values.mean():.2f}",
        )
        axes[1].axhline(
            sparsity_values.mean(),
            color=self.colors.aquamarine,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Mean Sparsity: {sparsity_values.mean():.1f}%",
        )
        axes[1].legend(frameon=False, fontsize=10)
        axes[1].grid(True, alpha=0.3, color=self.colors.grey)

        plt.tight_layout()
        file_path = self.output_dir / "ead_success_analysis.png"
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved to {file_path}")

    def _plot_sparsity_analysis(self, result: AttackResult) -> None:
        print("Creating sparsity analysis...")

        batch_size = result.original_images.size(0)
        num_display_sparse = min(10, batch_size)

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.3)

        for i in range(num_display_sparse):
            row = i // 5
            col = i % 5
            ax = fig.add_subplot(gs[row, col])

            perturbation = (
                result.adversarial_images[i] - result.original_images[i]
            ).detach().cpu().squeeze()
            ax.imshow(torch.abs(perturbation), cmap="hot", vmin=0, vmax=perturbation.abs().max())
            ax.axis("off")
            ax.set_title(f"Example {i + 1}", fontsize=10)

        sparsity_values = self._compute_sparsity(result).detach().cpu().numpy()

        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.set_title("Sparsity Distribution Across Examples", fontsize=14)
        ax_stats.set_xlabel("Example Index")
        ax_stats.set_ylabel("Sparsity (%)")
        ax_stats.bar(
            range(len(sparsity_values)),
            sparsity_values,
            color=self.colors.aquamarine,
            alpha=0.7,
            edgecolor=self.colors.black,
        )
        ax_stats.axhline(
            sparsity_values.mean(),
            color=self.colors.red,
            linestyle="--",
            linewidth=2,
            label=f"Mean: {sparsity_values.mean():.2f}%",
        )
        ax_stats.legend(frameon=False, fontsize=10)
        ax_stats.set_ylim([0, 100])

        plt.tight_layout()
        file_path = self.output_dir / "ead_sparsity_analysis.png"
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved to {file_path}")

    def _histogram(self, ax, values: np.ndarray, title: str, xlabel: str, color: str) -> None:
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.hist(values, bins=15, color=color, alpha=0.7, edgecolor=self.colors.black)
        ax.axvline(
            values.mean(),
            color=self.colors.red,
            linestyle="--",
            linewidth=2,
            label=f"Mean: {values.mean():.2f}",
        )
        ax.legend(frameon=False, fontsize=10)

    def _compute_sparsity(self, result: AttackResult) -> torch.Tensor:
        perturbations = result.adversarial_images - result.original_images
        nonzero_mask = torch.abs(perturbations) > 1e-6
        total_features = perturbations[0].numel()
        return (1 - nonzero_mask.float().sum(dim=(1, 2, 3)) / total_features) * 100
