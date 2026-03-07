from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from src.en_func import (
    check_attack_success,
    compute_distances,
    fista_step,
    update_binary_search_bounds,
)


@dataclass(frozen=True)
class AttackConfig:
    beta: float = 0.01
    confidence: float = 0.0
    learning_rate: float = 0.01
    max_iterations: int = 1000
    binary_search_steps: int = 5
    initial_const: float = 0.001
    clip_min: float = 0.0
    clip_max: float = 1.0


@dataclass
class AttackResult:
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


class ElasticNetAttack:
    """Single-responsibility attack runner for ElasticNet (EAD)."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: AttackConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    def run(
        self,
        attack_data: torch.Tensor,
        attack_targets: torch.Tensor,
        targeted: bool = False,
    ) -> AttackResult:
        batch_size = attack_data.size(0)
        original_images = attack_data.clone().detach()

        labels_onehot = self._to_one_hot(attack_targets, num_classes=10)
        lower_bound, upper_bound, const = self._initialize_binary_search(batch_size)

        best_adv = original_images.clone()
        best_l2 = torch.full((batch_size,), 1e10, device=self.device)

        for binary_step in range(self.config.binary_search_steps):
            print(f"Binary search step {binary_step + 1}/{self.config.binary_search_steps}")

            adv_images = original_images.clone().detach()
            y_momentum = adv_images.clone()

            for iteration in range(self.config.max_iterations):
                adv_images, y_momentum, _, _ = fista_step(
                    adv_images,
                    y_momentum,
                    original_images,
                    labels_onehot,
                    const,
                    self.model,
                    self.config.beta,
                    self.config.learning_rate,
                    self.config.confidence,
                    iteration,
                    targeted=targeted,
                    clip_min=self.config.clip_min,
                    clip_max=self.config.clip_max,
                )

            success_mask = check_attack_success(
                adv_images,
                attack_targets,
                self.model,
                targeted=targeted,
            )

            _, l2_dist, _ = compute_distances(
                adv_images,
                original_images,
                self.config.beta,
            )
            best_adv, best_l2 = self._update_best_examples(
                best_adv,
                best_l2,
                adv_images,
                l2_dist,
                success_mask,
            )

            lower_bound, upper_bound, const = update_binary_search_bounds(
                lower_bound,
                upper_bound,
                const,
                success_mask,
            )

            print(
                f"  Successfully generated {success_mask.sum().item()}/{batch_size} adversarial examples\n"
            )

        return self._build_result(best_adv, original_images, attack_targets, targeted)

    def _to_one_hot(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        labels_onehot = torch.zeros(labels.size(0), num_classes, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        return labels_onehot

    def _initialize_binary_search(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)
        const = torch.full((batch_size,), self.config.initial_const, device=self.device)
        return lower_bound, upper_bound, const

    def _update_best_examples(
        self,
        current_best_adv: torch.Tensor,
        current_best_l2: torch.Tensor,
        candidate_adv: torch.Tensor,
        candidate_l2: torch.Tensor,
        success_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        improve_mask = success_mask & (candidate_l2 < current_best_l2)
        current_best_adv[improve_mask] = candidate_adv[improve_mask]
        current_best_l2[improve_mask] = candidate_l2[improve_mask]
        return current_best_adv, current_best_l2

    def _build_result(
        self,
        best_adv: torch.Tensor,
        original_images: torch.Tensor,
        attack_targets: torch.Tensor,
        targeted: bool,
    ) -> AttackResult:
        with torch.no_grad():
            adv_outputs = self.model(best_adv)
            adv_predictions = adv_outputs.argmax(dim=1)

        if targeted:
            final_success = adv_predictions.eq(attack_targets)
        else:
            final_success = adv_predictions.ne(attack_targets)

        l1_dist, l2_dist, elastic_dist = compute_distances(
            best_adv,
            original_images,
            self.config.beta,
        )
        linf_dist = torch.max(
            torch.abs(best_adv - original_images).view(best_adv.size(0), -1),
            dim=1,
        )[0]

        return AttackResult(
            original_images=original_images,
            adversarial_images=best_adv,
            true_labels=attack_targets,
            adv_predictions=adv_predictions,
            success_mask=final_success,
            success_rate=final_success.float().mean().item() * 100,
            l1_dist=l1_dist,
            l2_dist=l2_dist,
            linf_dist=linf_dist,
            elastic_dist=elastic_dist,
        )


def select_correctly_classified_samples(
    model: torch.nn.Module,
    test_loader,
    num_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a batch of correctly classified examples for attack."""
    model.eval()

    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        predictions = outputs.argmax(dim=1)

        correct_mask = predictions.eq(targets)
        attack_data = data[correct_mask][:num_samples]
        attack_targets = targets[correct_mask][:num_samples]

        if attack_data.size(0) >= num_samples:
            return attack_data, attack_targets

    raise RuntimeError(
        "Could not find enough correctly classified samples in the test loader."
    )
