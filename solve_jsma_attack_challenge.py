"""
Solve the JSMA challenge endpoint using shared challenge utilities and a local JSMA implementation.

Usage:
  python3 solve_jsma_attack_challenge.py
  python3 solve_jsma_attack_challenge.py --base-url "http://instance_ip:port"

Environment fallback:
  BASE_URL=http://instance_ip:port python3 solve_jsma_attack_challenge.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from solve_attack_challenge import NormalizedModel, b64_png_from_x01, download_weights, x01_from_b64_png

DEFAULT_BASE_URL = "http://154.57.164.77:31717"
WEIGHTS_PATH = Path("output/jsma_weights.pth")


@dataclass(frozen=True)
class JSMAChallenge:
    original_label: int
    target_class: int
    l0_budget: int
    max_l2: float
    image_x01: np.ndarray


class MNISTClassifier(nn.Module):
    """LeNet-5 style classifier used by the challenge service."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def fetch_challenge(session: requests.Session, base_url: str) -> JSMAChallenge:
    resp = session.get(f"{base_url}/challenge", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return JSMAChallenge(
        original_label=int(data["original_label"]),
        target_class=int(data["target_class"]),
        l0_budget=int(data["l0_budget"]),
        max_l2=float(data["max_l2"]),
        image_x01=x01_from_b64_png(data["image_b64"]),
    )


def quantize_x01(x01_2d: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x01_2d * 255.0) / 255.0, 0.0, 1.0).astype(np.float32)


def count_modified_pixels(a: np.ndarray, b: np.ndarray, threshold: float = 1e-6) -> int:
    return int(np.sum(np.abs(a - b) > threshold))


def compute_l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.sum(diff ** 2)))


def within_constraints(challenge: JSMAChallenge, original: np.ndarray, candidate: np.ndarray) -> bool:
    return (
        count_modified_pixels(original, candidate) <= challenge.l0_budget
        and compute_l2(original, candidate) <= challenge.max_l2 + 1e-8
    )


def local_predict(model: nn.Module, x01_2d: np.ndarray, device: torch.device) -> Tuple[int, np.ndarray]:
    x = torch.from_numpy(x01_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(x)
        probs = torch.exp(log_probs)[0].cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs


def server_predict(session: requests.Session, base_url: str, x01_2d: np.ndarray) -> Dict[str, object]:
    payload = {"image_b64": b64_png_from_x01(x01_2d)}
    resp = session.post(f"{base_url}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def compute_jacobian(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.clone().detach().requires_grad_(True)
    outputs = model(x)
    num_classes = outputs.shape[1]
    jacobian_rows = []

    for class_idx in range(num_classes):
        model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(outputs[0, class_idx], x, retain_graph=class_idx < num_classes - 1)[0]
        jacobian_rows.append(grad.detach().view(-1))

    return torch.stack(jacobian_rows, dim=0), outputs.detach()


def select_salient_features(
    jacobian: torch.Tensor,
    target_class: int,
    search_space: torch.Tensor,
    increase: bool,
    remaining_budget: int,
) -> Optional[List[int]]:
    candidate_indices = torch.nonzero(search_space, as_tuple=False).view(-1)
    if candidate_indices.numel() == 0:
        return None

    target_grad = jacobian[target_class, candidate_indices]
    other_grad = jacobian[:, candidate_indices].sum(dim=0) - target_grad

    if remaining_budget >= 2 and candidate_indices.numel() >= 2:
        alpha = target_grad[:, None] + target_grad[None, :]
        beta = other_grad[:, None] + other_grad[None, :]
        diagonal = torch.eye(candidate_indices.numel(), dtype=torch.bool, device=jacobian.device)
        if increase:
            valid = (alpha > 0) & (beta < 0) & (~diagonal)
        else:
            valid = (alpha < 0) & (beta > 0) & (~diagonal)

        if valid.any():
            saliency = torch.full_like(alpha, float("-inf"))
            saliency[valid] = -alpha[valid] * beta[valid]
            best_pair = torch.argmax(saliency)
            row = int(best_pair // candidate_indices.numel())
            col = int(best_pair % candidate_indices.numel())
            return [int(candidate_indices[row].item()), int(candidate_indices[col].item())]

    if increase:
        valid_single = (target_grad > 0) & (other_grad < 0)
    else:
        valid_single = (target_grad < 0) & (other_grad > 0)

    if not valid_single.any() or remaining_budget <= 0:
        return None

    single_scores = torch.full_like(target_grad, float("-inf"))
    single_scores[valid_single] = -target_grad[valid_single] * other_grad[valid_single]
    best_single = int(candidate_indices[int(torch.argmax(single_scores).item())].item())
    return [best_single]


def run_jsma_attack(
    model: nn.Module,
    challenge: JSMAChallenge,
    device: torch.device,
    theta: float,
    max_iterations: int,
) -> Optional[np.ndarray]:
    original = torch.from_numpy(challenge.image_x01).float().unsqueeze(0).unsqueeze(0).to(device)
    adversarial = original.clone().detach()
    original_flat = original.view(-1)
    adversarial_flat = adversarial.view(-1)
    search_space = torch.ones_like(adversarial_flat, dtype=torch.bool)

    for _ in range(max_iterations):
        current = quantize_x01(adversarial.detach().cpu().numpy()[0, 0])
        current_pred, _ = local_predict(model, current, device)
        if current_pred == challenge.target_class and within_constraints(challenge, challenge.image_x01, current):
            return current

        remaining_budget = challenge.l0_budget - count_modified_pixels(challenge.image_x01, current)
        if remaining_budget <= 0:
            break

        if theta > 0:
            search_space &= adversarial_flat < 1.0 - 1e-6
        else:
            search_space &= adversarial_flat > 1e-6

        jacobian, _ = compute_jacobian(model, adversarial)
        salient = select_salient_features(
            jacobian=jacobian,
            target_class=challenge.target_class,
            search_space=search_space,
            increase=theta > 0,
            remaining_budget=remaining_budget,
        )
        if salient is None:
            break

        for feature_idx in salient[:remaining_budget]:
            adversarial_flat[feature_idx] = torch.clamp(adversarial_flat[feature_idx] + theta, 0.0, 1.0)
            search_space[feature_idx] = False

        candidate = quantize_x01(adversarial.detach().cpu().numpy()[0, 0])
        if compute_l2(challenge.image_x01, candidate) > challenge.max_l2 + 1e-8:
            break

    final_candidate = quantize_x01(adversarial.detach().cpu().numpy()[0, 0])
    final_pred, _ = local_predict(model, final_candidate, device)
    if final_pred == challenge.target_class and within_constraints(challenge, challenge.image_x01, final_candidate):
        return final_candidate
    return None


def submit_candidate(session: requests.Session, base_url: str, x01_2d: np.ndarray) -> requests.Response:
    payload = {"image_b64": b64_png_from_x01(x01_2d)}
    return session.post(f"{base_url}/submit", json=payload, timeout=15)


def print_constraints(challenge: JSMAChallenge) -> None:
    print("Challenge constraints:")
    print(f"  original_label: {challenge.original_label}")
    print(f"  target_class: {challenge.target_class}")
    print(f"  l0_budget: {challenge.l0_budget}")
    print(f"  max_l2: {challenge.max_l2}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the JSMA attack challenge.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL), help="Challenge base URL")
    parser.add_argument("--weights-path", default=str(WEIGHTS_PATH), help="Path to cache downloaded weights")
    parser.add_argument("--force-download-weights", action="store_true", help="Re-download /weights even if cached")
    parser.add_argument("--save-adv", default="output/jsma_challenge_adv.png", help="Path to save selected adversarial PNG")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    device = torch.device("cpu")
    session = requests.Session()

    health = session.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    print(f"Health: {health.json()}")

    challenge = fetch_challenge(session, base_url)
    print_constraints(challenge)

    server_clean = server_predict(session, base_url, challenge.image_x01)
    print(f"Server clean prediction: {server_clean}")

    weights_path = download_weights(
        session=session,
        base_url=base_url,
        output_path=Path(args.weights_path),
        force=args.force_download_weights,
    )
    print(f"Weights ready: {weights_path}")

    backbone = MNISTClassifier().to(device)
    state = torch.load(weights_path, map_location=device)
    backbone.load_state_dict(state)
    backbone.eval()

    model = NormalizedModel(backbone).to(device)
    model.eval()

    local_clean_pred, local_clean_probs = local_predict(model, challenge.image_x01, device)
    print(f"Local clean prediction: {local_clean_pred}")
    print(f"Local clean probabilities: {np.round(local_clean_probs, 4).tolist()}")

    schedules: Sequence[Tuple[float, int]] = [
        (1.0, challenge.l0_budget),
        (0.5, challenge.l0_budget * 2),
        (-1.0, challenge.l0_budget),
        (-0.5, challenge.l0_budget * 2),
    ]

    for attempt_idx, (theta, max_iterations) in enumerate(schedules, start=1):
        print("\n" + "-" * 70)
        print(f"Attempt {attempt_idx}: theta={theta}, max_iterations={max_iterations}")

        candidate = run_jsma_attack(
            model=model,
            challenge=challenge,
            device=device,
            theta=theta,
            max_iterations=max_iterations,
        )
        if candidate is None:
            print("No locally valid candidate found for this schedule.")
            continue

        pixels_modified = count_modified_pixels(challenge.image_x01, candidate)
        l2_value = compute_l2(challenge.image_x01, candidate)
        local_pred, local_probs = local_predict(model, candidate, device)
        print(
            "Local candidate stats: "
            f"pred={local_pred}, pixels_modified={pixels_modified}, l2={l2_value:.6f}, "
            f"target_prob={local_probs[challenge.target_class]:.6f}"
        )

        server_candidate = server_predict(session, base_url, candidate)
        print(f"Server candidate prediction: {server_candidate}")
        if int(server_candidate["predicted_class"]) != challenge.target_class:
            print("Server prediction does not match the target class. Continuing search.")
            continue

        resp = submit_candidate(session, base_url, candidate)
        if resp.status_code == 200:
            data = resp.json()
            print("\nSUCCESS")
            print(json.dumps(data, indent=2))

            save_path = Path(args.save_adv)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            x255 = np.clip(np.round(candidate * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(x255, mode="L").save(save_path)
            print(f"Saved adversarial image to: {save_path}")
            return

        print(f"Submit rejected ({resp.status_code}): {resp.text}")

    raise RuntimeError(
        "Failed to solve the JSMA challenge with the current schedules. "
        "Try adding more theta values or allowing more iterations per schedule."
    )


if __name__ == "__main__":
    main()