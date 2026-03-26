"""
Solve Skills Assessment 2 (DeepFool minimal perturbation on CIFAR-10).

Usage:
  python3 solve_first_order2_attack_challenge.py
  python3 solve_first_order2_attack_challenge.py --base-url "http://instance_ip:port"

Environment fallback:
  BASE_URL=http://instance_ip:port python3 solve_first_order2_attack_challenge.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
from PIL import Image

from solve_first_order1_attack_challenge import (
    CIFAR10_CLASSES,
    b64_png_from_x01_rgb,
    download_weights,
    load_model,
    local_predict,
    quantize_x01,
    x01_from_b64_png_rgb,
)

DEFAULT_BASE_URL = "http://154.57.164.70:31812"
WEIGHTS_PATH = Path("output/cifar10_model_best_first_order2.pth")


@dataclass(frozen=True)
class FirstOrder2Challenge:
    original_class: int
    l2_threshold: float
    image_x01: np.ndarray  # shape (1, 3, 32, 32)
    mean: Sequence[float]
    std: Sequence[float]
    num_classes_hint: int
    overshoot_hint: float
    max_iterations_hint: int


def l2_normalized_space(a: np.ndarray, b: np.ndarray, std: Sequence[float]) -> float:
    std_arr = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    diff_norm = (a - b) / std_arr
    return float(np.sqrt(np.sum(diff_norm ** 2)))


def fetch_challenge(session: requests.Session, base_url: str) -> FirstOrder2Challenge:
    resp = session.get(f"{base_url}/challenge", timeout=20)
    resp.raise_for_status()
    data = resp.json()

    normalization = data.get("normalization", {})
    mean = normalization.get("mean", [0.4914, 0.4822, 0.4465])
    std = normalization.get("std", [0.2470, 0.2435, 0.2616])

    return FirstOrder2Challenge(
        original_class=int(data["original_class"]),
        l2_threshold=float(data["l2_threshold"]),
        image_x01=x01_from_b64_png_rgb(data["image"]),
        mean=mean,
        std=std,
        num_classes_hint=int(data.get("num_classes_hint", 10)),
        overshoot_hint=float(data.get("overshoot_hint", 0.02)),
        max_iterations_hint=int(data.get("max_iterations_hint", 50)),
    )


def server_predict(session: requests.Session, base_url: str, x4d: np.ndarray) -> Dict[str, object]:
    payload = {"image": b64_png_from_x01_rgb(x4d)}
    resp = session.post(f"{base_url}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def submit_candidate(session: requests.Session, base_url: str, x4d: np.ndarray) -> requests.Response:
    payload = {"image": b64_png_from_x01_rgb(x4d)}
    return session.post(f"{base_url}/submit", json=payload, timeout=30)


def deepfool_untargeted(
    model: torch.nn.Module,
    original_x01: np.ndarray,
    device: torch.device,
    num_classes: int,
    max_iter: int,
    overshoot: float,
) -> np.ndarray:
    x0 = torch.from_numpy(original_x01).float().to(device)
    x_adv = x0.clone().detach()

    for _ in range(max_iter):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        current_class = int(torch.argmax(logits, dim=1).item())

        if current_class != int(torch.argmax(model(x0), dim=1).item()):
            break

        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        logits[0, current_class].backward(retain_graph=True)
        grad_current = x_adv.grad.detach().clone()

        min_pert = None
        best_w = None

        for k in range(num_classes):
            if k == current_class:
                continue

            model.zero_grad(set_to_none=True)
            x_adv.grad.zero_()
            logits[0, k].backward(retain_graph=True)
            grad_k = x_adv.grad.detach().clone()

            w_k = grad_k - grad_current
            f_k = logits[0, k] - logits[0, current_class]
            w_norm = torch.linalg.norm(w_k.reshape(-1), ord=2) + 1e-12
            pert_k = torch.abs(f_k) / w_norm

            if min_pert is None or pert_k < min_pert:
                min_pert = pert_k
                best_w = w_k

        if min_pert is None or best_w is None:
            break

        r_i = (min_pert + 1e-6) * best_w / (torch.linalg.norm(best_w.reshape(-1), ord=2) + 1e-12)
        x_adv = torch.clamp(x_adv.detach() + (1.0 + overshoot) * r_i, 0.0, 1.0)

    candidate = x_adv.detach().cpu().numpy().astype(np.float32)
    return quantize_x01(candidate)


def refine_to_threshold(
    model: torch.nn.Module,
    original_x01: np.ndarray,
    candidate_x01: np.ndarray,
    original_class: int,
    std: Sequence[float],
    l2_threshold: float,
    device: torch.device,
) -> Optional[np.ndarray]:
    candidate_x01 = quantize_x01(candidate_x01)
    pred, _ = local_predict(model, candidate_x01, device)
    dist = l2_normalized_space(candidate_x01, original_x01, std)

    if pred != original_class and dist <= l2_threshold + 1e-8:
        return candidate_x01

    # If already misclassified but distance is high, shrink toward original.
    if pred != original_class and dist > l2_threshold:
        for alpha in np.linspace(1.0, 0.0, 401):
            probe = np.clip(original_x01 + alpha * (candidate_x01 - original_x01), 0.0, 1.0).astype(np.float32)
            probe = quantize_x01(probe)
            probe_pred, _ = local_predict(model, probe, device)
            probe_dist = l2_normalized_space(probe, original_x01, std)
            if probe_pred != original_class and probe_dist <= l2_threshold + 1e-8:
                return probe

    return None


def find_valid_candidate(
    model: torch.nn.Module,
    challenge: FirstOrder2Challenge,
    session: requests.Session,
    base_url: str,
    device: torch.device,
) -> Optional[np.ndarray]:
    max_iter_schedule = sorted(
        set(
            [
                max(20, challenge.max_iterations_hint // 2),
                challenge.max_iterations_hint,
                challenge.max_iterations_hint + 25,
                challenge.max_iterations_hint + 50,
                150,
            ]
        )
    )
    overshoot_schedule = [challenge.overshoot_hint, 0.01, 0.03, 0.05, 0.1]

    for max_iter in max_iter_schedule:
        for overshoot in overshoot_schedule:
            print(f"Attempt: max_iter={max_iter}, overshoot={overshoot}")
            raw_candidate = deepfool_untargeted(
                model=model,
                original_x01=challenge.image_x01,
                device=device,
                num_classes=challenge.num_classes_hint,
                max_iter=max_iter,
                overshoot=overshoot,
            )

            refined = refine_to_threshold(
                model=model,
                original_x01=challenge.image_x01,
                candidate_x01=raw_candidate,
                original_class=challenge.original_class,
                std=challenge.std,
                l2_threshold=challenge.l2_threshold,
                device=device,
            )

            if refined is None:
                pred, probs = local_predict(model, raw_candidate, device)
                dist = l2_normalized_space(raw_candidate, challenge.image_x01, challenge.std)
                print(
                    f"  Rejected locally: pred={pred}, original_prob={probs[challenge.original_class]:.6f}, "
                    f"l2_norm={dist:.6f}"
                )
                continue

            local_pred, local_probs = local_predict(model, refined, device)
            dist = l2_normalized_space(refined, challenge.image_x01, challenge.std)
            print(
                f"  Local candidate: pred={local_pred}, original_prob={local_probs[challenge.original_class]:.6f}, "
                f"l2_norm={dist:.6f}"
            )

            server_out = server_predict(session, base_url, refined)
            server_pred = int(server_out.get("predicted_class", challenge.original_class))
            print(f"  Server prediction: {server_out}")

            if server_pred != challenge.original_class and dist <= challenge.l2_threshold + 1e-8:
                return refined

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Skills Assessment 2 (DeepFool untargeted on CIFAR-10).")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL), help="Challenge base URL")
    parser.add_argument("--weights-path", default=str(WEIGHTS_PATH), help="Path to cache downloaded weights")
    parser.add_argument("--force-download-weights", action="store_true", help="Re-download /model/weights even if cached")
    parser.add_argument("--save-adv", default="output/first_order2_challenge_adv.png", help="Path to save selected adversarial PNG")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session = requests.Session()

    health = session.get(f"{base_url}/health", timeout=15)
    health.raise_for_status()
    print(f"Health: {health.json()}")

    challenge = fetch_challenge(session, base_url)
    print("Challenge constraints:")
    print(f"  original_class: {challenge.original_class} ({CIFAR10_CLASSES[challenge.original_class]})")
    print(f"  l2_threshold (normalized space): {challenge.l2_threshold}")
    print(f"  max_iterations_hint: {challenge.max_iterations_hint}")
    print(f"  overshoot_hint: {challenge.overshoot_hint}")

    clean_server = server_predict(session, base_url, challenge.image_x01)
    print(f"Server clean prediction: {clean_server}")

    weights_path = download_weights(
        session=session,
        base_url=base_url,
        output_path=Path(args.weights_path),
        force=args.force_download_weights,
    )
    print(f"Weights ready: {weights_path}")

    model = load_model(weights_path, mean=challenge.mean, std=challenge.std, device=device)

    clean_local_pred, clean_local_probs = local_predict(model, challenge.image_x01, device)
    print(f"Local clean prediction: {clean_local_pred}")
    print(f"Local original-class probability: {clean_local_probs[challenge.original_class]:.6f}")

    candidate = find_valid_candidate(model, challenge, session, base_url, device)
    if candidate is None:
        raise RuntimeError(
            "Failed to generate a valid DeepFool candidate. "
            "Try extending max_iter_schedule or overshoot_schedule."
        )

    submit_resp = submit_candidate(session, base_url, candidate)
    if submit_resp.status_code != 200:
        raise RuntimeError(f"Submit rejected ({submit_resp.status_code}): {submit_resp.text}")

    data = submit_resp.json()
    print("\nSUCCESS")
    print(json.dumps(data, indent=2))

    save_path = Path(args.save_adv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.transpose(candidate[0], (1, 2, 0))
    x255 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(x255, mode="RGB").save(save_path)
    print(f"Saved adversarial image to: {save_path}")


if __name__ == "__main__":
    main()
