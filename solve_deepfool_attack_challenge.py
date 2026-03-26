"""
Solve the DeepFool challenge endpoint using shared MNIST challenge utilities.

Usage:
  python3 solve_deepfool_attack_challenge.py
  python3 solve_deepfool_attack_challenge.py --base-url "http://instance_ip:port"

Environment fallback:
  BASE_URL=http://instance_ip:port python3 solve_deepfool_attack_challenge.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

from solve_attack_challenge import (
    NormalizedModel,
    SimpleClassifier,
    b64_png_from_x01,
    download_weights,
    x01_from_b64_png,
)

DEFAULT_BASE_URL = "http://154.57.164.67:31713"
WEIGHTS_PATH = Path("output/deepfool_weights.pth")


@dataclass(frozen=True)
class DeepFoolChallenge:
    label: int
    target: int
    l2_threshold: float
    image_x01: np.ndarray


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.sum(diff ** 2)))


def quantize_x01(x2d: np.ndarray) -> np.ndarray:
    # Match server PNG round-trip behavior to avoid tiny metric mismatches.
    return np.clip(np.round(x2d * 255.0) / 255.0, 0.0, 1.0).astype(np.float32)


def fetch_challenge(session: requests.Session, base_url: str) -> DeepFoolChallenge:
    resp = session.get(f"{base_url}/challenge", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return DeepFoolChallenge(
        label=int(data["label"]),
        target=int(data["target"]),
        l2_threshold=float(data["l2_threshold"]),
        image_x01=x01_from_b64_png(data["image_b64"]),
    )


def local_predict(model: torch.nn.Module, x01_2d: np.ndarray, device: torch.device) -> Tuple[int, np.ndarray]:
    x = torch.from_numpy(x01_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs


def server_predict(session: requests.Session, base_url: str, x01_2d: np.ndarray) -> Dict[str, object]:
    payload = {"image_b64": b64_png_from_x01(x01_2d)}
    resp = session.post(f"{base_url}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def submit_candidate(session: requests.Session, base_url: str, x01_2d: np.ndarray) -> requests.Response:
    payload = {"image_b64": b64_png_from_x01(x01_2d)}
    return session.post(f"{base_url}/submit", json=payload, timeout=15)


def deepfool_targeted(
    model: torch.nn.Module,
    original_x01: np.ndarray,
    target_class: int,
    device: torch.device,
    max_iter: int,
    overshoot: float,
    step_scale: float,
) -> np.ndarray:
    x0 = torch.from_numpy(original_x01).float().unsqueeze(0).unsqueeze(0).to(device)
    x_adv = x0.clone().detach()

    for _ in range(max_iter):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        pred = int(torch.argmax(logits, dim=1).item())
        if pred == target_class:
            break

        current_class = pred

        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        logits[0, current_class].backward(retain_graph=True)
        grad_current = x_adv.grad.detach().clone()

        model.zero_grad(set_to_none=True)
        x_adv.grad.zero_()
        logits[0, target_class].backward()
        grad_target = x_adv.grad.detach().clone()

        w = grad_target - grad_current
        f = logits[0, target_class] - logits[0, current_class]

        w_norm_sq = torch.sum(w * w) + 1e-12
        # Minimal linearized perturbation to cross toward the target boundary.
        r_i = (torch.abs(f) + 1e-6) / w_norm_sq * w

        x_next = x_adv.detach() + step_scale * (1.0 + overshoot) * r_i
        x_adv = torch.clamp(x_next, 0.0, 1.0)

    adv_np = x_adv.detach().cpu().numpy()[0, 0].astype(np.float32)
    return quantize_x01(adv_np)


def refine_within_l2(
    model: torch.nn.Module,
    original_x01: np.ndarray,
    candidate_x01: np.ndarray,
    target_class: int,
    l2_threshold: float,
    device: torch.device,
) -> Optional[np.ndarray]:
    candidate_x01 = quantize_x01(candidate_x01)
    pred, _ = local_predict(model, candidate_x01, device)
    dist = l2_distance(candidate_x01, original_x01)

    if pred == target_class and dist <= l2_threshold + 1e-8:
        return candidate_x01

    # If target is hit but distance is too high, shrink toward original until valid.
    if pred == target_class and dist > l2_threshold:
        for alpha in np.linspace(1.0, 0.0, 301):
            probe = np.clip(original_x01 + alpha * (candidate_x01 - original_x01), 0.0, 1.0).astype(np.float32)
            probe = quantize_x01(probe)
            probe_pred, _ = local_predict(model, probe, device)
            probe_l2 = l2_distance(probe, original_x01)
            if probe_pred == target_class and probe_l2 <= l2_threshold + 1e-8:
                return probe

    return None


def project_to_l2_ball(
    original_x01: np.ndarray,
    candidate_x01: np.ndarray,
    l2_threshold: float,
) -> np.ndarray:
    delta = candidate_x01 - original_x01
    norm = float(np.sqrt(np.sum(delta ** 2)))
    if norm <= l2_threshold + 1e-12:
        return np.clip(candidate_x01, 0.0, 1.0).astype(np.float32)
    scaled = original_x01 + delta * (l2_threshold / (norm + 1e-12))
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def targeted_l2_fallback(
    model: torch.nn.Module,
    original_x01: np.ndarray,
    target_class: int,
    l2_threshold: float,
    device: torch.device,
    max_iter: int,
    step_size: float,
) -> Optional[np.ndarray]:
    """Targeted iterative update with projection onto the L2 ball around the original image."""
    x0 = torch.from_numpy(original_x01).float().unsqueeze(0).unsqueeze(0).to(device)
    x_adv = x0.clone().detach()
    target = torch.tensor([target_class], device=device)

    for _ in range(max_iter):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        pred = int(torch.argmax(logits, dim=1).item())
        if pred == target_class:
            break

        # Minimize CE to target class (targeted attack objective).
        loss = torch.nn.functional.cross_entropy(logits, target)
        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        grad = x_adv.grad.detach()
        grad_norm = torch.linalg.norm(grad.view(1, -1), ord=2) + 1e-12
        x_next = x_adv.detach() - step_size * grad / grad_norm

        x_next_np = x_next.cpu().numpy()[0, 0].astype(np.float32)
        x_next_np = project_to_l2_ball(original_x01, x_next_np, l2_threshold)
        x_next_np = quantize_x01(x_next_np)

        # Quantization can move points slightly; project once more for safety.
        x_next_np = project_to_l2_ball(original_x01, x_next_np, l2_threshold)
        x_adv = torch.from_numpy(x_next_np).float().unsqueeze(0).unsqueeze(0).to(device)

    candidate = quantize_x01(x_adv.detach().cpu().numpy()[0, 0].astype(np.float32))
    candidate = project_to_l2_ball(original_x01, candidate, l2_threshold)
    pred, _ = local_predict(model, candidate, device)
    if pred != target_class:
        return None
    return candidate


def find_valid_candidate(
    model: torch.nn.Module,
    challenge: DeepFoolChallenge,
    session: requests.Session,
    base_url: str,
    device: torch.device,
) -> Optional[np.ndarray]:
    # Wider schedule improves robustness across different challenge samples.
    schedule = [
        (30, 0.02, 1.0),
        (60, 0.02, 1.0),
        (80, 0.05, 1.0),
        (120, 0.05, 1.25),
        (150, 0.10, 1.25),
    ]

    for max_iter, overshoot, step_scale in schedule:
        print(
            f"Attempt: max_iter={max_iter}, overshoot={overshoot}, step_scale={step_scale}"
        )
        raw_candidate = deepfool_targeted(
            model=model,
            original_x01=challenge.image_x01,
            target_class=challenge.target,
            device=device,
            max_iter=max_iter,
            overshoot=overshoot,
            step_scale=step_scale,
        )

        refined = refine_within_l2(
            model=model,
            original_x01=challenge.image_x01,
            candidate_x01=raw_candidate,
            target_class=challenge.target,
            l2_threshold=challenge.l2_threshold,
            device=device,
        )
        if refined is None:
            pred, probs = local_predict(model, raw_candidate, device)
            dist = l2_distance(raw_candidate, challenge.image_x01)
            print(
                f"  Rejected locally: pred={pred}, target_prob={probs[challenge.target]:.6f}, l2={dist:.6f}"
            )
            continue

        local_pred, local_probs = local_predict(model, refined, device)
        dist = l2_distance(refined, challenge.image_x01)
        print(
            f"  Local candidate: pred={local_pred}, target_prob={local_probs[challenge.target]:.6f}, l2={dist:.6f}"
        )

        server_out = server_predict(session, base_url, refined)
        server_pred = int(server_out.get("pred", -1))
        print(f"  Server prediction: {server_out}")

        if server_pred == challenge.target and dist <= challenge.l2_threshold + 1e-8:
            return refined

    # Fallback: targeted L2-projected iterative optimization if DeepFool-style steps stall.
    fallback_schedule = [
        (120, 0.25),
        (200, 0.20),
        (300, 0.15),
    ]

    print("Falling back to targeted L2-projected optimization...")
    for max_iter, step_size in fallback_schedule:
        print(f"Fallback attempt: max_iter={max_iter}, step_size={step_size}")
        candidate = targeted_l2_fallback(
            model=model,
            original_x01=challenge.image_x01,
            target_class=challenge.target,
            l2_threshold=challenge.l2_threshold,
            device=device,
            max_iter=max_iter,
            step_size=step_size,
        )
        if candidate is None:
            print("  Fallback rejected locally (target not reached).")
            continue

        local_pred, local_probs = local_predict(model, candidate, device)
        dist = l2_distance(candidate, challenge.image_x01)
        print(
            f"  Fallback local candidate: pred={local_pred}, "
            f"target_prob={local_probs[challenge.target]:.6f}, l2={dist:.6f}"
        )

        server_out = server_predict(session, base_url, candidate)
        server_pred = int(server_out.get("pred", -1))
        print(f"  Fallback server prediction: {server_out}")

        if server_pred == challenge.target and dist <= challenge.l2_threshold + 1e-8:
            return candidate

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the DeepFool attack challenge.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL), help="Challenge base URL")
    parser.add_argument("--weights-path", default=str(WEIGHTS_PATH), help="Path to cache downloaded weights")
    parser.add_argument("--force-download-weights", action="store_true", help="Re-download /weights even if cached")
    parser.add_argument("--save-adv", default="output/deepfool_challenge_adv.png", help="Path to save selected adversarial PNG")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    device = torch.device("cpu")
    session = requests.Session()

    health = session.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    print(f"Health: {health.json()}")

    challenge = fetch_challenge(session, base_url)
    print("Challenge constraints:")
    print(f"  label: {challenge.label}")
    print(f"  target: {challenge.target}")
    print(f"  l2_threshold: {challenge.l2_threshold}")

    clean_server = server_predict(session, base_url, challenge.image_x01)
    print(f"Server clean prediction: {clean_server}")

    weights_path = download_weights(
        session=session,
        base_url=base_url,
        output_path=Path(args.weights_path),
        force=args.force_download_weights,
    )
    print(f"Weights ready: {weights_path}")

    backbone = SimpleClassifier().to(device)
    state = torch.load(weights_path, map_location=device)
    backbone.load_state_dict(state)
    backbone.eval()

    model = NormalizedModel(backbone).to(device)
    model.eval()

    clean_pred, clean_probs = local_predict(model, challenge.image_x01, device)
    print(f"Local clean prediction: {clean_pred}")
    print(f"Local clean label probability: {clean_probs[challenge.label]:.6f}")
    print(f"Local target probability: {clean_probs[challenge.target]:.6f}")

    candidate = find_valid_candidate(model, challenge, session, base_url, device)
    if candidate is None:
        raise RuntimeError(
            "Failed to generate a valid DeepFool candidate. "
            "Try extending schedule with larger max_iter/step_scale combinations."
        )

    response = submit_candidate(session, base_url, candidate)
    if response.status_code != 200:
        raise RuntimeError(f"Submit rejected ({response.status_code}): {response.text}")

    data = response.json()
    print("\nSUCCESS")
    print(json.dumps(data, indent=2))

    save_path = Path(args.save_adv)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x255 = np.clip(np.round(candidate * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(x255, mode="L").save(save_path)
    print(f"Saved adversarial image to: {save_path}")


if __name__ == "__main__":
    main()
