"""
Solve the FGSM challenge endpoint by reusing shared MNIST challenge utilities.

Usage:
  python3 solve_fgsm_attack_challenge.py
  python3 solve_fgsm_attack_challenge.py --base-url "http://instance_ip:port"

Environment fallback:
  BASE_URL=http://instance_ip:port python3 solve_fgsm_attack_challenge.py
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
import torch.nn.functional as F
from PIL import Image

from solve_attack_challenge import (
    DEFAULT_BASE_URL as EAD_DEFAULT_BASE_URL,
    NormalizedModel,
    SimpleClassifier,
    b64_png_from_x01,
    download_weights,
    x01_from_b64_png,
)

DEFAULT_BASE_URL = "http://154.57.164.74:32321"
WEIGHTS_PATH = Path("output/fgsm_weights.pth")


@dataclass(frozen=True)
class FGSMChallenge:
    label: int
    epsilon: float
    image_x01: np.ndarray


def linf_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def quantize_x01(x2d: np.ndarray) -> np.ndarray:
    # Match server PNG decode path to avoid local/server mismatch near epsilon boundary.
    return np.clip(np.round(x2d * 255.0) / 255.0, 0.0, 1.0).astype(np.float32)


def fetch_challenge(session: requests.Session, base_url: str) -> FGSMChallenge:
    resp = session.get(f"{base_url}/challenge", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return FGSMChallenge(
        label=int(data["label"]),
        epsilon=float(data["epsilon"]),
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


def build_fgsm_candidate(
    model: torch.nn.Module,
    original_x01: np.ndarray,
    label: int,
    epsilon: float,
    device: torch.device,
    alpha: float,
    direction: float,
) -> np.ndarray:
    x = torch.from_numpy(original_x01).float().unsqueeze(0).unsqueeze(0).to(device)
    x = x.clone().detach().requires_grad_(True)

    logits = model(x)
    y = torch.tensor([label], device=device)
    loss = F.cross_entropy(logits, y)

    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x.grad.sign()

    # Untargeted FGSM is +epsilon*sign(grad). We also allow direction=-1 fallback.
    step = float(alpha) * float(epsilon) * float(direction)
    adv = torch.clamp(x + step * grad_sign, 0.0, 1.0)
    adv_np = adv.detach().cpu().numpy()[0, 0].astype(np.float32)
    adv_np = quantize_x01(adv_np)

    # Final hard clamp in numpy space to keep perturbation exactly within epsilon.
    adv_np = np.clip(adv_np, original_x01 - epsilon, original_x01 + epsilon)
    adv_np = np.clip(adv_np, 0.0, 1.0).astype(np.float32)
    return adv_np


def find_valid_candidate(
    model: torch.nn.Module,
    challenge: FGSMChallenge,
    session: requests.Session,
    base_url: str,
    device: torch.device,
) -> Optional[np.ndarray]:
    # Try full epsilon first, then slightly smaller values for quantization edge cases.
    alpha_schedule = [1.0, 0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
    directions = [1.0, -1.0]

    for direction in directions:
        for alpha in alpha_schedule:
            candidate = build_fgsm_candidate(
                model=model,
                original_x01=challenge.image_x01,
                label=challenge.label,
                epsilon=challenge.epsilon,
                device=device,
                alpha=alpha,
                direction=direction,
            )

            local_pred, local_probs = local_predict(model, candidate, device)
            linf_val = linf_distance(candidate, challenge.image_x01)
            print(
                "Candidate stats: "
                f"direction={direction:+.0f}, alpha={alpha:.2f}, pred={local_pred}, "
                f"label_prob={local_probs[challenge.label]:.6f}, linf={linf_val:.6f}"
            )

            if linf_val > challenge.epsilon + 1e-8:
                continue

            if local_pred == challenge.label:
                continue

            server_out = server_predict(session, base_url, candidate)
            server_pred = int(server_out.get("pred", challenge.label))
            print(f"Server prediction: {server_out}")
            if server_pred != challenge.label:
                return candidate

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the FGSM attack challenge.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL), help="Challenge base URL")
    parser.add_argument("--weights-path", default=str(WEIGHTS_PATH), help="Path to cache downloaded weights")
    parser.add_argument("--force-download-weights", action="store_true", help="Re-download /weights even if cached")
    parser.add_argument("--save-adv", default="output/fgsm_challenge_adv.png", help="Path to save selected adversarial PNG")
    args = parser.parse_args()

    # Keep import-time reuse visible while preserving FGSM-specific default host.
    _ = EAD_DEFAULT_BASE_URL

    base_url = args.base_url.rstrip("/")
    device = torch.device("cpu")
    session = requests.Session()

    health = session.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    print(f"Health: {health.json()}")

    challenge = fetch_challenge(session, base_url)
    print("Challenge constraints:")
    print(f"  label: {challenge.label}")
    print(f"  epsilon: {challenge.epsilon}")

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

    clean_local_pred, clean_local_probs = local_predict(model, challenge.image_x01, device)
    print(f"Local clean prediction: {clean_local_pred}")
    print(f"Local clean label probability: {clean_local_probs[challenge.label]:.6f}")

    candidate = find_valid_candidate(model, challenge, session, base_url, device)
    if candidate is None:
        raise RuntimeError(
            "Failed to generate a valid FGSM adversarial sample. "
            "Try extending alpha_schedule or adding iterative FGSM fallback."
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
