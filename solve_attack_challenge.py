"""
Solve the ElasticNet challenge endpoint using the existing EAD framework.

Usage:
  python3 solve_attack_challenge.py --base-url "http://instance_ip:port"

Environment fallback:
  BASE_URL=http://instance_ip:port python3 solve_attack_challenge.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

from src.attack import AttackConfig, ElasticNetAttack

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
DEFAULT_BASE_URL = "http://154.57.164.65:30801"
WEIGHTS_PATH = Path("output/elasticnet_weights.pth")


@dataclass(frozen=True)
class Challenge:
    label: int
    beta: float
    elastic_max: float
    l2_max: float
    l1_max: float
    image_x01: np.ndarray


class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class NormalizedModel(nn.Module):
    """Wrapper so EAD runs in [0,1] pixel space while model sees normalized inputs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x_norm = (x01 - MNIST_MEAN) / MNIST_STD
        return self.model(x_norm)


def x01_from_b64_png(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError(f"Expected 28x28 image, got {img.size}")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def b64_png_from_x01(x2d: np.ndarray) -> str:
    x255 = np.clip(np.round(x2d * 255.0), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compute_metrics(orig_x01: np.ndarray, adv_x01: np.ndarray, beta: float) -> Dict[str, float]:
    diff = adv_x01 - orig_x01
    l1 = float(np.sum(np.abs(diff)))
    l2 = float(np.sqrt(np.sum(diff ** 2)))
    linf = float(np.max(np.abs(diff)))
    elastic = l2 + beta * l1
    return {"l1": l1, "l2": l2, "linf": linf, "elastic": elastic}


def within_constraints(metrics: Dict[str, float], challenge: Challenge, eps: float = 1e-8) -> bool:
    return (
        metrics["elastic"] <= challenge.elastic_max + eps
        and metrics["l2"] <= challenge.l2_max + eps
        and metrics["l1"] <= challenge.l1_max + eps
    )


def fetch_challenge(session: requests.Session, base_url: str) -> Challenge:
    resp = session.get(f"{base_url}/challenge", timeout=15)
    resp.raise_for_status()
    data = resp.json()

    return Challenge(
        label=int(data["label"]),
        beta=float(data["beta"]),
        elastic_max=float(data["elastic_max"]),
        l2_max=float(data["l2_max"]),
        l1_max=float(data["l1_max"]),
        image_x01=x01_from_b64_png(data["image_b64"]),
    )


def download_weights(session: requests.Session, base_url: str, output_path: Path, force: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    resp = session.get(f"{base_url}/weights", timeout=30)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return output_path


def local_predict(model: nn.Module, x01_2d: np.ndarray, device: torch.device) -> int:
    x = torch.from_numpy(x01_2d).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = int(model(x).argmax(dim=1).item())
    return pred


def server_predict(session: requests.Session, base_url: str, x01_2d: np.ndarray) -> Dict[str, float]:
    payload = {"image_b64": b64_png_from_x01(x01_2d)}
    resp = session.post(f"{base_url}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def line_search_candidate(
    model: nn.Module,
    orig_x01: np.ndarray,
    adv_x01: np.ndarray,
    label: int,
    challenge: Challenge,
    device: torch.device,
) -> Optional[Tuple[np.ndarray, Dict[str, float], int]]:
    """Move back toward the original image to satisfy bounds with minimal distortion."""
    best: Optional[Tuple[np.ndarray, Dict[str, float], int]] = None

    for alpha in np.linspace(1.0, 0.0, 101):
        cand = np.clip(orig_x01 + alpha * (adv_x01 - orig_x01), 0.0, 1.0).astype(np.float32)
        pred = local_predict(model, cand, device)
        metrics = compute_metrics(orig_x01, cand, challenge.beta)

        if pred != label and within_constraints(metrics, challenge):
            best = (cand, metrics, pred)

    return best


def run_attack_once(
    model: nn.Module,
    challenge: Challenge,
    config: AttackConfig,
    device: torch.device,
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    attacker = ElasticNetAttack(model=model, config=config, device=device)

    x = torch.from_numpy(challenge.image_x01).float().unsqueeze(0).unsqueeze(0).to(device)
    y = torch.tensor([challenge.label], device=device)

    result = attacker.run(x, y, targeted=False)
    adv = result.adversarial_images[0].detach().cpu().numpy().astype(np.float32)
    adv = np.clip(adv[0], 0.0, 1.0)

    pred = int(result.adv_predictions[0].item())
    metrics = compute_metrics(challenge.image_x01, adv, challenge.beta)
    return adv, pred, metrics


def build_config_schedule(beta: float) -> Iterable[AttackConfig]:
    # Wider schedule helps adapt to unknown per-instance bounds.
    settings = [
        (0.0, 0.01, 150, 4, 0.0005),
        (0.0, 0.01, 300, 6, 0.0010),
        (0.0, 0.005, 500, 7, 0.0030),
        (0.5, 0.01, 400, 7, 0.0030),
        (1.0, 0.01, 600, 8, 0.0100),
        (2.0, 0.005, 800, 9, 0.0200),
    ]

    for confidence, lr, max_iters, bs_steps, init_const in settings:
        yield AttackConfig(
            beta=beta,
            confidence=confidence,
            learning_rate=lr,
            max_iterations=max_iters,
            binary_search_steps=bs_steps,
            initial_const=init_const,
            clip_min=0.0,
            clip_max=1.0,
        )


def submit_candidate(
    session: requests.Session,
    base_url: str,
    x01_2d: np.ndarray,
) -> requests.Response:
    payload = {"image_b64": b64_png_from_x01(x01_2d)}
    return session.post(f"{base_url}/submit", json=payload, timeout=15)


def print_constraints(ch: Challenge) -> None:
    print("Challenge constraints:")
    print(f"  label: {ch.label}")
    print(f"  beta: {ch.beta}")
    print(f"  elastic_max: {ch.elastic_max}")
    print(f"  l2_max: {ch.l2_max}")
    print(f"  l1_max: {ch.l1_max}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve ElasticNet challenge using existing EAD framework.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL), help="Challenge base URL")
    parser.add_argument("--weights-path", default=str(WEIGHTS_PATH), help="Path to cache downloaded weights")
    parser.add_argument("--force-download-weights", action="store_true", help="Re-download /weights even if file exists")
    parser.add_argument("--save-adv", default="output/challenge_adv.png", help="Path to save selected adversarial PNG")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    device = torch.device("cpu")

    session = requests.Session()

    health = session.get(f"{base_url}/health", timeout=10)
    health.raise_for_status()
    print(f"Health: {health.json()}")

    challenge = fetch_challenge(session, base_url)
    print_constraints(challenge)

    clean_pred_server = server_predict(session, base_url, challenge.image_x01)
    print(f"Server clean prediction: {clean_pred_server}")

    weights_path = download_weights(
        session,
        base_url,
        Path(args.weights_path),
        force=args.force_download_weights,
    )
    print(f"Weights ready: {weights_path}")

    backbone = SimpleClassifier().to(device)
    state = torch.load(weights_path, map_location=device)
    backbone.load_state_dict(state)
    backbone.eval()

    model = NormalizedModel(backbone).to(device)
    model.eval()

    clean_pred_local = local_predict(model, challenge.image_x01, device)
    print(f"Local clean prediction: {clean_pred_local}")

    best_candidate: Optional[Tuple[np.ndarray, Dict[str, float], int]] = None

    for idx, config in enumerate(build_config_schedule(challenge.beta), start=1):
        print("\n" + "-" * 70)
        print(
            f"Attempt {idx}: conf={config.confidence}, lr={config.learning_rate}, "
            f"iters={config.max_iterations}, bs={config.binary_search_steps}, c0={config.initial_const}"
        )

        adv_x01, adv_pred_local, metrics = run_attack_once(model, challenge, config, device)
        print(f"Raw candidate local pred={adv_pred_local}, metrics={json.dumps(metrics)}")

        if adv_pred_local != challenge.label and within_constraints(metrics, challenge):
            best_candidate = (adv_x01, metrics, adv_pred_local)
            print("Raw candidate already satisfies constraints locally.")
        else:
            refined = line_search_candidate(
                model=model,
                orig_x01=challenge.image_x01,
                adv_x01=adv_x01,
                label=challenge.label,
                challenge=challenge,
                device=device,
            )
            if refined is not None:
                best_candidate = refined
                print(f"Refined candidate valid locally: pred={refined[2]}, metrics={json.dumps(refined[1])}")

        if best_candidate is None:
            continue

        # Verify with server prediction before submit.
        server_pred = server_predict(session, base_url, best_candidate[0])
        print(f"Server prediction for candidate: {server_pred}")

        if int(server_pred["pred"]) == challenge.label:
            print("Server still predicts baseline label. Continuing search.")
            best_candidate = None
            continue

        resp = submit_candidate(session, base_url, best_candidate[0])
        if resp.status_code == 200:
            data = resp.json()
            print("\nSUCCESS")
            print(json.dumps(data, indent=2))

            save_path = Path(args.save_adv)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            x255 = np.clip(np.round(best_candidate[0] * 255.0), 0, 255).astype(np.uint8)
            Image.fromarray(x255, mode="L").save(save_path)
            print(f"Saved adversarial image to: {save_path}")
            return

        print(f"Submit rejected ({resp.status_code}): {resp.text}")
        best_candidate = None

    raise RuntimeError(
        "Failed to find a valid adversarial example with current schedule. "
        "Try increasing max_iterations/binary_search_steps or adjust initial_const schedule."
    )


if __name__ == "__main__":
    main()
