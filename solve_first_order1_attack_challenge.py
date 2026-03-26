"""
Solve Skills Assessment 1 (targeted I-FGSM on CIFAR-10).

Usage:
  python3 solve_first_order1_attack_challenge.py
  python3 solve_first_order1_attack_challenge.py --base-url "http://instance_ip:port"

Environment fallback:
  BASE_URL=http://instance_ip:port python3 solve_first_order1_attack_challenge.py
"""

from __future__ import annotations

import argparse
import base64
import io
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

from solve_sparsity_skill_assessment_challenge import NormalizedModel

DEFAULT_BASE_URL = "http://154.57.164.66:30607"
WEIGHTS_PATH = Path("output/cifar10_model_best_first_order1.pth")

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class FirstOrderChallenge:
    original_class: int
    target_class: int
    epsilon: float
    image_x01: np.ndarray  # shape (1, 3, 32, 32)
    mean: Sequence[float]
    std: Sequence[float]
    max_iterations_hint: int


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


def x01_from_b64_png_rgb(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (32, 32):
        raise ValueError(f"Expected 32x32 image, got {img.size}")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def b64_png_from_x01_rgb(x4d: np.ndarray) -> str:
    x = np.transpose(x4d[0], (1, 2, 0))
    x255 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def quantize_x01(x4d: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x4d * 255.0) / 255.0, 0.0, 1.0).astype(np.float32)


def linf_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def fetch_challenge(session: requests.Session, base_url: str) -> FirstOrderChallenge:
    resp = session.get(f"{base_url}/challenge", timeout=20)
    resp.raise_for_status()
    data = resp.json()

    normalization = data.get("normalization", {})
    mean = normalization.get("mean", [0.4914, 0.4822, 0.4465])
    std = normalization.get("std", [0.2470, 0.2435, 0.2616])

    return FirstOrderChallenge(
        original_class=int(data["original_class"]),
        target_class=int(data["target_class"]),
        epsilon=float(data["epsilon"]),
        image_x01=x01_from_b64_png_rgb(data["image"]),
        mean=mean,
        std=std,
        max_iterations_hint=int(data.get("max_iterations_hint", 100)),
    )


def download_weights(session: requests.Session, base_url: str, output_path: Path, force: bool) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    resp = session.get(f"{base_url}/model/weights", timeout=60)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return output_path


def load_model(weights_path: Path, mean: Sequence[float], std: Sequence[float], device: torch.device) -> nn.Module:
    backbone = CIFAR10CNN(num_classes=10).to(device)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    backbone.load_state_dict(state_dict)
    backbone.eval()

    model = NormalizedModel(backbone, mean=mean, std=std).to(device)
    model.eval()
    return model


def local_predict(model: nn.Module, x4d: np.ndarray, device: torch.device) -> Tuple[int, np.ndarray]:
    x = torch.from_numpy(x4d).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs


def server_predict(session: requests.Session, base_url: str, x4d: np.ndarray) -> Dict[str, object]:
    payload = {"image": b64_png_from_x01_rgb(x4d)}
    resp = session.post(f"{base_url}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def submit_candidate(session: requests.Session, base_url: str, x4d: np.ndarray) -> requests.Response:
    payload = {"image": b64_png_from_x01_rgb(x4d)}
    return session.post(f"{base_url}/submit", json=payload, timeout=30)


def ifgsm_targeted(
    model: nn.Module,
    original_x01: np.ndarray,
    target_class: int,
    epsilon: float,
    device: torch.device,
    num_steps: int,
    alpha_scale: float,
) -> np.ndarray:
    x0 = torch.from_numpy(original_x01).float().to(device)
    x_adv = x0.clone().detach()
    target = torch.tensor([target_class], dtype=torch.long, device=device)

    alpha = alpha_scale * (epsilon / max(1, num_steps))

    for _ in range(num_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, target)

        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        # Targeted update: minimize loss to target class.
        grad_sign = x_adv.grad.sign()
        x_next = x_adv.detach() - alpha * grad_sign

        # Project to L_inf ball around original image and valid range.
        x_next = torch.max(torch.min(x_next, x0 + epsilon), x0 - epsilon)
        x_next = torch.clamp(x_next, 0.0, 1.0)
        x_adv = x_next

    candidate = x_adv.detach().cpu().numpy().astype(np.float32)
    candidate = quantize_x01(candidate)
    # Re-enforce bound after quantization.
    candidate = np.clip(candidate, original_x01 - epsilon, original_x01 + epsilon)
    candidate = np.clip(candidate, 0.0, 1.0).astype(np.float32)
    return candidate


def find_valid_candidate(
    model: nn.Module,
    challenge: FirstOrderChallenge,
    session: requests.Session,
    base_url: str,
    device: torch.device,
) -> Optional[np.ndarray]:
    steps_schedule = [
        max(10, challenge.max_iterations_hint // 10),
        max(20, challenge.max_iterations_hint // 5),
        max(40, challenge.max_iterations_hint // 2),
        challenge.max_iterations_hint,
        challenge.max_iterations_hint + 50,
    ]
    steps_schedule = sorted(set(steps_schedule))
    alpha_scales = [1.0, 1.25, 1.5, 0.75]

    for num_steps in steps_schedule:
        for alpha_scale in alpha_scales:
            candidate = ifgsm_targeted(
                model=model,
                original_x01=challenge.image_x01,
                target_class=challenge.target_class,
                epsilon=challenge.epsilon,
                device=device,
                num_steps=num_steps,
                alpha_scale=alpha_scale,
            )

            local_pred, local_probs = local_predict(model, candidate, device)
            linf_val = linf_distance(candidate, challenge.image_x01)
            print(
                f"Candidate: steps={num_steps}, alpha_scale={alpha_scale:.2f}, "
                f"pred={local_pred}, target_prob={local_probs[challenge.target_class]:.6f}, linf={linf_val:.6f}"
            )

            if linf_val > challenge.epsilon + 1e-8:
                continue

            if local_pred != challenge.target_class:
                continue

            server_out = server_predict(session, base_url, candidate)
            server_pred = int(server_out.get("predicted_class", -1))
            print(f"Server prediction: {server_out}")
            if server_pred == challenge.target_class:
                return candidate

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Skills Assessment 1 (targeted I-FGSM on CIFAR-10).")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL), help="Challenge base URL")
    parser.add_argument("--weights-path", default=str(WEIGHTS_PATH), help="Path to cache downloaded weights")
    parser.add_argument("--force-download-weights", action="store_true", help="Re-download /model/weights even if cached")
    parser.add_argument("--save-adv", default="output/first_order1_challenge_adv.png", help="Path to save selected adversarial PNG")
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
    print(f"  target_class: {challenge.target_class} ({CIFAR10_CLASSES[challenge.target_class]})")
    print(f"  epsilon: {challenge.epsilon}")
    print(f"  max_iterations_hint: {challenge.max_iterations_hint}")

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
    print(f"Local target-class probability: {clean_local_probs[challenge.target_class]:.6f}")

    candidate = find_valid_candidate(model, challenge, session, base_url, device)
    if candidate is None:
        raise RuntimeError(
            "Failed to generate a valid targeted I-FGSM candidate. "
            "Try expanding steps_schedule or alpha_scales."
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
