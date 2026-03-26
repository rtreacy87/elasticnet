"""
Solve the sparsity skill assessment challenge for CIFAR-10.

This script fetches all challenge items, loads the provided ResNet-18 style
classifier, generates a targeted adversarial example per item using either EAD
or JSMA as requested by the evaluator, performs a PNG round-trip to mirror the
server decode path, and submits the full payload to /submit_images.

Usage:
  python3 solve_sparsity_skill_assessment_challenge.py
  python3 solve_sparsity_skill_assessment_challenge.py --base-url "http://instance_ip:port"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

from src.attack import AttackConfig, ElasticNetAttack

DEFAULT_BASE_URL = "http://154.57.164.76:31213"
WEIGHTS_PATH = Path("output/cifar10_model.pth")
MIN_L2_THRESHOLD = 1.5


@dataclass(frozen=True)
class ChallengeItem:
    sample_id: int
    label: int
    target: int
    required_method: str
    image_x01: np.ndarray


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, num_blocks: Sequence[int] = (2, 2, 2, 2), num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        for current_stride in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, current_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class NormalizedModel(nn.Module):
    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.model((x01 - self.mean) / self.std)


def x01_from_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def b64_from_x01(x4d: np.ndarray) -> str:
    x = np.transpose(x4d[0], (1, 2, 0))
    x255 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_roundtrip_x01(x4d: np.ndarray) -> np.ndarray:
    return x01_from_b64(b64_from_x01(x4d))


def compute_l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.sum(diff ** 2)))


def count_modified_pixels(a: np.ndarray, b: np.ndarray, threshold: float = 1e-6) -> int:
    return int(np.sum(np.abs(a - b) > threshold))


def fetch_health(session: requests.Session, base_url: str) -> Dict[str, object]:
    response = session.get(f"{base_url}/health", timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_model_meta(session: requests.Session, base_url: str) -> Dict[str, object]:
    response = session.get(f"{base_url}/model", timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_challenge(session: requests.Session, base_url: str) -> List[ChallengeItem]:
    response = session.get(f"{base_url}/challenge", timeout=15)
    response.raise_for_status()
    payload = response.json()
    items = []
    for item in payload["items"]:
        items.append(
            ChallengeItem(
                sample_id=int(item["sample_id"]),
                label=int(item["label"]),
                target=int(item["target"]),
                required_method=str(item["required_method"]),
                image_x01=x01_from_b64(item["image_b64"]),
            )
        )
    return items


def download_model_weights(
    session: requests.Session,
    base_url: str,
    meta: Dict[str, object],
    output_path: Path,
    force: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    weights_url = str(meta["weights_url"])
    response = session.get(f"{base_url}{weights_url}", timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def local_predict(model: nn.Module, x4d: np.ndarray, device: torch.device) -> Tuple[int, np.ndarray]:
    tensor = torch.from_numpy(x4d).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
    return pred, probs


def server_predict(session: requests.Session, base_url: str, x4d: np.ndarray) -> Dict[str, object]:
    payload = {"image_b64": b64_from_x01(x4d)}
    response = session.post(f"{base_url}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_server_prediction(payload: Dict[str, object]) -> int:
    if "predicted_class" in payload:
        return int(payload["predicted_class"])
    if "pred" in payload:
        return int(payload["pred"])
    raise KeyError(f"Could not find prediction field in server payload: {payload}")


def strengthen_candidate(
    model: nn.Module,
    original: np.ndarray,
    candidate: np.ndarray,
    target_class: int,
    device: torch.device,
    min_l2: float = MIN_L2_THRESHOLD,
) -> Optional[np.ndarray]:
    candidate = png_roundtrip_x01(candidate)
    initial_pred, _ = local_predict(model, candidate, device)
    initial_l2 = compute_l2(original, candidate)
    if initial_pred == target_class and initial_l2 >= min_l2:
        return candidate

    direction = candidate - original
    if np.allclose(direction, 0.0):
        return None

    best: Optional[np.ndarray] = None
    for factor in np.linspace(1.0, 3.0, 17):
        scaled = np.clip(original + factor * direction, 0.0, 1.0).astype(np.float32)
        scaled = png_roundtrip_x01(scaled)
        pred, _ = local_predict(model, scaled, device)
        l2_value = compute_l2(original, scaled)
        if pred == target_class and l2_value >= min_l2:
            best = scaled
            break
    return best


def run_ead_targeted(
    model: nn.Module,
    item: ChallengeItem,
    device: torch.device,
) -> Optional[np.ndarray]:
    schedules = [
        AttackConfig(beta=0.001, confidence=0.0, learning_rate=0.01, max_iterations=250, binary_search_steps=5, initial_const=0.001),
        AttackConfig(beta=0.005, confidence=0.5, learning_rate=0.01, max_iterations=400, binary_search_steps=6, initial_const=0.003),
        AttackConfig(beta=0.01, confidence=1.0, learning_rate=0.005, max_iterations=600, binary_search_steps=7, initial_const=0.01),
    ]

    x = torch.from_numpy(item.image_x01).float().to(device)
    y = torch.tensor([item.target], device=device)

    for config in schedules:
        attacker = ElasticNetAttack(model=model, config=config, device=device)
        result = attacker.run(x, y, targeted=True)
        candidate = result.adversarial_images[0].detach().cpu().numpy()[None, ...].astype(np.float32)
        candidate = np.clip(candidate, 0.0, 1.0)
        candidate = png_roundtrip_x01(candidate)
        pred, probs = local_predict(model, candidate, device)
        l2_value = compute_l2(item.image_x01, candidate)
        print(
            f"    EAD candidate: pred={pred}, target_prob={probs[item.target]:.6f}, l2={l2_value:.6f}"
        )
        if pred == item.target:
            strengthened = strengthen_candidate(model, item.image_x01, candidate, item.target, device)
            if strengthened is not None:
                return strengthened
    return None


def compute_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    outputs = model(x)
    jacobian_rows = []
    for class_idx in range(outputs.shape[1]):
        model.zero_grad(set_to_none=True)
        grad = torch.autograd.grad(outputs[0, class_idx], x, retain_graph=class_idx < outputs.shape[1] - 1)[0]
        jacobian_rows.append(grad.detach().view(-1))
    return torch.stack(jacobian_rows, dim=0)


def select_salient_features(
    jacobian: torch.Tensor,
    target_class: int,
    search_space: torch.Tensor,
    increase: bool,
) -> Optional[List[int]]:
    candidate_indices = torch.nonzero(search_space, as_tuple=False).view(-1)
    if candidate_indices.numel() == 0:
        return None

    target_grad = jacobian[target_class, candidate_indices]
    other_grad = jacobian[:, candidate_indices].sum(dim=0) - target_grad

    if candidate_indices.numel() >= 2:
        alpha = target_grad[:, None] + target_grad[None, :]
        beta = other_grad[:, None] + other_grad[None, :]
        diagonal = torch.eye(candidate_indices.numel(), dtype=torch.bool, device=jacobian.device)
        valid = (alpha > 0) & (beta < 0) & (~diagonal) if increase else (alpha < 0) & (beta > 0) & (~diagonal)
        if valid.any():
            saliency = torch.full_like(alpha, float("-inf"))
            saliency[valid] = -alpha[valid] * beta[valid]
            best_pair = torch.argmax(saliency)
            row = int(best_pair // candidate_indices.numel())
            col = int(best_pair % candidate_indices.numel())
            return [int(candidate_indices[row].item()), int(candidate_indices[col].item())]

    valid_single = (target_grad > 0) & (other_grad < 0) if increase else (target_grad < 0) & (other_grad > 0)
    if not valid_single.any():
        return None
    scores = torch.full_like(target_grad, float("-inf"))
    scores[valid_single] = -target_grad[valid_single] * other_grad[valid_single]
    best_idx = int(candidate_indices[int(torch.argmax(scores).item())].item())
    return [best_idx]


def run_jsma_targeted(
    model: nn.Module,
    item: ChallengeItem,
    device: torch.device,
) -> Optional[np.ndarray]:
    schedules: Sequence[Tuple[float, int]] = [
        (1.0, 128),
        (-1.0, 128),
        (0.5, 256),
        (-0.5, 256),
    ]

    original = torch.from_numpy(item.image_x01).float().to(device)

    for theta, max_iterations in schedules:
        adversarial = original.clone().detach()
        flat = adversarial.view(-1)
        search_space = torch.ones_like(flat, dtype=torch.bool)

        for _ in range(max_iterations):
            candidate = png_roundtrip_x01(adversarial.detach().cpu().numpy().astype(np.float32))
            pred, probs = local_predict(model, candidate, device)
            if pred == item.target:
                strengthened = strengthen_candidate(model, item.image_x01, candidate, item.target, device)
                if strengthened is not None:
                    print(
                        f"    JSMA candidate: pred={pred}, target_prob={probs[item.target]:.6f}, "
                        f"l2={compute_l2(item.image_x01, strengthened):.6f}, pixels={count_modified_pixels(item.image_x01, strengthened)}"
                    )
                    return strengthened

            search_space &= flat < 1.0 - 1e-6 if theta > 0 else flat > 1e-6
            jacobian = compute_jacobian(model, adversarial)
            salient = select_salient_features(jacobian, item.target, search_space, increase=theta > 0)
            if salient is None:
                break

            for feature_idx in salient:
                flat[feature_idx] = torch.clamp(flat[feature_idx] + theta, 0.0, 1.0)
                search_space[feature_idx] = False

        final_candidate = png_roundtrip_x01(adversarial.detach().cpu().numpy().astype(np.float32))
        pred, probs = local_predict(model, final_candidate, device)
        if pred == item.target:
            strengthened = strengthen_candidate(model, item.image_x01, final_candidate, item.target, device)
            if strengthened is not None:
                print(
                    f"    JSMA final candidate: pred={pred}, target_prob={probs[item.target]:.6f}, "
                    f"l2={compute_l2(item.image_x01, strengthened):.6f}, pixels={count_modified_pixels(item.image_x01, strengthened)}"
                )
                return strengthened
    return None


def save_candidate_image(path: Path, x4d: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.transpose(x4d[0], (1, 2, 0))
    x255 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(x255, mode="RGB").save(path)


def solve_item(
    model: nn.Module,
    item: ChallengeItem,
    session: requests.Session,
    base_url: str,
    device: torch.device,
    save_dir: Path,
) -> Dict[str, object]:
    print(
        f"Solving sample_id={item.sample_id} with method={item.required_method}, "
        f"label={item.label}, target={item.target}"
    )
    clean_pred, clean_probs = local_predict(model, item.image_x01, device)
    print(f"  Local clean pred={clean_pred}, target_prob={clean_probs[item.target]:.6f}")

    requested_method = item.required_method.lower()
    if requested_method == "ead":
        methods_to_try = ["ead"]
    elif requested_method == "jsma":
        methods_to_try = ["jsma"]
    elif requested_method == "either":
        methods_to_try = ["ead", "jsma"]
    else:
        raise ValueError(f"Unsupported required_method: {item.required_method}")

    candidate = None
    method_used = None
    server_result = None

    for method_name in methods_to_try:
        print(f"  Trying method: {method_name}")
        if method_name == "ead":
            candidate = run_ead_targeted(model, item, device)
        else:
            candidate = run_jsma_targeted(model, item, device)

        if candidate is None:
            print(f"  Method {method_name} did not produce a valid local candidate.")
            continue

        server_result = server_predict(session, base_url, candidate)
        server_pred = extract_server_prediction(server_result)
        l2_value = compute_l2(item.image_x01, candidate)
        pixels = count_modified_pixels(item.image_x01, candidate)
        print(f"  Server prediction: {server_result}")
        print(f"  Final metrics: l2={l2_value:.6f}, pixels_modified={pixels}")

        if server_pred == item.target:
            method_used = method_name
            break

        print(
            f"  Method {method_name} failed server validation for sample {item.sample_id}: "
            f"got {server_pred}, expected {item.target}"
        )
        candidate = None

    if candidate is None or method_used is None or server_result is None:
        raise RuntimeError(f"Failed to generate valid adversarial example for sample {item.sample_id}")

    save_path = save_dir / f"sample_{item.sample_id}_{method_used}.png"
    save_candidate_image(save_path, candidate)
    print(f"  Saved candidate to: {save_path}")

    return {
        "sample_id": item.sample_id,
        "method": method_used,
        "image_b64": b64_from_x01(candidate),
    }


def submit_images(session: requests.Session, base_url: str, items: List[Dict[str, object]]) -> Dict[str, object]:
    response = session.post(f"{base_url}/submit_images", json={"items": items}, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve the sparsity skill assessment challenge.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL), help="Challenge base URL")
    parser.add_argument("--weights-path", default=str(WEIGHTS_PATH), help="Path to cache downloaded model weights")
    parser.add_argument("--force-download-weights", action="store_true", help="Re-download weights even if cached")
    parser.add_argument("--save-dir", default="output/sparsity_skill_assessment", help="Directory for generated PNGs")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session = requests.Session()

    print(f"Using device: {device}")
    health = fetch_health(session, base_url)
    print(f"Health: {health}")

    meta = fetch_model_meta(session, base_url)
    print(f"Model metadata: {meta}")
    challenge_items = fetch_challenge(session, base_url)
    print(f"Loaded {len(challenge_items)} challenge items")

    weights_path = download_model_weights(
        session=session,
        base_url=base_url,
        meta=meta,
        output_path=Path(args.weights_path),
        force=args.force_download_weights,
    )
    print(f"Weights ready: {weights_path}")

    backbone = ResNetCIFAR().to(device)
    state = torch.load(weights_path, map_location=device)
    state_dict = state.get("state_dict_ema") or state.get("state_dict") or state
    backbone.load_state_dict(state_dict)
    backbone.eval()

    normalize = meta.get("normalize", {})
    mean = normalize.get("mean", [0.4914, 0.4822, 0.4465])
    std = normalize.get("std", [0.2470, 0.2435, 0.2616])
    model = NormalizedModel(backbone, mean=mean, std=std).to(device)
    model.eval()

    submission_items: List[Dict[str, object]] = []
    save_dir = Path(args.save_dir)

    for item in challenge_items:
        submission_items.append(solve_item(model, item, session, base_url, device, save_dir))

    print("\nSubmitting all items...")
    response = submit_images(session, base_url, submission_items)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()