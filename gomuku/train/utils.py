from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

Sample = Tuple[np.ndarray, np.ndarray, float]

def resolve_device(preferred: str | torch.device) -> torch.device:
    """
    Prefer CUDA, fallback to CPU if unavailable.

    - preferred: "cuda", "cpu", "cuda:0", torch.device(...)
    """
    dev = torch.device(preferred)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def device_banner(requested: str | torch.device, resolved: torch.device) -> str:
    if torch.device(requested).type == resolved.type:
        return f"device={resolved}"
    return f"device={resolved} (requested {requested}, fallback applied)"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_tensors(
    dataset: List[Sample] | Deque[Sample], device: str | torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = np.stack([row[0] for row in dataset], axis=0).astype(np.float32)
    policies = np.stack([row[1] for row in dataset], axis=0).astype(np.float32)
    values = np.array([row[2] for row in dataset], dtype=np.float32).reshape(-1, 1)
    return (
        torch.from_numpy(states).to(device),
        torch.from_numpy(policies).to(device),
        torch.from_numpy(values).to(device),
    )


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    batch_size: int,
) -> float:
    model.train()
    num_samples = states.shape[0]
    perm = torch.randperm(num_samples, device=states.device)
    total_loss = 0.0
    total_batches = 0

    for start in range(0, num_samples, batch_size):
        idx = perm[start : start + batch_size]
        s = states[idx]
        p = policy_targets[idx]
        v = value_targets[idx]

        logits, value_pred = model(s)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(p * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value_pred, v)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(1, total_batches)


def load_model_weights(model: nn.Module, path: Path) -> bool:
    if not path.exists():
        return False
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    try:
        model.load_state_dict(state)
        return True
    except RuntimeError:
        return False
