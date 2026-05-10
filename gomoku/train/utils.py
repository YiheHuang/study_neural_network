from __future__ import annotations

import contextlib
import json
from collections import deque
from pathlib import Path
from typing import Deque, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

Sample = Tuple[np.ndarray, np.ndarray, float]


def augment_batch(
    states: torch.Tensor, policies: torch.Tensor, board_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 batch 中每个样本随机应用8重对称变换之一 (4旋转 × 2翻转)。"""
    batch = states.shape[0]
    aug_states = states.clone()
    aug_policies = policies.clone()
    for i in range(batch):
        k = int(torch.randint(4, (1,)).item())  # 旋转次数
        flip = torch.rand(1).item() > 0.5
        # 旋转 state: (C, H, W) -> rot90 on (H, W) = dims (1, 2)
        s = torch.rot90(aug_states[i], k, dims=(1, 2))
        p = aug_policies[i].reshape(board_size, board_size)
        p = torch.rot90(p, k, dims=(0, 1))
        if flip:
            s = torch.flip(s, dims=[2])  # 水平翻转
            p = torch.flip(p, dims=[1])
        aug_states[i] = s
        aug_policies[i] = p.reshape(-1)
    return aug_states, aug_policies

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
    sample_weights: torch.Tensor | None = None,
    board_size: int = 9,
    augment: bool = True,
    use_amp: bool = False,
) -> float:
    model.train()
    num_samples = states.shape[0]
    device = states.device
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None
    if sample_weights is not None:
        if sample_weights.shape[0] != num_samples:
            raise ValueError("sample_weights length must match num_samples")
        weights = sample_weights.to(device)
        weights = torch.clamp(weights, min=1e-8)
        perm = torch.multinomial(weights, num_samples=num_samples, replacement=True)
    else:
        perm = torch.randperm(num_samples, device=device)
    total_loss = 0.0
    total_batches = 0

    for start in range(0, num_samples, batch_size):
        idx = perm[start : start + batch_size]
        s = states[idx]
        p = policy_targets[idx]
        v = value_targets[idx]

        if augment:
            s, p = augment_batch(s, p, board_size)
            # 颜色翻转增强：交换两通道 + value取反，保证黑白完全对称
            s_flip = torch.stack([s[:, 1], s[:, 0]], dim=1)
            v_flip = -v
            s = torch.cat([s, s_flip], dim=0)
            p = torch.cat([p, p], dim=0)
            v = torch.cat([v, v_flip], dim=0)

        amp_ctx = (
            torch.amp.autocast("cuda")
            if scaler is not None
            else contextlib.nullcontext()
        )
        with amp_ctx:
            logits, value_pred = model(s)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(p * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, v)
            loss = policy_loss + value_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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


def build_decay_weights(
    birth_iters: Sequence[int], current_iter: int, decay: float
) -> torch.Tensor:
    if decay <= 0:
        return torch.ones(len(birth_iters), dtype=torch.float32)
    ages = np.maximum(0, current_iter - np.array(birth_iters, dtype=np.int64))
    weights = np.exp(-decay * ages).astype(np.float32)
    return torch.from_numpy(weights)


def save_replay_buffer(path: Path, samples: Sequence[Sample], birth_iters: Sequence[int]) -> None:
    if len(samples) != len(birth_iters):
        raise ValueError("samples and birth_iters length mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        np.savez_compressed(
            path,
            states=np.empty((0, 2, 1, 1), dtype=np.float32),
            policies=np.empty((0, 1), dtype=np.float32),
            values=np.empty((0,), dtype=np.float32),
            birth_iters=np.empty((0,), dtype=np.int32),
        )
        return
    states = np.stack([s[0] for s in samples], axis=0).astype(np.float32)
    policies = np.stack([s[1] for s in samples], axis=0).astype(np.float32)
    values = np.array([s[2] for s in samples], dtype=np.float32)
    birth = np.array(birth_iters, dtype=np.int32)
    np.savez_compressed(
        path, states=states, policies=policies, values=values, birth_iters=birth
    )


def load_replay_buffer(path: Path) -> Tuple[List[Sample], List[int]]:
    if not path.exists():
        return [], []
    blob = np.load(path)
    states = blob["states"]
    policies = blob["policies"]
    values = blob["values"]
    birth = blob["birth_iters"]
    count = int(values.shape[0])
    samples: List[Sample] = [
        (states[i].astype(np.float32), policies[i].astype(np.float32), float(values[i]))
        for i in range(count)
    ]
    return samples, [int(x) for x in birth.tolist()]
