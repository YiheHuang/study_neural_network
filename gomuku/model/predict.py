from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from env.board import Board, Player


def board_to_tensor(board: Board) -> torch.Tensor:
    size = board.size
    current = int(board.current_player)
    opponent = int(board.current_player.opponent())

    current_plane = (board.grid == current).astype(np.float32)
    opponent_plane = (board.grid == opponent).astype(np.float32)
    turn_plane = np.full((size, size), 1.0 if board.current_player == Player.BLACK else 0.0)

    stacked = np.stack([current_plane, opponent_plane, turn_plane], axis=0)
    return torch.from_numpy(stacked).float()


def legal_moves_mask(board: Board) -> np.ndarray:
    mask = np.zeros(board.size * board.size, dtype=np.float32)
    for row, col in board.legal_moves():
        idx = row * board.size + col
        mask[idx] = 1.0
    return mask


def _masked_policy_probs(board: Board, logits_1d: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits_1d, dim=0).detach().cpu().numpy().astype(np.float32)
    mask = legal_moves_mask(board)
    probs = probs * mask
    total = float(probs.sum())
    if total > 0.0:
        probs = probs / total
    else:
        legal_count = int(mask.sum())
        if legal_count > 0:
            probs = mask / legal_count
        else:
            probs = mask
    return probs.astype(np.float32)


@torch.no_grad()
def predict_policy_value(
    model: nn.Module, board: Board, device: str | torch.device = "cpu"
) -> Tuple[np.ndarray, float]:
    model = model.to(device)
    model.eval()

    x = board_to_tensor(board).unsqueeze(0).to(device)
    policy_logits, value = model(x)

    probs = _masked_policy_probs(board, policy_logits[0])
    value_scalar = float(value[0, 0].detach().cpu().item())
    return probs, value_scalar


@torch.no_grad()
def predict_policy_value_batch(
    model: nn.Module,
    boards: Sequence[Board],
    device: str | torch.device = "cpu",
) -> List[Tuple[np.ndarray, float]]:
    boards_list = list(boards)
    if not boards_list:
        return []

    model = model.to(device)
    model.eval()
    stacked = torch.stack([board_to_tensor(b) for b in boards_list], dim=0).to(device)
    policy_logits, value = model(stacked)

    logits_cpu = policy_logits.detach().cpu()
    values_cpu = value.detach().cpu().squeeze(-1)
    out: List[Tuple[np.ndarray, float]] = []
    for i, board in enumerate(boards_list):
        probs = _masked_policy_probs(board, logits_cpu[i])
        out.append((probs, float(values_cpu[i].item())))
    return out
