from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from env.board import Board, Player
from model.network import GomokuNet


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


@torch.no_grad()
def predict_policy_value(
    model: GomokuNet, board: Board, device: str | torch.device = "cpu"
) -> Tuple[np.ndarray, float]:
    model = model.to(device)
    model.eval()

    x = board_to_tensor(board).unsqueeze(0).to(device)
    policy_logits, value = model(x)

    probs = torch.softmax(policy_logits[0], dim=0).detach().cpu().numpy()
    mask = legal_moves_mask(board)
    probs = probs * mask

    total = float(probs.sum())
    if total > 0.0:
        probs = probs / total
    else:
        # Should be rare; fallback to uniform over legal moves.
        legal_count = int(mask.sum())
        if legal_count > 0:
            probs = mask / legal_count
        else:
            probs = mask

    value_scalar = float(value[0, 0].detach().cpu().item())
    return probs.astype(np.float32), value_scalar
