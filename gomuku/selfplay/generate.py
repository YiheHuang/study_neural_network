from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import torch

from env.board import Board, GameResult, Player
from mcts import MCTS
from model.predict import board_to_tensor


@dataclass
class Sample:
    state: np.ndarray
    policy: np.ndarray
    player: int


def _result_to_winner(result: GameResult) -> int:
    if result == GameResult.BLACK_WIN:
        return int(Player.BLACK)
    if result == GameResult.WHITE_WIN:
        return int(Player.WHITE)
    return int(Player.EMPTY)


def play_self_game(
    model: torch.nn.Module,
    board_size: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device = "cpu",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], GameResult, int]:
    board = Board(size=board_size)
    mcts = MCTS(model=model, c_puct=c_puct)
    history: List[Sample] = []

    while board.game_result() == GameResult.ONGOING:
        state = board_to_tensor(board).cpu().numpy()
        move, policy = mcts.run(board, simulations=simulations, device=device)
        history.append(Sample(state=state, policy=policy, player=int(board.current_player)))
        board.place_stone(*move)

    winner = _result_to_winner(board.game_result())
    data: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for item in history:
        if winner == int(Player.EMPTY):
            value = 0.0
        elif item.player == winner:
            value = 1.0
        else:
            value = -1.0
        data.append((item.state, item.policy, value))
    return data, board.game_result(), len(history)


def generate_selfplay_data(
    model: torch.nn.Module,
    num_games: int,
    board_size: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device = "cpu",
    on_game_end: Callable[[int, int, GameResult, int], None] | None = None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    dataset: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for game_idx in range(num_games):
        game_data, game_result, steps = play_self_game(
            model=model,
            board_size=board_size,
            simulations=simulations,
            c_puct=c_puct,
            device=device,
        )
        dataset.extend(game_data)
        if on_game_end is not None:
            on_game_end(game_idx + 1, len(game_data), game_result, steps)
    return dataset
