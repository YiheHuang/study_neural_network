from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from env.board import Board, GameResult, Player
from mcts import MCTS


@dataclass
class ArenaResult:
    candidate_wins: int
    best_wins: int
    draws: int
    games: int

    @property
    def candidate_win_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.candidate_wins / self.games

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "candidate_wins": self.candidate_wins,
            "best_wins": self.best_wins,
            "draws": self.draws,
            "games": self.games,
            "candidate_win_rate": self.candidate_win_rate,
        }


def _play_one_game(
    candidate_model: torch.nn.Module,
    best_model: torch.nn.Module,
    board_size: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device,
    candidate_as_black: bool,
) -> GameResult:
    board = Board(size=board_size)
    candidate_player = Player.BLACK if candidate_as_black else Player.WHITE
    candidate_search = MCTS(model=candidate_model, c_puct=c_puct)
    best_search = MCTS(model=best_model, c_puct=c_puct)

    while board.game_result() == GameResult.ONGOING:
        if board.current_player == candidate_player:
            move, _ = candidate_search.run(board, simulations=simulations, device=device)
        else:
            move, _ = best_search.run(board, simulations=simulations, device=device)
        board.place_stone(*move)
    return board.game_result()


def evaluate_models(
    candidate_model: torch.nn.Module,
    best_model: torch.nn.Module,
    board_size: int,
    games: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device = "cpu",
) -> ArenaResult:
    result = ArenaResult(candidate_wins=0, best_wins=0, draws=0, games=games)
    for i in range(games):
        candidate_as_black = (i % 2) == 0
        game_result = _play_one_game(
            candidate_model=candidate_model,
            best_model=best_model,
            board_size=board_size,
            simulations=simulations,
            c_puct=c_puct,
            device=device,
            candidate_as_black=candidate_as_black,
        )
        if game_result == GameResult.DRAW:
            result.draws += 1
            continue
        candidate_win = (
            (candidate_as_black and game_result == GameResult.BLACK_WIN)
            or ((not candidate_as_black) and game_result == GameResult.WHITE_WIN)
        )
        if candidate_win:
            result.candidate_wins += 1
        else:
            result.best_wins += 1
    return result
