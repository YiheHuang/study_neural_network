from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from env.board import Board, GameResult, Player
from mcts import MCTS
from model.network import GomokuNet


_ARENA_CAND: torch.nn.Module | None = None
_ARENA_BEST: torch.nn.Module | None = None
_ARENA_DEVICE: torch.device | None = None


def _cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _arena_init_worker(
    board_size: int,
    channels: int,
    num_res_blocks: int,
    candidate_state: Dict[str, torch.Tensor],
    best_state: Dict[str, torch.Tensor],
    device: str,
) -> None:
    global _ARENA_CAND, _ARENA_BEST, _ARENA_DEVICE
    _ARENA_DEVICE = torch.device(device)
    _ARENA_CAND = GomokuNet(
        board_size=board_size,
        channels=channels,
        num_res_blocks=num_res_blocks,
    )
    _ARENA_CAND.load_state_dict(candidate_state)
    _ARENA_CAND.to(_ARENA_DEVICE).eval()
    _ARENA_BEST = GomokuNet(
        board_size=board_size,
        channels=channels,
        num_res_blocks=num_res_blocks,
    )
    _ARENA_BEST.load_state_dict(best_state)
    _ARENA_BEST.to(_ARENA_DEVICE).eval()


def _arena_worker_one_game(
    game_index: int,
    board_size: int,
    simulations: int,
    c_puct: float,
) -> Tuple[int, int]:
    """Returns (int(GameResult), 1 if candidate is black else 0)."""
    if _ARENA_CAND is None or _ARENA_BEST is None or _ARENA_DEVICE is None:
        raise RuntimeError("Arena worker models not initialized.")
    candidate_as_black = (game_index % 2) == 0
    res = _play_one_game(
        candidate_model=_ARENA_CAND,
        best_model=_ARENA_BEST,
        board_size=board_size,
        simulations=simulations,
        c_puct=c_puct,
        device=_ARENA_DEVICE,
        candidate_as_black=candidate_as_black,
    )
    return int(res), 1 if candidate_as_black else 0


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
            move, _ = candidate_search.run(
                board, simulations=simulations, device=device,
                add_dirichlet_noise=True, dirichlet_alpha=0.15, dirichlet_epsilon=0.25,
            )
        else:
            move, _ = best_search.run(
                board, simulations=simulations, device=device,
                add_dirichlet_noise=True, dirichlet_alpha=0.15, dirichlet_epsilon=0.25,
            )
        board.place_stone(*move)
    return board.game_result()


def _accumulate_game_result(
    result: ArenaResult,
    game_result: GameResult,
    candidate_as_black: bool,
) -> None:
    if game_result == GameResult.DRAW:
        result.draws += 1
        return
    candidate_win = (
        (candidate_as_black and game_result == GameResult.BLACK_WIN)
        or ((not candidate_as_black) and game_result == GameResult.WHITE_WIN)
    )
    if candidate_win:
        result.candidate_wins += 1
    else:
        result.best_wins += 1


def evaluate_models(
    candidate_model: torch.nn.Module,
    best_model: torch.nn.Module,
    board_size: int,
    games: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device = "cpu",
    num_workers: int = 1,
    worker_device: str | torch.device | None = None,
) -> ArenaResult:
    result = ArenaResult(candidate_wins=0, best_wins=0, draws=0, games=games)
    if games <= 0:
        return result

    if num_workers <= 1:
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
            _accumulate_game_result(result, game_result, candidate_as_black)
        return result

    if not isinstance(candidate_model, GomokuNet) or not isinstance(
        best_model, GomokuNet
    ):
        raise TypeError("Parallel arena requires GomokuNet for both models.")

    actual_worker_device = str(worker_device or "cpu")
    cand_state = _cpu_state_dict(candidate_model)
    best_state = _cpu_state_dict(best_model)
    channels = int(candidate_model.stem[0].out_channels)
    num_res_blocks = len(candidate_model.backbone)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_arena_init_worker,
        initargs=(
            board_size,
            channels,
            num_res_blocks,
            cand_state,
            best_state,
            actual_worker_device,
        ),
    ) as executor:
        future_to_index = {
            executor.submit(
                _arena_worker_one_game, i, board_size, simulations, c_puct
            ): i
            for i in range(games)
        }
        for future in as_completed(future_to_index):
            res_int, cand_black = future.result()
            game_result = GameResult(res_int)
            candidate_as_black = cand_black == 1
            _accumulate_game_result(result, game_result, candidate_as_black)

    return result
