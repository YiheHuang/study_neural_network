from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from env.board import Board, GameResult, Player
from mcts import MCTS
from model.network import GomokuNet
from model.predict import board_to_tensor


@dataclass
class Sample:
    state: np.ndarray
    policy: np.ndarray
    player: int


_WORKER_MODEL: torch.nn.Module | None = None
_WORKER_DEVICE: torch.device | None = None


def _result_to_winner(result: GameResult) -> int:
    if result == GameResult.BLACK_WIN:
        return int(Player.BLACK)
    if result == GameResult.WHITE_WIN:
        return int(Player.WHITE)
    return int(Player.EMPTY)


def _apply_random_opening(board: Board, num_moves: int) -> None:
    """Uniformly random legal moves at game start; does not produce training rows."""
    if num_moves <= 0:
        return
    for _ in range(num_moves):
        if board.game_result() != GameResult.ONGOING:
            return
        legal = board.legal_moves()
        if not legal:
            return
        choice = int(np.random.randint(len(legal)))
        row, col = legal[choice]
        board.place_stone(row, col)


def _random_opening_num_moves(max_moves: int) -> int:
    """
    Uniformly sample n ∈ {0,…,max_moves} inclusive. Then run n alternating random
    legal moves (Black first): Black plays ⌊(n+1)/2⌋ times, White ⌊n/2⌋ times.
    If max_moves <= 0 return 0.
    """
    if max_moves <= 0:
        return 0
    return int(np.random.randint(max_moves + 1))


def play_self_game(
    model: torch.nn.Module,
    board_size: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device = "cpu",
    temperature: float = 1.0,
    temperature_drop_move: int = 20,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    mcts_infer_batch_size: int = 1,
    mcts_virtual_loss_weight: float = 1.0,
    opening_random_moves: int = 0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], GameResult, int]:
    """
    opening_random_moves: k; each episode samples n ~ Uniform({0,…,k}); n alternating random moves
    (Black first ⇒ Black (n+1)//2, White n//2). Not in replay.
    """
    board = Board(size=board_size)
    _apply_random_opening(board, _random_opening_num_moves(opening_random_moves))
    mcts = MCTS(model=model, c_puct=c_puct)
    history: List[Sample] = []

    while board.game_result() == GameResult.ONGOING:
        state = board_to_tensor(board).cpu().numpy()
        move_number = len(history)
        move_temp = temperature if move_number < temperature_drop_move else 0.0
        move, policy = mcts.run(
            board,
            simulations=simulations,
            device=device,
            temperature=move_temp,
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            infer_batch_size=mcts_infer_batch_size,
            virtual_loss_weight=mcts_virtual_loss_weight,
        )
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


def _cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _init_worker(
    board_size: int,
    channels: int,
    num_res_blocks: int,
    model_state: Dict[str, torch.Tensor],
    device: str,
) -> None:
    global _WORKER_MODEL, _WORKER_DEVICE
    _WORKER_DEVICE = torch.device(device)
    _WORKER_MODEL = GomokuNet(
        board_size=board_size,
        channels=channels,
        num_res_blocks=num_res_blocks,
    )
    _WORKER_MODEL.load_state_dict(model_state)
    _WORKER_MODEL.to(_WORKER_DEVICE)
    _WORKER_MODEL.eval()


def _play_self_game_worker(
    game_id: int,
    board_size: int,
    simulations: int,
    c_puct: float,
    temperature: float,
    temperature_drop_move: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    mcts_infer_batch_size: int,
    mcts_virtual_loss_weight: float,
    opening_random_moves: int,
) -> Dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_DEVICE is None:
        raise RuntimeError("Worker model not initialized.")
    game_data, game_result, steps = play_self_game(
        model=_WORKER_MODEL,
        board_size=board_size,
        simulations=simulations,
        c_puct=c_puct,
        device=_WORKER_DEVICE,
        temperature=temperature,
        temperature_drop_move=temperature_drop_move,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        mcts_infer_batch_size=mcts_infer_batch_size,
        mcts_virtual_loss_weight=mcts_virtual_loss_weight,
        opening_random_moves=opening_random_moves,
    )
    return {
        "game_id": game_id,
        "game_data": game_data,
        "game_result": game_result,
        "steps": steps,
    }


def generate_selfplay_data(
    model: torch.nn.Module,
    num_games: int,
    board_size: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device = "cpu",
    on_game_end: Callable[[int, int, GameResult, int], None] | None = None,
    num_workers: int = 1,
    worker_device: str | torch.device | None = None,
    temperature: float = 1.0,
    temperature_drop_move: int = 20,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    mcts_infer_batch_size: int = 1,
    mcts_virtual_loss_weight: float = 1.0,
    opening_random_moves: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    dataset: List[Tuple[np.ndarray, np.ndarray, float]] = []
    if num_workers <= 1:
        for game_idx in range(num_games):
            game_data, game_result, steps = play_self_game(
                model=model,
                board_size=board_size,
                simulations=simulations,
                c_puct=c_puct,
                device=device,
                temperature=temperature,
                temperature_drop_move=temperature_drop_move,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                mcts_infer_batch_size=mcts_infer_batch_size,
                mcts_virtual_loss_weight=mcts_virtual_loss_weight,
                opening_random_moves=opening_random_moves,
            )
            dataset.extend(game_data)
            if on_game_end is not None:
                on_game_end(game_idx + 1, len(game_data), game_result, steps)
        return dataset

    if not isinstance(model, GomokuNet):
        raise TypeError("Parallel self-play currently supports GomokuNet only.")

    state = _cpu_state_dict(model)
    actual_worker_device = str(worker_device or device)
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(
            model.board_size,
            model.stem[0].out_channels,
            len(model.backbone),
            state,
            actual_worker_device,
        ),
    ) as executor:
        future_to_game_id = {
            executor.submit(
                _play_self_game_worker,
                game_id + 1,
                board_size,
                simulations,
                c_puct,
                temperature,
                temperature_drop_move,
                dirichlet_alpha,
                dirichlet_epsilon,
                mcts_infer_batch_size,
                mcts_virtual_loss_weight,
                opening_random_moves,
            ): game_id
            for game_id in range(num_games)
        }
        finished = 0
        for future in as_completed(future_to_game_id):
            result = future.result()
            game_data = result["game_data"]
            game_result = result["game_result"]
            steps = result["steps"]
            finished += 1
            dataset.extend(game_data)
            if on_game_end is not None:
                on_game_end(finished, len(game_data), game_result, steps)

    return dataset


def play_match_game(
    black_model: torch.nn.Module,
    white_model: torch.nn.Module,
    board_size: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device = "cpu",
    temperature: float = 1.0,
    temperature_drop_move: int = 20,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    mcts_infer_batch_size: int = 1,
    mcts_virtual_loss_weight: float = 1.0,
    opening_random_moves: int = 0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], GameResult, int]:
    board = Board(size=board_size)
    _apply_random_opening(board, _random_opening_num_moves(opening_random_moves))
    black_search = MCTS(model=black_model, c_puct=c_puct)
    white_search = MCTS(model=white_model, c_puct=c_puct)
    history: List[Sample] = []

    while board.game_result() == GameResult.ONGOING:
        state = board_to_tensor(board).cpu().numpy()
        move_number = len(history)
        move_temp = temperature if move_number < temperature_drop_move else 0.0
        search = black_search if board.current_player == Player.BLACK else white_search
        move, policy = search.run(
            board,
            simulations=simulations,
            device=device,
            temperature=move_temp,
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            infer_batch_size=mcts_infer_batch_size,
            virtual_loss_weight=mcts_virtual_loss_weight,
        )
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


def generate_mixed_selfplay_data(
    latest_model: torch.nn.Module,
    best_model: torch.nn.Module,
    num_games: int,
    board_size: int,
    simulations: int,
    c_puct: float,
    device: str | torch.device = "cpu",
    on_game_end: Callable[[int, int, GameResult, int], None] | None = None,
    temperature: float = 1.0,
    temperature_drop_move: int = 20,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    mcts_infer_batch_size: int = 1,
    mcts_virtual_loss_weight: float = 1.0,
    opening_random_moves: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    dataset: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for game_idx in range(num_games):
        latest_as_black = (game_idx % 2) == 0
        black_model = latest_model if latest_as_black else best_model
        white_model = best_model if latest_as_black else latest_model
        game_data, game_result, steps = play_match_game(
            black_model=black_model,
            white_model=white_model,
            board_size=board_size,
            simulations=simulations,
            c_puct=c_puct,
            device=device,
            temperature=temperature,
            temperature_drop_move=temperature_drop_move,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            mcts_infer_batch_size=mcts_infer_batch_size,
            mcts_virtual_loss_weight=mcts_virtual_loss_weight,
            opening_random_moves=opening_random_moves,
        )
        dataset.extend(game_data)
        if on_game_end is not None:
            on_game_end(game_idx + 1, len(game_data), game_result, steps)
    return dataset
