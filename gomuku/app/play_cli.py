from __future__ import annotations

import argparse
from typing import Tuple

import torch

from env.board import Board, GameResult, Player
from mcts import MCTS
from model.network import GomokuNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Gomoku against NN+MCTS AI.")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--human", choices=["black", "white"], default="black")
    parser.add_argument("--simulations", type=int, default=120)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=6)
    return parser.parse_args()


def _player_from_choice(choice: str) -> Player:
    return Player.BLACK if choice == "black" else Player.WHITE


def _load_model(args: argparse.Namespace) -> GomokuNet:
    model = GomokuNet(
        board_size=args.board_size,
        channels=args.channels,
        num_res_blocks=args.res_blocks,
    )
    if args.model_path:
        state = torch.load(args.model_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"Loaded model from: {args.model_path}")
    else:
        print("Using randomly initialized model parameters.")
    return model


def _parse_human_move(text: str, board_size: int) -> Tuple[int, int]:
    parts = text.strip().replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("Please input exactly 2 numbers: row col")
    row, col = int(parts[0]), int(parts[1])
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Move out of range. Valid range is [0, {board_size - 1}]")
    return row, col


def main() -> None:
    args = parse_args()
    board = Board(size=args.board_size)
    human_player = _player_from_choice(args.human)
    ai_player = human_player.opponent()

    model = _load_model(args)
    mcts = MCTS(model=model, c_puct=1.5)

    print("Game start. Coordinates are zero-based.")
    print(f"Human: {human_player.name}, AI: {ai_player.name}\n")

    while board.game_result() == GameResult.ONGOING:
        print(board.to_pretty_string())
        current = board.current_player

        if current == human_player:
            while True:
                try:
                    raw = input("\nYour move (row col): ")
                    row, col = _parse_human_move(raw, board.size)
                    board.place_stone(row, col)
                    break
                except ValueError as exc:
                    print(f"Invalid move: {exc}")
        else:
            print(f"\nAI is thinking ({args.simulations} simulations)...")
            move, _ = mcts.run(board, simulations=args.simulations, device=args.device)
            board.place_stone(*move)
            print(f"AI move: {move[0]} {move[1]}")

    print("\nFinal board:")
    print(board.to_pretty_string())
    result = board.game_result()
    if result == GameResult.DRAW:
        print("\nResult: DRAW.")
    elif result == GameResult.BLACK_WIN:
        print("\nResult: BLACK wins.")
    else:
        print("\nResult: WHITE wins.")


if __name__ == "__main__":
    main()
