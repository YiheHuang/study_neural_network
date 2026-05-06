from __future__ import annotations

import random

from env.board import Board, GameResult, Player


def main() -> None:
    board = Board(size=15)
    step = 0
    print("Start random game: BLACK(X) first, WHITE(O) second.")
    while board.game_result() == GameResult.ONGOING:
        legal = board.legal_moves()
        row, col = random.choice(legal)
        current = board.current_player
        board.place_stone(row, col)
        step += 1
        print(f"Step {step:03d}: {current.name} -> ({row}, {col})")

    print("\nFinal board:")
    print(board.to_pretty_string())
    result = board.game_result()
    if result == GameResult.BLACK_WIN:
        print("\nResult: BLACK wins.")
    elif result == GameResult.WHITE_WIN:
        print("\nResult: WHITE wins.")
    else:
        print("\nResult: DRAW.")


if __name__ == "__main__":
    main()
