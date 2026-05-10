from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


class Player(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    def opponent(self) -> "Player":
        if self == Player.BLACK:
            return Player.WHITE
        if self == Player.WHITE:
            return Player.BLACK
        return Player.EMPTY


class GameResult(IntEnum):
    ONGOING = 0
    BLACK_WIN = 1
    WHITE_WIN = -1
    DRAW = 2


@dataclass
class Move:
    row: int
    col: int
    player: Player


class Board:
    def __init__(self, size: int = 15) -> None:
        if size < 5:
            raise ValueError("Board size must be at least 5.")
        self.size = size
        self.grid: np.ndarray = np.zeros((size, size), dtype=np.int8)
        self.current_player: Player = Player.BLACK
        self.winner: Optional[Player] = None
        self.last_move: Optional[Move] = None
        self.move_count = 0

    def copy(self) -> "Board":
        new_board = Board(self.size)
        new_board.grid = self.grid.copy()
        new_board.current_player = self.current_player
        new_board.winner = self.winner
        new_board.last_move = self.last_move
        new_board.move_count = self.move_count
        return new_board

    def reset(self) -> None:
        self.grid.fill(Player.EMPTY)
        self.current_player = Player.BLACK
        self.winner = None
        self.last_move = None
        self.move_count = 0

    def is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_legal_move(self, row: int, col: int) -> bool:
        if not self.is_on_board(row, col):
            return False
        if self.winner is not None:
            return False
        return self.grid[row, col] == Player.EMPTY

    def legal_moves(self) -> List[Tuple[int, int]]:
        if self.winner is not None:
            return []
        rows, cols = np.where(self.grid == Player.EMPTY)
        return list(zip(rows.tolist(), cols.tolist()))

    def place_stone(self, row: int, col: int) -> None:
        if not self.is_legal_move(row, col):
            raise ValueError(f"Illegal move: ({row}, {col})")

        player = self.current_player
        self.grid[row, col] = int(player)
        self.last_move = Move(row=row, col=col, player=player)
        self.move_count += 1

        if self._check_five(row, col, player):
            self.winner = player
        else:
            self.current_player = player.opponent()

    def game_result(self) -> GameResult:
        if self.winner == Player.BLACK:
            return GameResult.BLACK_WIN
        if self.winner == Player.WHITE:
            return GameResult.WHITE_WIN
        if self.move_count == self.size * self.size:
            return GameResult.DRAW
        return GameResult.ONGOING

    def _count_in_direction(
        self, row: int, col: int, dr: int, dc: int, player: Player
    ) -> int:
        count = 0
        r, c = row + dr, col + dc
        while self.is_on_board(r, c) and self.grid[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def _check_five(self, row: int, col: int, player: Player) -> bool:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            total = 1
            total += self._count_in_direction(row, col, dr, dc, player)
            total += self._count_in_direction(row, col, -dr, -dc, player)
            if total >= 5:
                return True
        return False

    def to_pretty_string(self) -> str:
        symbols = {
            Player.EMPTY: ".",
            Player.BLACK: "X",
            Player.WHITE: "O",
        }
        lines = []
        header = "   " + " ".join(f"{c:02d}" for c in range(self.size))
        lines.append(header)
        for row in range(self.size):
            line = [f"{row:02d}"]
            for col in range(self.size):
                line.append(symbols[Player(int(self.grid[row, col]))])
            lines.append(" ".join(line))
        return "\n".join(lines)
