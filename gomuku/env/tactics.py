"""One-ply tactical rules for freestyle gomoku (first to five, no bans)."""

from __future__ import annotations

from typing import List, Tuple

from env.board import Board, Player


def _would_player_win_here(board: Board, row: int, col: int, player: Player) -> bool:
    if board.grid[row, col] != int(Player.EMPTY):
        return False
    b = board.copy()
    b.current_player = player
    b.place_stone(row, col)
    return b.winner == player


def immediate_winning_moves(board: Board) -> List[Tuple[int, int]]:
    """Legal cells where the current player wins in one move."""
    me = board.current_player
    return [
        (r, c)
        for r, c in board.legal_moves()
        if _would_player_win_here(board, r, c, me)
    ]


def opponent_winning_moves(board: Board) -> List[Tuple[int, int]]:
    """Legal cells where the opponent would win if they played there (must-block set)."""
    opp = board.current_player.opponent()
    return [
        (r, c)
        for r, c in board.legal_moves()
        if _would_player_win_here(board, r, c, opp)
    ]


def tactical_priority_move(board: Board) -> Tuple[int, int] | None:
    """
    Deterministic forced move: any immediate win for side to move, else any must-block.

    Tie-break among multiple winning (or blocking) cells: lexicographically smallest (row, col).
    """
    wins = immediate_winning_moves(board)
    if wins:
        return min(wins)
    blocks = opponent_winning_moves(board)
    if blocks:
        return min(blocks)
    return None
