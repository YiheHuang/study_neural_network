from env.board import Board, Player
from env.tactics import (
    immediate_winning_moves,
    opponent_winning_moves,
    tactical_priority_move,
)


def test_immediate_win_on_fifth_stone() -> None:
    b = Board(size=9)
    # Black: four in a row on row 4; noise white moves
    pairs = [
        (4, 0),
        (8, 8),
        (4, 1),
        (8, 7),
        (4, 2),
        (8, 6),
        (4, 3),
        (8, 5),
    ]
    for r, c in pairs:
        assert b.current_player == Player.BLACK or b.current_player == Player.WHITE
        b.place_stone(r, c)
    assert b.current_player == Player.BLACK
    assert immediate_winning_moves(b) == [(4, 4)]
    assert tactical_priority_move(b) == (4, 4)


def test_must_block_opponent_one_move_win() -> None:
    b = Board(size=9)
    pairs = [
        (5, 0),
        (0, 0),
        (6, 1),
        (0, 1),
        (7, 2),
        (0, 2),
        (8, 3),
        (0, 3),
    ]
    for r, c in pairs:
        b.place_stone(r, c)
    assert b.current_player == Player.BLACK
    assert immediate_winning_moves(b) == []
    assert set(opponent_winning_moves(b)) == {(0, 4)}
    assert tactical_priority_move(b) == (0, 4)


def test_win_priority_over_block() -> None:
    """Own five beats defending elsewhere."""
    b = Board(size=9)
    pairs = [
        (0, 0),
        (8, 0),
        (0, 1),
        (8, 1),
        (0, 2),
        (8, 2),
        (0, 3),
        (8, 3),
    ]
    for r, c in pairs:
        b.place_stone(r, c)
    assert b.current_player == Player.BLACK
    assert (0, 4) in immediate_winning_moves(b)
    assert (8, 4) in opponent_winning_moves(b)
    assert tactical_priority_move(b) == (0, 4)
