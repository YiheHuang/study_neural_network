import pytest

from env.board import Board, GameResult, Player


def test_horizontal_five_black_win() -> None:
    board = Board(size=15)
    # Black makes five in row 0.
    for col in range(5):
        board.place_stone(0, col)  # Black
        if col < 4:
            board.place_stone(1, col)  # White filler
    assert board.game_result() == GameResult.BLACK_WIN


def test_vertical_five_black_win() -> None:
    board = Board(size=15)
    for row in range(5):
        board.place_stone(row, 0)  # Black
        if row < 4:
            board.place_stone(row, 1)  # White filler
    assert board.game_result() == GameResult.BLACK_WIN


def test_main_diagonal_five_black_win() -> None:
    board = Board(size=15)
    for i in range(5):
        board.place_stone(i, i)  # Black
        if i < 4:
            board.place_stone(i, i + 1)  # White filler
    assert board.game_result() == GameResult.BLACK_WIN


def test_anti_diagonal_five_black_win() -> None:
    board = Board(size=15)
    positions = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
    fillers = [(0, 5), (1, 4), (2, 3), (3, 2)]
    for i, pos in enumerate(positions):
        board.place_stone(*pos)  # Black
        if i < 4:
            board.place_stone(*fillers[i])  # White filler
    assert board.game_result() == GameResult.BLACK_WIN


def test_boundary_horizontal_win() -> None:
    board = Board(size=15)
    # Place at last row, right edge sequence.
    black_moves = [(14, 10), (14, 11), (14, 12), (14, 13), (14, 14)]
    white_moves = [(13, 0), (13, 1), (13, 2), (13, 3)]
    for idx, move in enumerate(black_moves):
        board.place_stone(*move)
        if idx < 4:
            board.place_stone(*white_moves[idx])
    assert board.game_result() == GameResult.BLACK_WIN


def test_illegal_move_same_position_raises() -> None:
    board = Board(size=15)
    board.place_stone(7, 7)
    with pytest.raises(ValueError):
        board.place_stone(7, 7)


def test_illegal_move_out_of_board_raises() -> None:
    board = Board(size=15)
    with pytest.raises(ValueError):
        board.place_stone(-1, 0)


def test_copy_is_deep_for_grid() -> None:
    board = Board(size=15)
    board.place_stone(0, 0)
    copied = board.copy()
    copied.place_stone(0, 1)
    assert board.grid[0, 1] == Player.EMPTY
    assert copied.grid[0, 1] != Player.EMPTY


def test_draw_detection_on_small_board() -> None:
    board = Board(size=5)
    # Prebuilt full board with no five-in-a-row for either side.
    pattern = [
        [1, 1, -1, -1, 1],
        [-1, -1, 1, 1, -1],
        [1, 1, -1, -1, 1],
        [-1, -1, 1, 1, -1],
        [1, 1, -1, -1, 1],
    ]
    board.grid[:, :] = pattern
    board.current_player = Player.BLACK
    board.move_count = 25
    board.winner = None
    assert board.game_result() == GameResult.DRAW
