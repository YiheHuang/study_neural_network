import numpy as np

from model.network import GomokuNet
from selfplay.generate import _random_opening_num_moves, play_self_game
from train.utils import set_seed


def test_random_opening_num_moves_zero_when_k_nonpos() -> None:
    assert _random_opening_num_moves(-1) == 0
    assert _random_opening_num_moves(0) == 0


def test_random_opening_num_moves_inclusive_range_k() -> None:
    set_seed(202)
    k = 6
    drawn = [_random_opening_num_moves(k) for _ in range(3000)]
    assert min(drawn) == 0 and max(drawn) == k


def test_opening_zero_no_stones_before_first_mcts() -> None:
    set_seed(123)
    model = GomokuNet(board_size=9, channels=8, num_res_blocks=1)
    data, _result, steps = play_self_game(
        model=model,
        board_size=9,
        simulations=4,
        c_puct=1.5,
        device="cpu",
        opening_random_moves=0,
        mcts_infer_batch_size=1,
    )
    assert steps > 0
    stones = np.count_nonzero(data[0][0][0]) + np.count_nonzero(data[0][0][1])
    assert stones == 0


def test_random_open_then_mcts_occasionally_empty_board_start() -> None:
    """With k>0, sometimes n==0 ⇒ first observed state matches empty-opening case."""
    set_seed(777)
    model = GomokuNet(board_size=9, channels=8, num_res_blocks=1)
    saw_empty_first = False
    for trial in range(80):
        data, _, steps = play_self_game(
            model=model,
            board_size=9,
            simulations=4,
            c_puct=1.5,
            device="cpu",
            opening_random_moves=12,
            mcts_infer_batch_size=1,
        )
        assert steps > 0
        s = np.count_nonzero(data[0][0][0]) + np.count_nonzero(data[0][0][1])
        if s == 0:
            saw_empty_first = True
            break
    assert saw_empty_first
