import numpy as np
import torch

from env.board import Board, Player
from model.network import GomokuNet
from model.predict import board_to_tensor, legal_moves_mask, predict_policy_value


def test_board_to_tensor_shape_and_planes() -> None:
    board = Board(size=15)
    board.place_stone(7, 7)  # Black
    board.place_stone(7, 8)  # White, now black to move

    x = board_to_tensor(board)
    assert x.shape == (3, 15, 15)
    assert x.dtype == torch.float32

    # Current player is BLACK.
    assert x[0, 7, 7].item() == 1.0
    assert x[1, 7, 8].item() == 1.0
    assert torch.all(x[2] == 1.0)


def test_legal_moves_mask() -> None:
    board = Board(size=15)
    board.place_stone(0, 0)
    mask = legal_moves_mask(board)
    assert mask.shape == (225,)
    assert mask.dtype == np.float32
    assert mask[0] == 0.0
    assert int(mask.sum()) == 224


def test_gomokunet_forward_shapes_and_value_range() -> None:
    model = GomokuNet(board_size=15, channels=32, num_res_blocks=2)
    x = torch.randn(4, 3, 15, 15)
    policy_logits, value = model(x)
    assert policy_logits.shape == (4, 225)
    assert value.shape == (4, 1)
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_predict_policy_value_masks_illegal_and_normalizes() -> None:
    model = GomokuNet(board_size=15, channels=32, num_res_blocks=1)
    board = Board(size=15)
    board.place_stone(0, 0)
    board.place_stone(0, 1)

    policy, value = predict_policy_value(model, board)
    assert policy.shape == (225,)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0

    illegal_indices = [0, 1]
    for idx in illegal_indices:
        assert policy[idx] == 0.0

    assert np.isclose(policy.sum(), 1.0, atol=1e-6)
