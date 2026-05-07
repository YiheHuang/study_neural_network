from env.board import Board
from mcts import MCTS
from model.network import GomokuNet


def test_mcts_returns_legal_move_and_policy_shape() -> None:
    board = Board(size=15)
    board.place_stone(7, 7)
    board.place_stone(7, 8)

    model = GomokuNet(board_size=15, channels=32, num_res_blocks=1)
    mcts = MCTS(model=model, c_puct=1.5)
    move, policy = mcts.run(board, simulations=30)

    assert board.is_legal_move(*move)
    assert policy.shape == (225,)
    assert abs(float(policy.sum()) - 1.0) < 1e-6


def test_mcts_batched_inference_valid_policy() -> None:
    board = Board(size=9)
    model = GomokuNet(board_size=9, channels=16, num_res_blocks=2)
    mcts = MCTS(model=model, c_puct=1.5)
    move, policy = mcts.run(
        board,
        simulations=24,
        device="cpu",
        infer_batch_size=8,
        virtual_loss_weight=1.0,
    )
    assert board.is_legal_move(*move)
    assert policy.shape == (81,)
    assert abs(float(policy.sum()) - 1.0) < 1e-5
