from model.network import GomokuNet
from selfplay.generate import generate_selfplay_data


def test_generate_selfplay_data_not_empty() -> None:
    model = GomokuNet(board_size=15, channels=16, num_res_blocks=1)
    data = generate_selfplay_data(
        model=model,
        num_games=1,
        board_size=15,
        simulations=5,
        c_puct=1.5,
        device="cpu",
    )
    assert len(data) > 0
    state, policy, value = data[0]
    assert state.shape == (3, 15, 15)
    assert policy.shape == (225,)
    assert -1.0 <= value <= 1.0
