from eval import evaluate_models
from model.network import GomokuNet


def test_evaluate_models_returns_valid_stats() -> None:
    candidate = GomokuNet(board_size=15, channels=16, num_res_blocks=1)
    best = GomokuNet(board_size=15, channels=16, num_res_blocks=1)
    result = evaluate_models(
        candidate_model=candidate,
        best_model=best,
        board_size=15,
        games=2,
        simulations=3,
        c_puct=1.5,
        device="cpu",
    )
    stats = result.to_dict()
    assert stats["games"] == 2
    assert stats["candidate_wins"] + stats["best_wins"] + stats["draws"] == 2
    assert 0.0 <= stats["candidate_win_rate"] <= 1.0
