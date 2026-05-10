
from pathlib import Path

import torch

from model.network import GomokuNet
from selfplay.generate import (
    discover_iter_snapshots,
    generate_mixed_selfplay_vs_opponent_pool,
)

from train.utils import set_seed


def test_discover_iter_snapshots_sorts(tmp_path: Path) -> None:
    (tmp_path / "iter_00300_model.pt").write_bytes(b"x")
    (tmp_path / "iter_00005_model.pt").write_bytes(b"y")
    found = discover_iter_snapshots(tmp_path)
    assert [p.name for p in found] == ["iter_00005_model.pt", "iter_00300_model.pt"]


def test_mixed_pool_runs_with_best_only() -> None:
    set_seed(0)
    bs = 9
    latest = GomokuNet(board_size=bs, channels=8, num_res_blocks=1)
    best = GomokuNet(board_size=bs, channels=8, num_res_blocks=1)
    shell = GomokuNet(board_size=bs, channels=8, num_res_blocks=1)
    data = generate_mixed_selfplay_vs_opponent_pool(
        latest_model=latest,
        best_model=best,
        opponent_reload=shell,
        opponent_pool=["best"],
        num_games=1,
        board_size=bs,
        simulations=4,
        c_puct=1.5,
        device="cpu",
        mcts_infer_batch_size=1,
    )
    assert len(data) > 0


def test_mixed_pool_loads_snapshot_once(tmp_path: Path) -> None:
    set_seed(1)
    bs = 9
    snap = GomokuNet(board_size=bs, channels=8, num_res_blocks=2)
    p = tmp_path / "iter_00001_model.pt"
    torch.save(snap.state_dict(), p)
    latest = GomokuNet(board_size=bs, channels=8, num_res_blocks=2)
    best = GomokuNet(board_size=bs, channels=8, num_res_blocks=2)
    shell = GomokuNet(board_size=bs, channels=8, num_res_blocks=2)
    generate_mixed_selfplay_vs_opponent_pool(
        latest_model=latest,
        best_model=best,
        opponent_reload=shell,
        opponent_pool=["best", p],
        num_games=2,
        board_size=bs,
        simulations=3,
        c_puct=1.5,
        device="cpu",
        mcts_infer_batch_size=1,
    )

