from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import optim

from model.network import GomokuNet
from selfplay import generate_selfplay_data
from train.utils import Sample, load_config, prepare_tensors, set_seed, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gomoku model with self-play data.")
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iter", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="checkpoints/latest_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    board_size = int(cfg["board_size"])
    c_puct = float(cfg["c_puct"])
    simulations = int(args.simulations or cfg["mcts_simulations"])
    device = torch.device(args.device)

    model = GomokuNet(
        board_size=board_size,
        channels=args.channels,
        num_res_blocks=args.res_blocks,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Training start")
    print(
        f"settings: iterations={args.iterations}, games/iter={args.games_per_iter}, "
        f"simulations={simulations}, epochs={args.epochs}, device={device}"
    )

    all_samples: List[Sample] = []
    for it in range(1, args.iterations + 1):
        print(f"\n[Iteration {it}] generating self-play data...")
        new_data = generate_selfplay_data(
            model=model,
            num_games=args.games_per_iter,
            board_size=board_size,
            simulations=simulations,
            c_puct=c_puct,
            device=device,
        )
        all_samples.extend(new_data)
        print(f"collected samples: +{len(new_data)} (total={len(all_samples)})")

        states, policies, values = prepare_tensors(all_samples, device=device)
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(
                model=model,
                optimizer=optimizer,
                states=states,
                policy_targets=policies,
                value_targets=values,
                batch_size=args.batch_size,
            )
            print(f"[Iteration {it}] epoch {epoch}/{args.epochs} loss={loss:.4f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to: {save_path}")


if __name__ == "__main__":
    main()
