from __future__ import annotations

import argparse
import json
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import torch
from torch import optim

from eval import evaluate_models
from env.board import GameResult
from model.network import GomokuNet
from selfplay import generate_selfplay_data
from train.utils import (
    Sample,
    load_config,
    load_model_weights,
    prepare_tensors,
    set_seed,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous self-play training with best-model promotion."
    )
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=30)
    parser.add_argument("--eval-simulations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latest-path", type=str, default="checkpoints/latest_model.pt")
    parser.add_argument("--best-path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--log-path", type=str, default="logs/train_log.jsonl")
    parser.add_argument("--game-log-path", type=str, default="logs/train_game_log.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _append_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _result_name(result: GameResult) -> str:
    if result == GameResult.BLACK_WIN:
        return "BLACK_WIN"
    if result == GameResult.WHITE_WIN:
        return "WHITE_WIN"
    if result == GameResult.DRAW:
        return "DRAW"
    return "ONGOING"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    board_size = int(cfg["board_size"])
    replay_buffer_size = int(cfg["replay_buffer_size"])
    c_puct = float(cfg["c_puct"])
    eval_games = int(cfg["eval_games"])
    promote_threshold = float(cfg["promote_threshold"])
    device = torch.device(args.device)

    latest_path = Path(args.latest_path)
    best_path = Path(args.best_path)
    log_path = Path(args.log_path)
    game_log_path = Path(args.game_log_path)

    latest_model = GomokuNet(
        board_size=board_size,
        channels=args.channels,
        num_res_blocks=args.res_blocks,
    ).to(device)
    best_model = GomokuNet(
        board_size=board_size,
        channels=args.channels,
        num_res_blocks=args.res_blocks,
    ).to(device)

    loaded_latest = load_model_weights(latest_model, latest_path)
    loaded_best = load_model_weights(best_model, best_path)

    if not loaded_latest and loaded_best:
        best_state = deepcopy(best_model.state_dict())
        latest_model.load_state_dict(best_state)
        loaded_latest = True
    if not loaded_best:
        best_model.load_state_dict(deepcopy(latest_model.state_dict()))
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model.state_dict(), best_path)

    optimizer = optim.Adam(latest_model.parameters(), lr=args.lr)
    replay = deque(maxlen=replay_buffer_size)

    print("Continuous training start")
    print(
        f"from latest={latest_path} exists={loaded_latest}, "
        f"best={best_path} exists={best_path.exists()}"
    )

    interrupted = False
    try:
        for it in range(1, args.max_iters + 1):
            iter_start = time.time()
            print(f"\n[Iter {it}] self-play generating...")
            iter_game_start = time.time()

            def on_game_end(
                game_idx: int, sample_count: int, result: GameResult, steps: int
            ) -> None:
                elapsed = time.time() - iter_game_start
                payload = {
                    "iteration": it,
                    "game_in_iteration": game_idx,
                    "result": _result_name(result),
                    "steps": steps,
                    "samples": sample_count,
                    "elapsed_sec": round(elapsed, 2),
                }
                _append_log(game_log_path, payload)
                print(
                    f"[Iter {it}][Game {game_idx}/{args.games_per_iter}] "
                    f"result={payload['result']} steps={steps} samples={sample_count} "
                    f"elapsed={elapsed:.1f}s"
                )

            new_samples = generate_selfplay_data(
                model=latest_model,
                num_games=args.games_per_iter,
                board_size=board_size,
                simulations=args.simulations,
                c_puct=c_puct,
                device=device,
                on_game_end=on_game_end,
            )
            replay.extend(new_samples)
            print(f"[Iter {it}] samples +{len(new_samples)}; replay={len(replay)}")

            states, policies, values = prepare_tensors(list(replay), device=device)
            loss = 0.0
            for epoch in range(1, args.epochs + 1):
                loss = train_one_epoch(
                    model=latest_model,
                    optimizer=optimizer,
                    states=states,
                    policy_targets=policies,
                    value_targets=values,
                    batch_size=args.batch_size,
                )
                print(f"[Iter {it}] epoch {epoch}/{args.epochs} loss={loss:.4f}")

            latest_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(latest_model.state_dict(), latest_path)

            print(f"[Iter {it}] arena evaluating against best...")
            arena = evaluate_models(
                candidate_model=latest_model,
                best_model=best_model,
                board_size=board_size,
                games=eval_games,
                simulations=args.eval_simulations,
                c_puct=c_puct,
                device=device,
            )
            promoted = arena.candidate_win_rate >= promote_threshold
            if promoted:
                best_model.load_state_dict(deepcopy(latest_model.state_dict()))
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model.state_dict(), best_path)

            elapsed = time.time() - iter_start
            log_item = {
                "iteration": it,
                "new_samples": len(new_samples),
                "replay_size": len(replay),
                "loss": round(loss, 6),
                "arena": arena.to_dict(),
                "promoted": promoted,
                "promote_threshold": promote_threshold,
                "elapsed_sec": round(elapsed, 2),
                "latest_model_path": str(latest_path),
                "best_model_path": str(best_path),
            }
            _append_log(log_path, log_item)
            print(
                f"[Iter {it}] arena winrate={arena.candidate_win_rate:.3f}, "
                f"promoted={promoted}, elapsed={elapsed:.1f}s"
            )
            print(f"[Iter {it}] logged to {log_path}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received, saving latest model before exit...")
    finally:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latest_model.state_dict(), latest_path)
        if interrupted:
            print(f"Latest model checkpoint saved to: {latest_path}")
            print("Interrupted gracefully.")

    print("\nContinuous training finished.")


if __name__ == "__main__":
    main()
