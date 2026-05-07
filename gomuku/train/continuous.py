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
    build_decay_weights,
    Sample,
    device_banner,
    load_config,
    load_model_weights,
    load_replay_buffer,
    prepare_tensors,
    resolve_device,
    save_replay_buffer,
    set_seed,
    train_one_epoch,
)
from selfplay import generate_mixed_selfplay_data


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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latest-path", type=str, default="checkpoints/latest_model.pt")
    parser.add_argument("--best-path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--log-path", type=str, default="logs/train_log.jsonl")
    parser.add_argument("--game-log-path", type=str, default="logs/train_game_log.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selfplay-workers", type=int, default=1)
    parser.add_argument("--selfplay-worker-device", type=str, default="cpu")
    parser.add_argument("--selfplay-temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-move", type=int, default=20)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.15)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--selfplay-vs-best-ratio", type=float, default=0.2)
    parser.add_argument(
        "--replay-path", type=str, default="logs/replay_buffer_latest.npz"
    )
    parser.add_argument("--replay-decay", type=float, default=0.03)
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
    device = resolve_device(args.device)

    latest_path = Path(args.latest_path)
    best_path = Path(args.best_path)
    log_path = Path(args.log_path)
    game_log_path = Path(args.game_log_path)
    replay_path = Path(args.replay_path)

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

    optimizer = optim.Adam(latest_model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay = deque(maxlen=replay_buffer_size)
    replay_birth_iters = deque(maxlen=replay_buffer_size)
    loaded_samples, loaded_birth_iters = load_replay_buffer(replay_path)
    if loaded_samples:
        loaded_samples = loaded_samples[-replay_buffer_size:]
        loaded_birth_iters = loaded_birth_iters[-replay_buffer_size:]
        replay.extend(loaded_samples)
        replay_birth_iters.extend(loaded_birth_iters)
    global_iter_base = max(replay_birth_iters) if replay_birth_iters else 0

    print("Continuous training start")
    print(
        f"from latest={latest_path} exists={loaded_latest}, "
        f"best={best_path} exists={best_path.exists()}"
    )
    print(device_banner(args.device, device))
    print(f"loaded replay samples={len(replay)} from {replay_path}")
    print(f"global_iter_base={global_iter_base}")
    if args.selfplay_workers > 1:
        print(
            f"parallel self-play enabled: workers={args.selfplay_workers}, "
            f"worker_device={args.selfplay_worker_device}"
        )
    print(
        "self-play exploration: "
        f"temperature={args.selfplay_temperature}, "
        f"temperature_drop_move={args.temperature_drop_move}, "
        f"dirichlet_alpha={args.dirichlet_alpha}, "
        f"dirichlet_epsilon={args.dirichlet_epsilon}"
    )
    print(
        f"training mix: vs_best_ratio={args.selfplay_vs_best_ratio}, "
        f"replay_decay={args.replay_decay}"
    )

    interrupted = False
    try:
        for it in range(1, args.max_iters + 1):
            global_it = global_iter_base + it
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

            mixed_games = int(args.games_per_iter * max(0.0, min(1.0, args.selfplay_vs_best_ratio)))
            self_games = max(0, args.games_per_iter - mixed_games)
            new_samples: list[Sample] = []
            if self_games > 0:
                self_samples = generate_selfplay_data(
                    model=latest_model,
                    num_games=self_games,
                    board_size=board_size,
                    simulations=args.simulations,
                    c_puct=c_puct,
                    device=device,
                    on_game_end=on_game_end,
                    num_workers=args.selfplay_workers,
                    worker_device=args.selfplay_worker_device,
                    temperature=args.selfplay_temperature,
                    temperature_drop_move=args.temperature_drop_move,
                    dirichlet_alpha=args.dirichlet_alpha,
                    dirichlet_epsilon=args.dirichlet_epsilon,
                )
                new_samples.extend(self_samples)
            if mixed_games > 0:
                mixed_offset = self_games

                def on_mixed_game_end(
                    game_idx: int, sample_count: int, result: GameResult, steps: int
                ) -> None:
                    if on_game_end is not None:
                        on_game_end(mixed_offset + game_idx, sample_count, result, steps)

                mixed_samples = generate_mixed_selfplay_data(
                    latest_model=latest_model,
                    best_model=best_model,
                    num_games=mixed_games,
                    board_size=board_size,
                    simulations=args.simulations,
                    c_puct=c_puct,
                    device=device,
                    on_game_end=on_mixed_game_end,
                    temperature=args.selfplay_temperature,
                    temperature_drop_move=args.temperature_drop_move,
                    dirichlet_alpha=args.dirichlet_alpha,
                    dirichlet_epsilon=args.dirichlet_epsilon,
                )
                new_samples.extend(mixed_samples)
            replay.extend(new_samples)
            replay_birth_iters.extend([global_it] * len(new_samples))
            print(f"[Iter {it}] samples +{len(new_samples)}; replay={len(replay)}")

            states, policies, values = prepare_tensors(list(replay), device=device)
            decay_weights = build_decay_weights(
                birth_iters=list(replay_birth_iters),
                current_iter=global_it,
                decay=args.replay_decay,
            ).to(device)
            loss = 0.0
            for epoch in range(1, args.epochs + 1):
                loss = train_one_epoch(
                    model=latest_model,
                    optimizer=optimizer,
                    states=states,
                    policy_targets=policies,
                    value_targets=values,
                    batch_size=args.batch_size,
                    sample_weights=decay_weights,
                    board_size=board_size,
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
                num_workers=args.selfplay_workers,
                worker_device=args.selfplay_worker_device
                if args.selfplay_workers > 1
                else None,
            )
            promoted = arena.candidate_win_rate >= promote_threshold
            if promoted:
                best_model.load_state_dict(deepcopy(latest_model.state_dict()))
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model.state_dict(), best_path)

            elapsed = time.time() - iter_start
            log_item = {
                "iteration": it,
                "global_iteration": global_it,
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
            save_replay_buffer(replay_path, list(replay), list(replay_birth_iters))
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
        save_replay_buffer(replay_path, list(replay), list(replay_birth_iters))
        if interrupted:
            print(f"Latest model checkpoint saved to: {latest_path}")
            print("Interrupted gracefully.")

    print("\nContinuous training finished.")


if __name__ == "__main__":
    main()
