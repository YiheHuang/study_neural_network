"""训练状态监控工具 —— 分析回放池、训练日志、对局日志，检测模式坍塌等异常。

用法:
  python training_monitor.py                     # 默认：完整报告
  python training_monitor.py --replay-only        # 仅回放池分析
  python training_monitor.py --trend              # 仅训练趋势
  python training_monitor.py --games              # 仅对局分析
  python training_monitor.py --watch              # 仅预警检查
  python training_monitor.py --replay-path logs/replay_buffer_latest.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# 修复 Windows GBK 编码下的 Unicode 输出问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np


# ── 工具函数 ──────────────────────────────────────────────

def _load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  -"
    return f"{n / total * 100:5.1f}%"


def _bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    filled = max(0, min(width, int(value / max_val * width)))
    return "█" * filled + "░" * (width - filled)


# ── 回放池分析 ────────────────────────────────────────────

def analyze_replay(replay_path: str) -> dict | None:
    p = Path(replay_path)
    if not p.exists():
        print(f"[提示] 回放池文件不存在: {replay_path}")
        return None

    data = np.load(p)
    values = data["values"]
    policies = data["policies"]
    states = data["states"]
    birth_iters = data["birth_iters"]
    n = len(values)

    total_stones = (states[:, 0, :, :] + states[:, 1, :, :]).sum(axis=(1, 2))
    max_probs = policies.max(axis=1)

    win_cnt = int((values == 1.0).sum())
    lose_cnt = int((values == -1.0).sum())
    draw_cnt = int((values == 0.0).sum())

    # 阶段划分
    opening = int((total_stones < 15).sum())
    midgame = int(((total_stones >= 15) & (total_stones < 40)).sum())
    endgame = int((total_stones >= 40).sum())

    return {
        "n": n,
        "values": values,
        "policies": policies,
        "stones": total_stones,
        "max_probs": max_probs,
        "birth_iters": birth_iters,
        "win_cnt": win_cnt,
        "lose_cnt": lose_cnt,
        "draw_cnt": draw_cnt,
        "opening": opening,
        "midgame": midgame,
        "endgame": endgame,
        "unique_iters": sorted(set(birth_iters.tolist())),
    }


def print_replay_report(r: dict) -> None:
    n = r["n"]
    if n == 0:
        print("[回放池为空]")
        return

    print("═══ 回放池分析 ═══")
    print(f"  总样本数: {n:>8,}")
    print(f"  覆盖迭代: {len(r['unique_iters'])} 轮 (iter {r['unique_iters'][0]} ~ {r['unique_iters'][-1]})")
    print()

    # 价值标签分布
    print("  ┌─ 价值标签分布 ─────────────────────────────┐")
    imbalance = abs(r["win_cnt"] - r["lose_cnt"]) / n * 100
    status = "✓ 均衡" if imbalance < 5 else ("△ 轻微偏斜" if imbalance < 10 else "✗ 严重偏斜")
    print(f"  │  +1 (胜): {r['win_cnt']:>6,}  {_pct(r['win_cnt'], n)}                  │")
    print(f"  │  -1 (负): {r['lose_cnt']:>6,}  {_pct(r['lose_cnt'], n)}                  │")
    print(f"  │   0 (平): {r['draw_cnt']:>6,}  {_pct(r['draw_cnt'], n)}                  │")
    print(f"  │  胜负数差: {abs(r['win_cnt'] - r['lose_cnt']):>6,}  偏斜率={imbalance:.1f}%  {status}     │")
    v_mean = float(r["values"].mean())
    print(f"  │  value均值: {v_mean:+.4f}  (正=偏胜, 零=均衡)               │")
    print(f"  └──────────────────────────────────────────────┘")
    print()

    # 对局阶段分布
    print("  ┌─ 对局阶段分布 ──────────────────────────────┐")
    for label, cnt, lo, hi in [
        ("开局 (<15手) ", r["opening"], 0, 14),
        ("中盘 (15-39)", r["midgame"], 15, 39),
        ("残局 (>=40) ", r["endgame"], 40, 81),
    ]:
        bar = _bar(cnt, n)
        print(f"  │  {label}: {cnt:>6,}  {_pct(cnt, n)}  {bar} │")
    avg_stones = float(r["stones"].mean())
    med_stones = float(np.median(r["stones"]))
    print(f"  │  落子数: 均值={avg_stones:.1f}  中位数={med_stones:.0f}                     │")
    stone_warn = ""
    if avg_stones < 20:
        stone_warn = "  ✗ 对局过短！可能模式坍塌"
    elif avg_stones < 30:
        stone_warn = "  △ 对局偏短，关注趋势"
    else:
        stone_warn = "  ✓ 对局长度健康"
    print(f"  │{stone_warn}           │")
    print(f"  └──────────────────────────────────────────────┘")
    print()

    # 策略质量
    print("  ┌─ 策略质量 ───────────────────────────────────┐")
    p_mean = float(r["max_probs"].mean())
    p_med = float(np.median(r["max_probs"]))
    p_25 = float(np.percentile(r["max_probs"], 25))
    p_75 = float(np.percentile(r["max_probs"], 75))
    p_90 = float(np.percentile(r["max_probs"], 90))
    print(f"  │  最大概率均值: {p_mean:.4f}   中位数: {p_med:.4f}                 │")
    print(f"  │  分位: 25%={p_25:.4f}  50%={p_med:.4f}  75%={p_75:.4f}  90%={p_90:.4f}   │")
    p_warn = "✓ 策略分布合理" if p_med < 0.5 else "△ MCTS过于自信（可能坍塌）"
    print(f"  │  {p_warn}                          │")
    print(f"  └──────────────────────────────────────────────┘")
    print()

    # 按迭代的样本量变化
    print("  ┌─ 每迭代样本量趋势 ───────────────────────────┐")
    for bi in r["unique_iters"]:
        cnt = int((r["birth_iters"] == bi).sum())
        bar = _bar(cnt, max(1, max((r["birth_iters"] == u).sum() for u in r["unique_iters"])))
        print(f"  │  iter {bi:>4d}: {cnt:>5,} 样本  {bar} │")
    print(f"  └──────────────────────────────────────────────┘")


# ── 训练日志分析 ──────────────────────────────────────────

def analyze_training(train_log_path: str) -> dict | None:
    p = Path(train_log_path)
    if not p.exists():
        print(f"[提示] 训练日志不存在: {train_log_path}")
        return None
    lines = _load_jsonl(train_log_path)
    if not lines:
        print("[提示] 训练日志为空")
        return None

    losses = [d["loss"] for d in lines]
    wrs = [d["arena"]["candidate_win_rate"] for d in lines]
    promoted = [d["promoted"] for d in lines]
    samples = [d["new_samples"] for d in lines]
    global_iters = [d["global_iteration"] for d in lines]

    return {
        "lines": lines,
        "losses": losses,
        "wrs": wrs,
        "promoted": promoted,
        "samples": samples,
        "global_iters": global_iters,
        "n": len(lines),
    }


def print_training_report(t: dict) -> None:
    if t is None:
        return

    print("═══ 训练趋势 ═══")
    print(f"  总迭代: {t['n']} 轮  (global {t['global_iters'][0]} ~ {t['global_iters'][-1]})")
    print()

    # Loss
    print(f"  Loss: {t['losses'][0]:.4f} → {t['losses'][-1]:.4f}  (Δ={t['losses'][-1] - t['losses'][0]:+.4f})")

    # 最近5轮loss变化
    if t["n"] >= 5:
        recent_deltas = [t["losses"][i] - t["losses"][i + 1] for i in range(-5, -1)]
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        if avg_delta > 0.02:
            print(f"  最近5轮 loss 下降速度: {avg_delta:.4f}/轮  ✓ 仍在快速学习")
        elif avg_delta > 0.005:
            print(f"  最近5轮 loss 下降速度: {avg_delta:.4f}/轮  △ 学习放缓")
        else:
            print(f"  最近5轮 loss 下降速度: {avg_delta:.4f}/轮  ✗ 接近停滞")

    # Arena
    promotions = sum(t["promoted"])
    recent_wrs = t["wrs"][-10:] if t["n"] >= 10 else t["wrs"]
    stuck_count = sum(1 for w in recent_wrs if w == 0.5)

    print()
    print(f"  晋升次数: {promotions}/{t['n']}  ({promotions / t['n'] * 100:.0f}%)")
    print(f"  最近{len(recent_wrs)}轮 Arena WR: ", end="")
    for w in recent_wrs:
        print(f"{w:.2f} ", end="")
    print()
    if stuck_count >= 5 and len(recent_wrs) >= 5:
        print(f"  ✗ 连续{stuck_count}轮 WR=0.50，可能已停滞！")

    # 样本量趋势
    if t["n"] >= 4:
        first_half_avg = sum(t["samples"][: t["n"] // 2]) / (t["n"] // 2)
        second_half_avg = sum(t["samples"][t["n"] // 2:]) / (t["n"] - t["n"] // 2)
        print()
        print(f"  前半段平均样本/轮: {first_half_avg:.0f}")
        print(f"  后半段平均样本/轮: {second_half_avg:.0f}")
        if second_half_avg < first_half_avg * 0.6:
            print(f"  ✗ 样本量大幅下降 ({(1 - second_half_avg / first_half_avg) * 100:.0f}%)，对局可能在缩短！")

    # 详细表格
    print()
    print(f"  {'Iter':<6} {'Loss':<9} {'Arena WR':<10} {'晋升':<6} {'样本数':<8}")
    print(f"  {'-' * 42}")
    for i, d in enumerate(t["lines"]):
        gi = d["global_iteration"]
        loss = d["loss"]
        wr = d["arena"]["candidate_win_rate"]
        prom = "✓" if d["promoted"] else ""
        samples = d["new_samples"]
        marker = ""
        if i == len(t["lines"]) - 1:
            marker = " ← 最新"
        print(f"  {gi:<6} {loss:<9.4f} {wr:<10.2f} {prom:<6} {samples:<8}{marker}")


# ── 对局日志分析 ──────────────────────────────────────────

def analyze_games(game_log_path: str, last_games: int = 0) -> dict | None:
    p = Path(game_log_path)
    if not p.exists():
        print(f"[提示] 对局日志不存在: {game_log_path}")
        return None
    lines = _load_jsonl(game_log_path)
    if not lines:
        print("[提示] 对局日志为空")
        return None

    if last_games > 0 and len(lines) > last_games:
        lines = lines[-last_games:]

    results = [g["result"] for g in lines]
    steps = [g["steps"] for g in lines]
    rc = Counter(results)
    total = len(lines)

    return {
        "lines": lines,
        "total": total,
        "results": results,
        "steps": steps,
        "black_wins": rc.get("BLACK_WIN", 0),
        "white_wins": rc.get("WHITE_WIN", 0),
        "draws": rc.get("DRAW", 0),
    }


def print_game_report(g: dict, bucket_size: int = 0) -> None:
    if g is None:
        return

    total = g["total"]
    results = g["results"]
    steps = g["steps"]

    print("═══ 对局分析 ═══")
    print(f"  总局数: {total:,}")
    print(f"  平均步数: {sum(steps) / total:.1f}")
    print()

    rc = Counter(results)
    bw_pct = rc.get("BLACK_WIN", 0) / total * 100 if total > 0 else 0
    print("  ┌─ 胜负分布 ───────────────────────────────────┐")
    print(f"  │  BLACK_WIN: {rc.get('BLACK_WIN',0):>6,}  {_pct(rc.get('BLACK_WIN',0), total)}                  │")
    print(f"  │  WHITE_WIN: {rc.get('WHITE_WIN',0):>6,}  {_pct(rc.get('WHITE_WIN',0), total)}                  │")
    print(f"  │  DRAW:      {rc.get('DRAW',0):>6,}  {_pct(rc.get('DRAW',0), total)}                  │")
    if bw_pct > 75:
        print(f"  │  ✗ 黑方胜率过高 ({bw_pct:.0f}%)，先手优势失控！              │")
    elif bw_pct > 65:
        print(f"  │  △ 黑方胜率偏高 ({bw_pct:.0f}%)，关注趋势                    │")
    else:
        print(f"  │  ✓ 黑白基本均衡                                    │")
    print(f"  └──────────────────────────────────────────────┘")
    print()

    # 分段趋势
    step_size = bucket_size if bucket_size > 0 else max(50, total // 15)
    n_buckets = (total + step_size - 1) // step_size

    print(f"  ┌─ 每{step_size}局趋势 ({n_buckets}档) ──────────────────────────────────────┐")
    print(f"  │  {'区间':<14} {'黑胜':>6} {'白胜':>6} {'平':>5} {'步数':>6}  趋势           │")
    print(f"  │  {'-' * 55} │")
    for i in range(0, total, step_size):
        bucket_r = results[i : i + step_size]
        bucket_s = steps[i : i + step_size]
        if len(bucket_r) < step_size // 4:
            continue
        bw = sum(1 for r in bucket_r if r == "BLACK_WIN")
        ww = sum(1 for r in bucket_r if r == "WHITE_WIN")
        dr = sum(1 for r in bucket_r if r == "DRAW")
        avg_s = sum(bucket_s) / len(bucket_s)
        game_range = f"{i + 1}-{i + len(bucket_r)}"
        bw_pct_val = bw / len(bucket_r)
        bar = f" 黑:{_bar(bw_pct_val, 1.0, 10)}" if bw_pct_val > 0.55 else ""
        flag = " !" if bw_pct_val > 0.75 else (" ?" if bw_pct_val > 0.65 else "")
        print(f"  │  {game_range:<14} {bw:>3}({bw_pct_val*100:2.0f}%) {ww:>3}({ww/len(bucket_r)*100:2.0f}%) {dr:>3}  {avg_s:>5.1f}{flag}  {bar} │")
    print(f"  └──────────────────────────────────────────────┘")
    print()

    # 最近一组的详细状态
    last_n = min(200, total)
    last_r = results[-last_n:]
    last_s = steps[-last_n:]
    last_bw = sum(1 for r in last_r if r == "BLACK_WIN")
    last_ww = sum(1 for r in last_r if r == "WHITE_WIN")
    last_dr = sum(1 for r in last_r if r == "DRAW")
    print(f"  最近{last_n}局:")
    print(f"    黑胜={last_bw}({last_bw / last_n * 100:.0f}%)  白胜={last_ww}({last_ww / last_n * 100:.0f}%)  平={last_dr}({last_dr / last_n * 100:.0f}%)  步数={sum(last_s) / last_n:.1f}")


# ── 预警检测 ──────────────────────────────────────────────

def print_warnings(replay: dict | None, training: dict | None, games: dict | None) -> int:
    warnings = 0

    print("═══ 预警检查 ═══")
    print()

    # 回放池预警
    if replay and replay["n"] > 0:
        n = replay["n"]
        imbalance = abs(replay["win_cnt"] - replay["lose_cnt"]) / n * 100
        avg_stones = float(replay["stones"].mean())

        if imbalance > 10:
            print(f"  ✗ [回放池] 价值偏斜率={imbalance:.1f}%（>10%），黑白训练数据严重不对称")
            warnings += 1
        elif imbalance > 5:
            print(f"  △ [回放池] 价值偏斜率={imbalance:.1f}%（>5%），稍有不均")
            warnings += 1
        else:
            print(f"  ✓ [回放池] 价值偏斜率={imbalance:.1f}%，黑白均衡")

        if avg_stones < 15:
            print(f"  ✗ [回放池] 平均落子数={avg_stones:.1f}（<15），对局极短，模式坍塌！")
            warnings += 1
        elif avg_stones < 25:
            print(f"  △ [回放池] 平均落子数={avg_stones:.1f}（<25），对局偏短")
            warnings += 1
        else:
            print(f"  ✓ [回放池] 平均落子数={avg_stones:.1f}，对局长度健康")

        # 最近迭代的样本量是否骤降
        if len(replay["unique_iters"]) >= 3:
            recent_iters = replay["unique_iters"][-3:]
            samples_per = [(replay["birth_iters"] == bi).sum() for bi in recent_iters]
            if len(samples_per) >= 3 and samples_per[0] > 0:
                if samples_per[-1] < samples_per[0] * 0.5:
                    print(f"  ✗ [回放池] 最近迭代样本量骤降: {samples_per[0]} → {samples_per[-1]}")
                    warnings += 1

    # 训练日志预警
    if training and training["n"] >= 8:
        recent_wrs = training["wrs"][-8:]
        stuck = sum(1 for w in recent_wrs if w == 0.5)
        if stuck >= 6:
            print(f"  ✗ [训练] 最近8轮中{stuck}轮 Arena WR=0.50，模型已停滞")
            warnings += 1
        elif stuck >= 3:
            print(f"  △ [训练] 最近8轮中{stuck}轮 Arena WR=0.50，可能开始停滞")
        else:
            print(f"  ✓ [训练] Arena 胜率正常波动，未停滞")

        # Loss下降速度
        if training["n"] >= 10:
            old_loss = sum(training["losses"][-10:-5]) / 5
            new_loss = sum(training["losses"][-5:]) / 5
            delta = old_loss - new_loss
            if delta < 0.01:
                print(f"  ✗ [训练] Loss 近乎停滞: 最近5轮均值={new_loss:.4f}, 前5轮={old_loss:.4f}")
                warnings += 1
            else:
                print(f"  ✓ [训练] Loss 仍在下降 (近5轮均值={new_loss:.4f})")

    # 对局日志预警
    if games and games["total"] >= 200:
        total = games["total"]
        window = min(200, total)
        last_r = games["results"][-window:]
        last_s = games["steps"][-window:]
        last_bw_pct = sum(1 for r in last_r if r == "BLACK_WIN") / len(last_r) * 100
        last_avg_s = sum(last_s) / len(last_s)

        if last_bw_pct > 80:
            print(f"  ✗ [对局] 最近{window}局黑胜率={last_bw_pct:.0f}%（>80%），先手优势失控！")
            warnings += 1
        elif last_bw_pct > 65:
            print(f"  △ [对局] 最近{window}局黑胜率={last_bw_pct:.0f}%（>65%），关注趋势")
        else:
            print(f"  ✓ [对局] 最近{window}局黑胜率={last_bw_pct:.0f}%，正常")

        if last_avg_s < 15:
            print(f"  ✗ [对局] 最近{window}局平均步数={last_avg_s:.1f}（<15），对局崩溃")
            warnings += 1
        elif last_avg_s < 25:
            print(f"  △ [对局] 最近{window}局平均步数={last_avg_s:.1f}（<25），偏短")
        else:
            print(f"  ✓ [对局] 最近{window}局平均步数={last_avg_s:.1f}，健康")

        # 趋势检测：黑胜率是否在持续上升
        seg_size = 50
        check_window = min(seg_size * 6, total)
        if check_window >= seg_size * 4:
            bw_trend = []
            for i in range(total - check_window, total, seg_size):
                seg = games["results"][i : i + seg_size]
                bw_trend.append(sum(1 for r in seg if r == "BLACK_WIN") / len(seg))
            if len(bw_trend) >= 3:
                increasing = all(bw_trend[j] <= bw_trend[j + 1] + 0.05 for j in range(len(bw_trend) - 1))
                if increasing and bw_trend[-1] > bw_trend[0] + 0.15:
                    print(f"  ✗ [对局] 黑胜率持续上升: {bw_trend[0]*100:.0f}% → {bw_trend[-1]*100:.0f}% (最近{check_window}局)")
                    warnings += 1

    print()
    if warnings == 0:
        print("  结论: ✓ 未检测到异常，训练状态健康")
    else:
        print(f"  结论: 检测到 {warnings} 个预警信号，建议检查")
    return warnings


# ── 主函数 ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gomoku AI 训练状态监控工具")
    p.add_argument("--replay-only", action="store_true", help="仅分析回放池")
    p.add_argument("--trend", action="store_true", help="仅训练趋势")
    p.add_argument("--games", action="store_true", help="仅对局分析")
    p.add_argument("--watch", action="store_true", help="仅预警检查")
    p.add_argument(
        "--replay-path", type=str, default="logs/replay_buffer_latest.npz"
    )
    p.add_argument(
        "--train-log", type=str, default="logs/train_log.jsonl"
    )
    p.add_argument(
        "--game-log", type=str, default="logs/train_game_log.jsonl"
    )
    p.add_argument(
        "--last-games", type=int, default=0,
        help="仅分析最近 N 盘对局 (0=全部)"
    )
    p.add_argument(
        "--bucket-size", type=int, default=0,
        help="趋势分析每档局数 (0=自动选择, 如100)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.replay_only:
        r = analyze_replay(args.replay_path)
        if r:
            print_replay_report(r)
    elif args.trend:
        t = analyze_training(args.train_log)
        if t:
            print_training_report(t)
    elif args.games:
        g = analyze_games(args.game_log, last_games=args.last_games)
        if g:
            print_game_report(g, bucket_size=args.bucket_size)
    elif args.watch:
        r = analyze_replay(args.replay_path)
        t = analyze_training(args.train_log)
        g = analyze_games(args.game_log, last_games=args.last_games)
        print_warnings(r, t, g)
    else:
        # 完整报告
        print()
        r = analyze_replay(args.replay_path)
        if r:
            print_replay_report(r)
        print()

        t = analyze_training(args.train_log)
        if t:
            print_training_report(t)
        print()

        g = analyze_games(args.game_log, last_games=args.last_games)
        if g:
            print_game_report(g, bucket_size=args.bucket_size)
        print()

        print_warnings(r, t, g)

    print()


if __name__ == "__main__":
    main()
