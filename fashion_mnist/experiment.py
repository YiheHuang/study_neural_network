"""
在 5 组固定随机种子下，并行跑三种训练脚本，并记录每组中三者的「测试集最佳 accuracy」。

依赖环境变量（由本脚本注入子进程）：
- FASHION_SEED：PyTorch / random / numpy 种子
- FASHION_CKPT_TAG：检查点文件名后缀，避免并行时互相覆盖

直接运行：python experiment.py
结果：同目录下 experiment_log.json 与 experiment_log.csv
"""
import concurrent.futures
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_BASE = Path(__file__).resolve().parent

# 五组种子（可改成你想要的整数）
SEEDS = [42, 123, 456, 1024, 2025]

# 三者并行时的检查点后缀（须与脚本内文件名前缀对应）
def _ckpt_tag(script: str, seed: int) -> str:
    mapping = {
        "main.py": f"s{seed}_opt",
        "main_fcnn.py": f"s{seed}_fcnn",
        "main_originalcnn.py": f"s{seed}_orig",
    }
    return mapping[script]


JOBS = [
    ("main.py", "optimized_cnn"),
    ("main_fcnn.py", "fcnn"),
    ("main_originalcnn.py", "original_cnn"),
]


def _parse_best_acc(stdout: str, stderr: str) -> Optional[float]:
    """从子进程输出中提取「最终摘要」里的 acc=xx.xx%。取最后一次匹配。"""
    text = stdout + "\n" + stderr
    matches = re.findall(r"acc=([\d.]+)%", text)
    if not matches:
        return None
    return float(matches[-1])


def _run_script(script: str, seed: int) -> dict:
    env = os.environ.copy()
    env["FASHION_SEED"] = str(seed)
    env["PYTHONHASHSEED"] = str(seed)
    env["FASHION_CKPT_TAG"] = _ckpt_tag(script, seed)

    proc = subprocess.run(
        [sys.executable, str(_BASE / script)],
        cwd=str(_BASE),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    acc = _parse_best_acc(proc.stdout, proc.stderr)
    return {
        "script": script,
        "seed": seed,
        "best_test_accuracy_percent": acc,
        "returncode": proc.returncode,
        "ckpt_tag": env["FASHION_CKPT_TAG"],
    }


def _run_seed_parallel(seed: int) -> Dict[str, Dict[str, Any]]:
    """同一种子下，三者并行各起一个子进程。"""
    results: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_map = {
            pool.submit(_run_script, script, seed): (script, label)
            for script, label in JOBS
        }
        for fut in concurrent.futures.as_completed(future_map):
            script, label = future_map[fut]
            try:
                results[label] = fut.result()
            except Exception as e:
                results[label] = {
                    "script": script,
                    "seed": seed,
                    "error": str(e),
                    "best_test_accuracy_percent": None,
                    "returncode": -1,
                }
    return results


def main():
    all_rows: List[Dict[str, Any]] = []
    summary_by_seed: Dict[int, Dict[str, Any]] = {}

    for seed in SEEDS:
        print(f"\n========== seed={seed}：并行启动三种脚本 ==========\n")
        per = _run_seed_parallel(seed)
        summary_by_seed[seed] = per
        for label in ["optimized_cnn", "fcnn", "original_cnn"]:
            row = per.get(label, {})
            row_out = {
                "seed": seed,
                "model": label,
                "best_test_accuracy_percent": row.get(
                    "best_test_accuracy_percent"
                ),
                "returncode": row.get("returncode"),
                "script": row.get("script"),
                "ckpt_tag": row.get("ckpt_tag"),
            }
            all_rows.append(row_out)
            acc = row_out["best_test_accuracy_percent"]
            rc = row_out["returncode"]
            print(
                f"  [{label}] best_acc={acc}% returncode={rc} "
                f"tag={row_out.get('ckpt_tag')}"
            )

    out_json = _BASE / "experiment_log.json"
    out_csv = _BASE / "experiment_log.csv"

    payload = {
        "seeds": SEEDS,
        "summary_by_seed": {
            str(k): v for k, v in summary_by_seed.items()
        },
        "flat_rows": all_rows,
    }
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "model",
                "best_test_accuracy_percent",
                "returncode",
                "script",
                "ckpt_tag",
            ],
        )
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(f"\n已写入：\n  {out_json}\n  {out_csv}")


if __name__ == "__main__":
    main()
