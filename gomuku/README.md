# Gomuku: 9x9 Gomoku AI

这个项目用于从零构建一个基于神经网络的五子棋 AI。当前已完成：

- 9x9 无禁手规则环境
- 胜负/和棋判定
- 神经网络（策略头 + 价值头）
- MCTS 搜索落子；在展开树之前若存在 **己方一步成五** 或 **对手一步成五必堵** 的落点，则**直接下该手**（不跑仿真），减少明显漏着；连续杀等仍依赖搜索与网络

## 环境准备
- 命令行人机对战（支持随机参数模型或加载权重）
- 单元测试

## 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行测试

```bash
python -m pytest
```

## 运行随机对弈（阶段 A 验收）

```bash
python -m app.random_play
```

## 运行人机对战（随机参数神经网络）

```bash
python -m app.play_cli --human black --simulations 120
```

## 运行人机对战（加载训练权重）

```bash
python -m app.play_cli --human white --simulations 200 --model-path checkpoints/best_model.pt
```

## 启动最小训练（自对弈 + 训练 + 保存）

```bash
python -m train.run --iterations 1 --games-per-iter 2 --simulations 40 --epochs 2 --save-path checkpoints/latest_model.pt
```

训练完成后可直接加载：

```bash
python -m app.play_cli --human black --simulations 120 --model-path checkpoints/latest_model.pt
```

## 持续训练（自动从 latest 接着训练 + best 晋升 + 日志）

```bash
python -m train.continuous --max-iters 20 --games-per-iter 4 --simulations 40 --eval-simulations 20 --epochs 2
```

并行自对弈示例（推荐用于提速）：

```bash
python -m train.continuous --max-iters 20 --games-per-iter 64 --simulations 40 --selfplay-workers 8 --selfplay-worker-device cpu
```

增强随机探索示例（避免每盘几乎相同）：

```bash
python -m train.continuous --max-iters 20 --games-per-iter 64 --simulations 40 --selfplay-workers 8 --selfplay-worker-device cpu --selfplay-temperature 1.0 --temperature-drop-move 24 --dirichlet-alpha 0.3 --dirichlet-epsilon 0.25
```

随机开局示例（缓和先手偏重：每盘 **n~U{0,…,k}**；默认 **均匀随机** 开局 n 步后接 MCTS；**人机与 arena 仍空盘开局**）：

```bash
python -m train.continuous --opening-random-moves 8 --max-iters 20 --games-per-iter 50 --simulations 40 --selfplay-workers 1 --device cuda
```

与自对弈主阶段一致的 **MCTS visit softmax + `--selfplay-temperature`**（及 Dirichlet）开局示例：

```bash
python -m train.continuous --opening-random-moves 8 --opening-policy mcts --opening-simulations 16 ...
```

要点：`--opening-random-moves k`：每盘先 **等概率抽 n∈{0,…,k}**。`--opening-policy uniform`（默认）：交替 **均匀随机合法** n 步；`mcts`：交替 **各步跑一次 MCTS**（每步仿真次数由 **`--opening-simulations`** 控制，`≤0` 时用 **`--simulations`**），采样温度 **`--selfplay-temperature`**、根噪声 **`--dirichlet-*`** 与主对弈一致。开局步 **不写回放**；`--temperature-drop-move` 仍以 **第一手「训练」MCTS**（历史首条样本）起算。仅 **自对弈 / mixed**；人机与 arena 仍空盘。极少数开局已决胜负时该局可无训练样本。

行为说明：

- 启动时优先加载 `checkpoints/latest_model.pt`，不存在则随机初始化
- `checkpoints/best_model.pt` 不存在时会用当前模型初始化
- 每轮：
  - 自对弈采样并加入回放池
  - 训练 `latest_model`
  - 候选 `latest_model` 与 `best_model` 对战评估
  - 胜率达到阈值（`configs/default.json` 的 `promote_threshold`）才晋升为 `best_model`
- 每轮结果写入 `logs/train_log.jsonl`

## 启动 GUI（可点击落子）

```bash
python -m app.gui --human black --simulations 80
```

可选加载指定模型：

```bash
python -m app.gui --human white --simulations 120 --model-path checkpoints/best_model.pt
```

## 参数说明（全部 --arguments）

以下是当前项目所有可执行脚本的命令行参数说明。

### `python -m app.random_play`

该脚本当前无 `--参数`，直接运行即可。

### `python -m app.play_cli`

- `--board-size`（默认 `9`）：棋盘大小，表示 `N x N`。
- `--human`（默认 `black`，可选 `black|white`）：人类执子颜色。
- `--simulations`（默认 `120`）：AI 每步 MCTS 模拟次数，越大通常越强但更慢。
- `--model-path`（默认空）：模型权重路径；留空则使用随机初始化参数。
- `--device`（默认 `cuda`）：推理设备，如 `cuda` 或 `cpu`；若 CUDA 不可用会自动回退到 CPU。
- `--channels`（默认 `64`）：网络主干通道数。
- `--res-blocks`（默认 `6`）：残差块数量。
- `--mcts-infer-batch-size`（默认 `8`）：MCTS 叶节点 NN 批量推理尺寸；`1` 为逐步串行推理。
- `--disable-tactical-forced-moves`：关闭 MCTS 前「一步必杀 / 一步必堵」短路（消融用；默认启用战术层）。

### `python -m app.gui`

- `--board-size`（默认 `9`）：棋盘大小，表示 `N x N`。
- `--cell-size`（默认 `36`）：GUI 每个网格的像素大小。
- `--human`（默认 `black`，可选 `black|white`）：人类执子颜色。
- `--simulations`（默认 `80`）：AI 每步 MCTS 模拟次数。
- `--model-path`（默认空）：模型权重路径；留空时自动尝试 `best_model.pt`/`latest_model.pt`。
- `--device`（默认 `cuda`）：推理设备，如 `cuda` 或 `cpu`；若 CUDA 不可用会自动回退到 CPU。
- `--channels`（默认 `64`）：网络主干通道数。
- `--res-blocks`（默认 `6`）：残差块数量。
- `--mcts-infer-batch-size`（默认 `8`）：同 `play_cli`；`1` 等价于严格串行 NN 评估。
- `--disable-tactical-forced-moves`：同 `play_cli`。

### `python -m train.run`

- `--config`（默认 `configs/default.json`）：配置文件路径。
- `--iterations`（默认 `1`）：训练迭代轮数。
- `--games-per-iter`（默认 `2`）：每轮自对弈局数。
- `--simulations`（默认 `40`）：自对弈时每步 MCTS 模拟次数。
- `--batch-size`（默认 `64`）：训练批大小。
- `--epochs`（默认 `2`）：每轮训练 epoch 数。
- `--channels`（默认 `64`）：网络主干通道数。
- `--res-blocks`（默认 `6`）：残差块数量。
- `--device`（默认 `cuda`）：训练设备，如 `cuda` 或 `cpu`；若 CUDA 不可用会自动回退到 CPU。
- `--lr`（默认 `1e-3`）：Adam 学习率。
- `--save-path`（默认 `checkpoints/latest_model.pt`）：训练后模型保存路径。
- `--seed`（默认 `42`）：随机种子。
- `--selfplay-temperature`（默认 `1.0`）：自对弈前期按访问分布采样的温度；越大越随机。
- `--temperature-drop-move`（默认 `20`）：前多少手使用温度采样；之后改为贪心落子。
- `--dirichlet-alpha`（默认 `0.3`）：根节点 Dirichlet 噪声参数（探索强度）。
- `--dirichlet-epsilon`（默认 `0.25`）：根节点先验与噪声的混合比例。
- `--mcts-infer-batch-size`（默认 `8`）：MCTS 叶节点 NN 批量推理大小；设为 `1` 即与「每扩展一次单次前向」同构（无批量 virtual loss）。
- `--mcts-virtual-loss-weight`（默认 `1.0`）：批量 MCTS 的 virtual loss 强度。
- `--opening-random-moves`（默认 `0`）：整数 **k**。每盘抽样 **n∼Uniform({0,…,k})**，再走 **n** 步开局（见下条；不写回放）；`k≤0` 关闭。
- `--opening-policy`（默认 `uniform`，可选 `uniform|mcts`）：`uniform` = 开局步均匀随机合法点；`mcts` = 每步等同训练 MCTS 的 visit-softmax + `--selfplay-temperature`，根 Dirichlet 同 `--dirichlet-alpha`/`--dirichlet-epsilon`。
- `--opening-simulations`（默认 `0`）：`opening-policy=mcts` 时每步 MCTS 模拟次数；`≤0` 时使用 `--simulations`。
- `--disable-tactical-forced-moves`：关闭自对弈 MCTS 的一步必胜/必堵短路（消融）。

### `python -m train.continuous`

- `--config`（默认 `configs/default.json`）：配置文件路径。
- `--max-iters`（默认 `10`）：持续训练最大迭代轮数。
- `--games-per-iter`（默认 `2`）：每轮自对弈局数。
- `--simulations`（默认 `30`）：训练自对弈阶段每步 MCTS 模拟次数。
- `--eval-simulations`（默认 `20`）：候选模型与 best 对战评估时每步 MCTS 模拟次数。
- `--batch-size`（默认 `64`）：训练批大小。
- `--epochs`（默认 `2`）：每轮训练 epoch 数。
- `--channels`（默认 `64`）：网络主干通道数。
- `--res-blocks`（默认 `6`）：残差块数量。
- `--device`（默认 `cuda`）：训练/评估设备，如 `cuda` 或 `cpu`；若 CUDA 不可用会自动回退到 CPU。
- `--lr`（默认 `1e-3`）：Adam 学习率。
- `--latest-path`（默认 `checkpoints/latest_model.pt`）：latest 模型路径（启动时优先加载，训练后覆盖保存）。
- `--best-path`（默认 `checkpoints/best_model.pt`）：best 模型路径（晋升后覆盖保存）。
- `--log-path`（默认 `logs/train_log.jsonl`）：每轮训练日志输出路径。
- `--game-log-path`（默认 `logs/train_game_log.jsonl`）：每盘自对弈即时日志输出路径。
- `--seed`（默认 `42`）：随机种子。
- `--selfplay-workers`（默认 `1`）：自对弈与 arena 评估的并行进程数；大于 1 时，自对弈与 `latest vs best` 评估均按该进程数并行（单进程时 arena 使用 `--device`）。
- `--selfplay-worker-device`（默认 `cpu`）：并行 worker 的推理设备（自对弈与 arena 共用），推荐 `cpu`（避免多进程争抢单卡）。
- `--selfplay-temperature`（默认 `1.0`）：自对弈前期按访问分布采样的温度；越大越随机。
- `--temperature-drop-move`（默认 `20`）：前多少手使用温度采样；之后改为贪心落子。
- `--dirichlet-alpha`（默认 `0.3`）：根节点 Dirichlet 噪声参数（探索强度）。
- `--dirichlet-epsilon`（默认 `0.25`）：根节点先验与噪声的混合比例。
- `--selfplay-vs-old-ratio`（默认 `0.2`）：混入 **latest vs 随机老对手** 的局占比。对手在 **「当前 best」** 与 **`--snapshot-dir` 下所有 `iter_GGGGG_model.pt` 快照** 之间 **均匀随机**，每局重新抽一次（其余局为 `latest vs latest`）。
- `--selfplay-vs-best-ratio`（已弃用）：若显式传入则 **覆盖** `--selfplay-vs-old-ratio` 数值，语义与 vs-old 相同。
- `--snapshot-every`（默认 `20`）：按 **global_iteration**，每满 N 轮（且 `N>0`）在 `--snapshot-dir` 保存 `iter_<iteration>_model.pt`（当前 **`latest` 训练权重**）。设为 `0` 关闭周期快照（仍可用目录里已有快照进池）。
- `--snapshot-dir`（默认 `checkpoints/snapshots`）：周期快照存放目录；启动时会 **扫描**其中符合 `iter_*_model.pt` 的文件加入对手池。
- `--replay-path`（默认 `logs/replay_buffer_latest.npz`）：回放池持久化文件路径；启动会自动恢复并在训练中持续覆盖保存。
- `--replay-decay`（默认 `0.03`）：训练采样衰减系数；越大越偏向最近数据。
- `--mcts-infer-batch-size`（默认 `8`）：自对弈、mixed、`arena` 评估（`latest vs best`）中的 MCTS 批量推理尺寸。
- `--mcts-virtual-loss-weight`（默认 `1.0`）：批量 MCTS 的 virtual loss 强度。
- `--opening-random-moves`（默认 `0`）：**k**。每盘 **n∼U({0,…,k})** 后执行 **n** 步开局（**不写回放**）；人机与 arena **不参与**；`k≤0` 等价关闭。
- `--opening-policy`（默认 `uniform`）：开局步：`uniform`|`mcts`（见 train.run）。
- `--opening-simulations`（默认 `0`）：见 train.run。
- `--disable-tactical-forced-moves`：关闭自对弈、mixed、`arena` 中 MCTS 的一步必胜/必堵短路（消融）。

## 目录说明

- `env/`: 棋盘状态与规则；`env/tactics.py` 为一步必胜/必堵检测（由 MCTS 调用）
- `tests/`: 规则测试
- `app/`: 命令行演示、命令行对战、GUI 对战入口
- `model/`: 神经网络与推理接口
- `mcts/`: 蒙特卡洛树搜索
- `selfplay/`: 自对弈采样
- `train/`: 单轮与持续训练脚本
- `eval/`: 新旧模型对战评估与晋升依据
