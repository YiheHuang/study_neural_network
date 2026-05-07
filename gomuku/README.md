# Gomuku: 9x9 Gomoku AI

这个项目用于从零构建一个基于神经网络的五子棋 AI。当前已完成：

- 9x9 无禁手规则环境
- 胜负/和棋判定
- 神经网络（策略头 + 价值头）
- MCTS 搜索落子
- 命令行随机对弈演示
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

随机开局示例（缓和先手偏重：每盘 **n~U{0,…,k}** 再随机开局 n 步，再 MCTS；**人机与 arena 仍从空盘开始**）：

```bash
python -m train.continuous --opening-random-moves 8 --max-iters 20 --games-per-iter 50 --simulations 40 --selfplay-workers 1 --device cuda
```

要点：`--opening-random-moves k` 表示每盘先 **等概率抽 n∈{0,…,k}**，再交替 **均匀随机合法落子** n 步（黑先手 ⇒ 黑子数 **⌊(n+1)/2⌋**、白子数 **⌊n/2⌋**）。随机步 **不写回放**；`--temperature-drop-move` 仍以 **第一手 MCTS** 起算。仅 **自对弈 / mixed**；**人机与 arena 仍空盘**。极少数随机子已决出胜负时该局可无 MCTS 样本。

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
- `--opening-random-moves`（默认 `0`）：整数 **k**。每盘抽样 **n∼Uniform({0,…,k})**，再走 n 步随机开局（不写回放）；`k≤0` 关闭。

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
- `--selfplay-vs-best-ratio`（默认 `0.2`）：训练采样时混入 `latest vs best` 对局占比（其余为 `latest vs latest`）。
- `--replay-path`（默认 `logs/replay_buffer_latest.npz`）：回放池持久化文件路径；启动会自动恢复并在训练中持续覆盖保存。
- `--replay-decay`（默认 `0.03`）：训练采样衰减系数；越大越偏向最近数据。
- `--mcts-infer-batch-size`（默认 `8`）：自对弈、mixed、`arena` 评估（`latest vs best`）中的 MCTS 批量推理尺寸。
- `--mcts-virtual-loss-weight`（默认 `1.0`）：批量 MCTS 的 virtual loss 强度。
- `--opening-random-moves`（默认 `0`）：**k**。每盘 **n∼U({0,…,k})** 后执行 n 步随机开局（**不写回放**）；**不包含**人机与 arena；`k≤0` 等价关闭。

## 目录说明

- `env/`: 棋盘状态与规则
- `tests/`: 规则测试
- `app/`: 命令行演示、命令行对战、GUI 对战入口
- `model/`: 神经网络与推理接口
- `mcts/`: 蒙特卡洛树搜索
- `selfplay/`: 自对弈采样
- `train/`: 单轮与持续训练脚本
- `eval/`: 新旧模型对战评估与晋升依据
