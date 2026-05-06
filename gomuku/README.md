# Gomuku: 15x15 Gomoku AI

这个项目用于从零构建一个基于神经网络的五子棋 AI。当前已完成：

- 15x15 无禁手规则环境
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

- `--board-size`（默认 `15`）：棋盘大小，表示 `N x N`。
- `--human`（默认 `black`，可选 `black|white`）：人类执子颜色。
- `--simulations`（默认 `120`）：AI 每步 MCTS 模拟次数，越大通常越强但更慢。
- `--model-path`（默认空）：模型权重路径；留空则使用随机初始化参数。
- `--device`（默认 `cpu`）：推理设备，如 `cpu` 或 `cuda`。
- `--channels`（默认 `64`）：网络主干通道数。
- `--res-blocks`（默认 `6`）：残差块数量。

### `python -m app.gui`

- `--board-size`（默认 `15`）：棋盘大小，表示 `N x N`。
- `--cell-size`（默认 `36`）：GUI 每个网格的像素大小。
- `--human`（默认 `black`，可选 `black|white`）：人类执子颜色。
- `--simulations`（默认 `80`）：AI 每步 MCTS 模拟次数。
- `--model-path`（默认空）：模型权重路径；留空时自动尝试 `best_model.pt`/`latest_model.pt`。
- `--device`（默认 `cpu`）：推理设备，如 `cpu` 或 `cuda`。
- `--channels`（默认 `64`）：网络主干通道数。
- `--res-blocks`（默认 `6`）：残差块数量。

### `python -m train.run`

- `--config`（默认 `configs/default.json`）：配置文件路径。
- `--iterations`（默认 `1`）：训练迭代轮数。
- `--games-per-iter`（默认 `2`）：每轮自对弈局数。
- `--simulations`（默认 `40`）：自对弈时每步 MCTS 模拟次数。
- `--batch-size`（默认 `64`）：训练批大小。
- `--epochs`（默认 `2`）：每轮训练 epoch 数。
- `--channels`（默认 `64`）：网络主干通道数。
- `--res-blocks`（默认 `6`）：残差块数量。
- `--device`（默认 `cpu`）：训练设备，如 `cpu` 或 `cuda`。
- `--lr`（默认 `1e-3`）：Adam 学习率。
- `--save-path`（默认 `checkpoints/latest_model.pt`）：训练后模型保存路径。
- `--seed`（默认 `42`）：随机种子。

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
- `--device`（默认 `cpu`）：训练/评估设备，如 `cpu` 或 `cuda`。
- `--lr`（默认 `1e-3`）：Adam 学习率。
- `--latest-path`（默认 `checkpoints/latest_model.pt`）：latest 模型路径（启动时优先加载，训练后覆盖保存）。
- `--best-path`（默认 `checkpoints/best_model.pt`）：best 模型路径（晋升后覆盖保存）。
- `--log-path`（默认 `logs/train_log.jsonl`）：每轮训练日志输出路径。
- `--game-log-path`（默认 `logs/train_game_log.jsonl`）：每盘自对弈即时日志输出路径。
- `--seed`（默认 `42`）：随机种子。

## 目录说明

- `env/`: 棋盘状态与规则
- `tests/`: 规则测试
- `app/`: 命令行演示、命令行对战、GUI 对战入口
- `model/`: 神经网络与推理接口
- `mcts/`: 蒙特卡洛树搜索
- `selfplay/`: 自对弈采样
- `train/`: 单轮与持续训练脚本
- `eval/`: 新旧模型对战评估与晋升依据
