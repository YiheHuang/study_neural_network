# Gomoku AI 报告（当前实现说明）

本文档解释目前 `gomuku/` 项目中 AI 的 **网络构建**、**训练过程**、以及 **对弈逻辑**（从局面输入到落子输出）的完整链路。文中提到的代码以当前仓库实际实现为准。

---

## 1. 总览：从局面到落子的一条链路

在对弈（CLI 或 GUI）中，AI 的一次落子过程是：

1. **拿到当前棋盘局面**：`env/board.py::Board`
2. **编码为神经网络输入张量**：`model/predict.py::board_to_tensor(board)` 生成形状 `(3, 15, 15)`（再加 batch 维变为 `(1, 3, 15, 15)`）
3. **神经网络前向**：`model/network.py::GomokuNet.forward(x)`
   - 输出 `policy_logits`（225 维）与 `value`（标量，范围 -1~1）
4. **把 `policy_logits` 变成概率分布，并对非法落子做 mask**：`model/predict.py::predict_policy_value(...)`
5. **MCTS 搜索**：`mcts/search.py::MCTS.run(board, simulations=...)`
   - 多次模拟中不断调用神经网络来扩展叶子节点（得到先验 `P` 与价值 `V`）
6. **根据根节点孩子访问次数选择最终落子**：默认选择访问次数最多的落子
7. **落子到环境**：`Board.place_stone(row, col)`

训练过程使用同一套 “MCTS + 网络” 来生成自对弈数据，再用这些数据更新网络参数。

---

## 2. 环境与状态表示（`env/board.py`）

### 2.1 棋盘表示

- 棋盘大小默认 `15x15`
- 内部棋盘为 `numpy.ndarray`，形状 `(15, 15)`，类型 `int8`
- 棋子编码（`Player`）：
  - 空位：`0`
  - 黑：`1`
  - 白：`-1`

### 2.2 关键接口

- **合法落子检查**：`Board.is_legal_move(row, col)`
- **枚举所有合法点**：`Board.legal_moves()`
- **执行落子**：`Board.place_stone(row, col)`
- **胜负/和棋判断**：`Board.game_result()`
  - `GameResult.ONGOING | BLACK_WIN | WHITE_WIN | DRAW`
- **复制局面（供 MCTS 分支扩展）**：`Board.copy()`

### 2.3 胜负判断逻辑

`Board._check_five(row, col, player)` 在最近落子的四个方向统计连续子数：

- 垂直 `(1, 0)`
- 水平 `(0, 1)`
- 主对角 `(1, 1)`
- 副对角 `(1, -1)`

每个方向都向正反两侧扩展计数，若总数 `>= 5` 则判胜。

---

## 3. 神经网络构建（`model/network.py`）

### 3.1 输入与输出定义

网络类：`GomokuNet`

- **输入**：`x`，形状 `(B, 3, 15, 15)`
- **输出**：
  - `policy_logits`：形状 `(B, 225)`（对应 15x15 全部落点，未做 mask）
  - `value`：形状 `(B, 1)`，范围 `[-1, 1]`（通过 `tanh` 限幅）

`policy_logits` 之后会通过 `softmax` 变成概率分布；`value` 用于评估当前局面对当前行动方的好坏（训练中用终局胜负监督）。

### 3.2 状态编码（`model/predict.py::board_to_tensor`）

当前实现使用 3 个通道：

1. **当前行动方棋子平面**：当前位置是“当前行动方棋子”则为 1，否则 0
2. **对手棋子平面**：当前位置是“对手棋子”则为 1，否则 0
3. **回合标记平面**：若当前行动方是黑，整张平面填 1；否则整张平面填 0

输出张量为 `float32`，形状 `(3, 15, 15)`。

> 注：第 3 个通道是一种“让网络知道当前轮到谁”的方式；当前实现选择用“是否黑方行动”来编码。

### 3.3 网络骨干：Stem + Residual Blocks

#### Stem

```text
Conv3x3(in=3, out=channels) + BN + ReLU
```

#### ResidualBlock（重复 N 次）

每个残差块结构：

```text
Conv3x3 + BN + ReLU + Conv3x3 + BN + skip-add + ReLU
```

默认超参数（可通过命令行改）：

- `channels=64`
- `num_res_blocks=6`

### 3.4 双头输出（Policy Head / Value Head）

#### Policy Head

- `1x1 Conv: channels -> 2` + BN + ReLU
- Flatten 成 `2*15*15`
- `Linear(2*15*15 -> 225)` 得到 `policy_logits`

#### Value Head

- `1x1 Conv: channels -> 1` + BN + ReLU
- Flatten 成 `15*15`
- `Linear(225 -> 128) + ReLU`
- `Linear(128 -> 1) + tanh` 得到 `value`

---

## 4. 推理接口与合法落子 mask（`model/predict.py`）

### 4.1 `predict_policy_value(model, board, device)`

步骤：

1. `board_to_tensor(board)` 得到 `(3, 15, 15)`
2. `unsqueeze(0)` 得到 batch 维 `(1, 3, 15, 15)`
3. 网络前向得到 `policy_logits(1,225)`, `value(1,1)`
4. 对 `policy_logits[0]` 做 `softmax` 得到初始概率 `probs(225)`
5. 构造合法点 mask：`legal_moves_mask(board)`（合法为 1，否则 0）
6. `probs *= mask`，再做归一化，得到仅在合法点上分布的策略概率

输出：

- `policy_probs: np.ndarray(shape=(225,), dtype=float32)`：合法点上的概率分布
- `value: float`：局面价值

### 4.2 为什么要 mask？

网络的 policy 头输出的是对 **全部 225 个格点** 的偏好，但其中包含已落子位置（非法）。  
mask 的目的就是让 MCTS/落子选择只在合法点上进行。

---

## 5. MCTS 对弈逻辑（`mcts/search.py`）

### 5.1 节点统计量

节点类：`MCTSNode`

- `prior`：先验概率 \(P(s,a)\)，来自网络 policy
- `visit_count`：访问次数 \(N(s,a)\)
- `value_sum`：累计价值和 \(W(s,a)\)
- `value()`：平均价值 \(Q(s,a)=W/N\)
- `children`：落子 -> 子节点

### 5.2 搜索流程：选择 → 扩展 → 回传

入口：`MCTS.run(root_board, simulations, device)`

1. **根节点扩展**：调用 `_expand(root)`
2. 重复 `simulations` 次：
   - **选择（Selection）**：从根开始用 PUCT 选择子节点，直到到达叶子
   - **扩展（Expansion）**：叶子若非终局，调用网络得到 `priors,value` 并生成子节点
   - **回传（Backprop）**：沿路径更新 `visit_count/value_sum`，并做 value 的符号翻转

### 5.3 选择公式（PUCT）

当前实现对每个 child 计算：

- \(Q\)：使用 `child.value()`，并做了一次符号处理 `q = -child.value()`
- \(U\)：探索项

```text
score = q + c_puct * P * sqrt(N_parent) / (1 + N_child)
```

取 `score` 最大的 child 继续向下。

### 5.4 输出策略与最终落子

搜索结束后：

- 统计根节点每个子节点的访问次数，归一化为 `policy(225)`（visit distribution）
- 最终落子：选择访问次数最大的 move

> 这也是训练时 `policy_target` 的来源：根节点 visit count 分布。

---

## 6. 自对弈数据生成（`selfplay/generate.py`）

### 6.1 单局自对弈：`play_self_game(...)`

循环直到终局：

1. 保存当前局面的 `state = board_to_tensor(board)`（numpy）
2. 用 `MCTS.run(...)` 得到：
   - `move`：要下的点
   - `policy`：根节点 visit 分布（225 维）
3. 记录一条样本（state, policy, 当前行动方 player）
4. 执行 `board.place_stone(move)`

终局后，根据胜负结果回填每一步的 `value_target`：

- 若平局：`value=0`
- 若当前样本对应的行动方最终获胜：`value=+1`
- 若最终失败：`value=-1`

最终产出数据格式为列表：

```text
(state(3,15,15), policy(225), value(float))
```

### 6.2 多局数据生成：`generate_selfplay_data(...)`

参数 `num_games` 指定要生成多少盘。

此外提供 `on_game_end` 回调（训练脚本用它实现“每盘日志”）：

- 回调参数：`(game_idx, sample_count, game_result, steps)`

---

## 7. 训练过程（`train/run.py` 与 `train/continuous.py`）

训练有两种入口：

### 7.1 单次训练（最小闭环）`python -m train.run`

做法（简化版）：

1. 初始化模型
2. 生成自对弈数据（固定局数）
3. 把所有数据转成 Tensor（states/policies/values）
4. 做若干 epoch 梯度下降
5. 保存 `--save-path`（默认 `checkpoints/latest_model.pt`）

用途：

- 快速验证“能训练、能保存、能加载”

### 7.2 持续训练（推荐）`python -m train.continuous`

这是目前更完整的训练脚本，包含：

#### 7.2.1 断点续训与 best 初始化

启动时：

- `latest_model`：
  - 优先加载 `--latest-path`（默认 `checkpoints/latest_model.pt`）
  - 若不存在但 best 存在，则用 best 初始化 latest
- `best_model`：
  - 加载 `--best-path`（默认 `checkpoints/best_model.pt`）
  - 若不存在，则用 latest 初始化 best 并保存

#### 7.2.2 回放池（Replay Buffer）

使用 `deque(maxlen=replay_buffer_size)` 维护历史样本（容量来自 `configs/default.json` 的 `replay_buffer_size`）。

每轮迭代：

1. 自对弈生成新数据 `new_samples`
2. `replay.extend(new_samples)` 加入回放池
3. 从 `replay` 准备训练张量
4. 训练若干 epoch
5. 保存 `latest_model.pt`

#### 7.2.3 损失函数（Policy + Value）

在 `train/utils.py::train_one_epoch` 里：

- **policy loss**：使用交叉熵的等价形式（目标分布 `p` 与预测 log-prob 的点积）
  - `log_probs = log_softmax(logits)`
  - `policy_loss = -(p * log_probs).sum(dim=1).mean()`
- **value loss**：均方误差
  - `value_loss = mse(value_pred, v_target)`
- 总损失：
  - `loss = policy_loss + value_loss`

优化器：Adam（学习率 `--lr`）

#### 7.2.4 best 晋升（Arena 评估）

每轮训练后，用 `eval/arena.py::evaluate_models(...)` 进行对战评估：

- candidate：`latest_model`
- best：`best_model`
- 双方轮换先手（每局交换黑白）
- 使用 MCTS 来下棋（模拟次数可独立设置 `--eval-simulations`）

若胜率满足阈值（来自 `configs/default.json::promote_threshold`，默认 0.55）：

- 将 `latest_model` 晋升为 `best_model`
- 覆盖保存 `best_model.pt`

#### 7.2.5 日志输出（每盘 + 每轮）

`train.continuous` 现在有两类日志：

- **每盘自对弈日志**（即时输出 + JSONL）
  - 默认路径：`logs/train_game_log.jsonl`（参数 `--game-log-path`）
  - 字段包括：iteration、game_in_iteration、result、steps、samples、elapsed_sec
- **每轮迭代日志**（总结 JSONL）
  - 默认路径：`logs/train_log.jsonl`（参数 `--log-path`）
  - 字段包括：iteration、loss、replay_size、arena 统计、promoted 等

#### 7.2.6 中断保护（Ctrl+C）

脚本捕获 `KeyboardInterrupt`：

- 收到中断信号后会进入 `finally`，尽量保存一次 `latest_model.pt` 再退出

---

## 8. 对战应用：CLI 与 GUI

### 8.1 CLI：`app/play_cli.py`

- 支持人类输入坐标 `(row, col)`
- AI 回合使用 MCTS 搜索落子
- 可通过 `--model-path` 加载模型；空则随机初始化网络

### 8.2 GUI：`app/gui.py`（Tkinter）

- 鼠标点击落子
- AI 回合在后台线程搜索，避免界面卡死
- 默认会自动尝试加载：
  - `checkpoints/best_model.pt`，否则 `checkpoints/latest_model.pt`

---

## 9. 当前实现的关键假设与注意事项

1. **策略输出不包含“pass/认输”**：policy 维度固定为 225 个落点。
2. **无禁手规则**：黑先白后，不限制长连/三三等。
3. **训练数据来自自对弈**：没有外部棋谱监督。
4. **评估也用 MCTS**：因此评估耗时与 `eval_games * eval_simulations` 强相关。

