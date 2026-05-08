# 五子棋AI项目审查报告

## 审查范围

- 全部源代码模块：`env/`, `model/`, `mcts/`, `selfplay/`, `train/`, `eval/`, `app/`
- 训练日志：314轮迭代，16,043局对局
- 回放池数据：128,958条样本
- 配置：`configs/default.json` 及命令行参数

---

## 一、总体评价

项目整体架构遵循AlphaZero思路，核心流程（MCTS自对弈→回放池训练→best模型晋升）完整。代码质量良好，MCTS的virtual loss并行化、战术强制走法优化、数据增强（8-fold对称）、混合对手池等设计都是正确的。

**但存在若干关键问题，导致训练进展缓慢。以下按严重程度排序。**

---

## 二、关键问题

### 🔴 问题1（严重）：开局排除 + 对局过短 → 几乎无训练数据

**现象**：
- 最近50个迭代，平均每轮仅产生 **579条** 训练样本，而每轮打了 **200盘** 对局
- 平均每盘对局仅有 **2.9条** 训练样本
- 对局平均总步数约14手，但 `opening-random-moves=12` 意味着期望6手开局（Uniform{0..12}），这些开局步**不写入训练数据**

**根因**：
```python
# selfplay/generate.py: play_self_game()
n_open = _random_opening_num_moves(opening_random_moves)  # 0~12
_apply_mcts_opening_one_model(board, model, num_moves=n_open)  # 不写入history
# ...
while board.game_result() == GameResult.ONGOING:
    history.append(Sample(...))  # 仅此处的步写入训练数据
```

当对局总长度仅14手、开局期望6手时，训练数据被压缩到约8手/盘。更糟的是，许多对局在开局阶段就已结束（一方成五），产生**0条**训练样本——这解释了为何实际仅2.9条/盘。

**影响**：大量MCTS模拟算力（800 simulations × 200 games = 160K 次MCTS搜索）被浪费在产生不了训练数据的走法上。有效训练数据极度稀疏。

### 🔴 问题2（严重）：策略模式坍塌 — 对局越来越短

**现象**：
| 训练阶段 | 平均对局长度 |
|---------|------------|
| 早期（Games 1-1000） | ~35手 |
| 中期（Games 1000-4000） | ~25-30手 |
| 后期（Games 8000+） | ~9-14手 |

对局长度从35手跌至9手，模型找到某种"快速获胜模式"后不再探索深层次的对局。

**根因**：
1. 当 `temperature-drop-move=14` 且对局仅9-12手时，**所有落子都使用temperature采样**，从未进入贪心模式（temperature=0）
2. 这意味着self-play的MCTS始终进行探索性落子，模型从未在"最佳策略"下自我对弈
3. 策略网络收敛到局部最优——某种短期制胜模式，但缺乏深度

### 🔴 问题3（严重）：黑白胜负率严重振荡

**现象**：

| 对局区间 | 黑方胜率 |
|---------|---------|
| Games 1-700 | 46%-70%（正常波动）|
| Games 700-3600 | 70%-96%（黑方垄断）|
| Games 8000-11000 | 35%-41%（白方垄断）|
| Games 14000-15000 | 20%（白方80%胜率）|

模型在"极度偏向黑方"和"极度偏向白方"的策略之间反复振荡，而非收敛于平衡策略。

**根因**：
1. 回放池中黑方样本胜率61.6% vs 白方样本胜率43.0%——**价值标签存在系统性偏斜**
2. 当模型学到一个极度偏黑的策略后，self-play产生的数据也偏黑，形成正反馈循环
3. 回放池衰减（decay=0.03）不足以打破这个循环——新数据始终由当前偏斜的策略产生
4. 策略头瓶颈（见问题4）限制了模型学习对称策略的能力

### 🟡 问题4（中等）：策略头架构瓶颈

**当前设计**：
```python
self.policy_head = nn.Sequential(
    nn.Conv2d(64, 2, kernel_size=1),  # 64通道 → 2通道!
    nn.BatchNorm2d(2),
    nn.ReLU(),
)
self.policy_fc = nn.Linear(2 * 9 * 9, 81)  # 162 → 81
```

**问题**：将64通道的丰富特征压缩到仅2个通道，丢失了大量空间信息。AlphaZero原论文中策略头保持与主干相同的通道数（如256→256），然后用Conv2d直接映射到每格的动作logit。

**改进方向**：
```python
# 推荐方案
self.policy_head = nn.Conv2d(64, 32, kernel_size=1)
self.policy_fc = nn.Linear(32 * 81, 81)
```
或使用AlphaZero风格的全卷积策略头。

### 🟡 问题5（中等）：Arena评估对局数过少

**变化轨迹**：
- Global iter 1-20: **40局**评估
- Global iter 21-265: **20局**评估
- Global iter 266-315: **10局**评估

**问题**：10局评估的统计噪声极大。一个60%胜率（6胜4负）的95%置信区间约为[26%, 88%]。这意味着：
- 真实胜率40%的模型有约25%概率被误判为≥55%
- 真实胜率70%的模型有约15%概率被误判为<55%
- **晋升决策几乎随机**

### 🟡 问题6（中等）：输入编码的绝对方向平面

**当前设计**：
```python
turn_plane = np.full((size, size), 1.0 if board.current_player == Player.BLACK else 0.0)
```

turn_plane编码"当前是否是黑方回合"（绝对信息），而非总是1（相对视角）。这迫使网络学习"turn_plane=1时第一通道是黑子，turn_plane=0时第一通道是白子"——增加了不必要的学习负担。

**改进**：将turn_plane改为恒为1的全1平面（因为前两个通道已经从当前玩家视角编码了棋盘），或去除该平面减少冗余。

### 🟡 问题7（中等）：损失函数平台期

Loss从5.0降至约2.3-2.7后停滞。对于9×9棋盘（81个动作），随机策略的cross-entropy loss约为 ln(81)≈4.4。当前loss≈2.3意味着平均预测概率约10%，仍有提升空间。停滞原因可能是上述架构瓶颈与数据质量问题共同导致。

---

## 三、优化建议（按优先级）

### 🔥 优先级1：修复训练数据产量

**方案A**（推荐）：将开局步也纳入训练数据
```python
# 在 _apply_mcts_opening_one_model 中同时记录 history
# OR 将开局阶段的(state, policy, 最终value)也写入数据
```

**方案B**：大幅降低 `opening-random-moves`
```bash
--opening-random-moves 4  # 从12降到4，期望2手开局
```
这样平均每盘可产生~12条训练样本（vs 当前的2.9条）。

**方案C**：在开局阶段也记录训练样本，但使用更小的权重或独立的温度参数。

### 🔥 优先级2：修复策略头架构

将policy_head从 Conv2d(64→2) 升级为 Conv2d(64→32)，大幅增加策略信息容量。

具体修改 `model/network.py`：
```python
self.policy_head = nn.Sequential(
    nn.Conv2d(channels, 32, kernel_size=1, bias=False),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
)
self.policy_fc = nn.Linear(32 * board_size * board_size, self.policy_dim)
```

### 🔥 优先级3：增加Arena评估对局数

```bash
--eval-games 40  # 恢复至少40局
```
或在 `configs/default.json` 中将 `eval_games` 改为40。

10局评估时，observed 60%胜率的真实值可能在26%-88%之间，晋升决策基本无效。

### 优先级4：调整temperature_drop_move与对局长度匹配

当前对局平均仅9-14手，而 `temperature-drop-move=14`，导致永不用贪心。

**短期方案**：降低temperature_drop_move
```bash
--temperature-drop-move 6
```
让模型在开局后尽快切换到贪心模式。

**长期方案**：修复其他问题后对局会变长，可恢复较大的temperature_drop_move。

### 优先级5：修复黑白偏斜问题

1. **对称数据增强**：已有的8-fold旋转/翻转增强是正确的，但可以考虑加入**颜色翻转增强**（交换黑白通道 + 翻转value符号 + 翻转turn_plane），让模型学习到黑白对称性。

2. **将turn_plane改为相对编码**：恒为1.0或恒为0.0。

3. **平衡采样**：在训练时确保每个batch中黑方样本和白方样本数量平衡。

### 优先级6：调整损失函数

考虑将policy loss的权重降低、或使用不同的loss组合。当前 `loss = policy_loss + value_loss`，两者等权。如果策略头容量不足导致policy_loss始终较高，会压制value的学习。

### 优先级7：降低Dirichlet噪声强度

当前 `dirichlet_alpha=0.15`，对9×9约70个合法动作而言，Dirichlet(0.15)会产生少量动作获得大部分噪声质量的效果。可以考虑：
```bash
--dirichlet-alpha 0.3  # 或更高，使噪声更均匀
```

---

## 四、推荐的完整训练参数

```bash
python -m train.continuous \
  --games-per-iter 200 \
  --max-iters 10 \
  --simulations 800 \
  --eval-simulations 1200 \
  --batch-size 256 \
  --epochs 15 \
  --device cuda \
  --replay-decay 0.03 \
  --dirichlet-alpha 0.3 \
  --lr 5e-4 \
  --mcts-infer-batch-size 16 \
  --opening-random-moves 4 \
  --temperature-drop-move 8 \
  --opening-policy mcts \
  --opening-simulations 400 \
  --eval-games 40
```

主要调整：
| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| opening-random-moves | 12 | 4 | 大幅增加训练数据产量 |
| temperature-drop-move | 14 | 8 | 让后半段对局使用贪心 |
| dirichlet-alpha | 0.15 | 0.3 | 更均匀的探索 |
| eval-games | 10(默认) | 40 | 统计显著 |

**更重要的是**：请同步修改 `model/network.py` 中的策略头（问题4）和 `model/predict.py` 中的turn_plane编码（问题6）。

---

## 五、数据验证结论

回放池当前状态（128,958样本）：
- 价值分布基本均衡：52.2%胜 / 47.6%负 / 0.2%平
- 但对局日志显示黑白胜率严重偏斜，说明数据**内部存在时间序列相关**（不同时期产生不同偏斜的数据）
- 策略分布极度稀疏：中位数=0，均值=0.012（81个动作均匀分布则是0.0123）
- 49.6%的样本来自开局（<12手），这可能是因为对局本身就偏短

---

## 六、总结

核心问题是**训练数据产量不足**（opening_random_moves过大 + 对局过短）+ **模型架构瓶颈**（策略头过窄）+ **黑白偏斜振荡**三者的恶性循环：

```
策略头过窄 → 模型学到偏斜策略 (如只擅黑)
  → self-play产生偏斜数据
    → 训练强化偏斜
      → 对局变短 (快速制胜模式)
        → 更多步数落入开局排除区
          → 训练数据更少 → 模型无法学习纠正
```

修复建议的核心是：**扩大策略头、降低开局排除比例、增加评估可靠性**。
