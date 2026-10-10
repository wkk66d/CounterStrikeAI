# CounterStrikeAI —— CS2 回合胜率 GSI 单截面预测

> 从 Counter-Strike 2 中**某一个回合时刻的状态切面**，预测该回合 **CT 方获胜概率 P(CT)**。
> 本仓库是一个**独立可运行的最小项目**：只包含已训练好的模型 + 为 GSI（Game State Integration）设计的单截面实时预测部分。
> 项目语言：中文。

## 这是什么

CS2 的 GSI 会以 JSON 形式推送每一帧的玩家状态。本项目的目标是：

```
GSI JSON 单帧
   ↓  (gsi_to_snapshot)
内部回合切面快照
   ↓  (build_feature_vector)
117 维特征向量
   ↓  (模型 + 后处理)
P(CT 胜) ∈ (0,1)，T 胜 = 1 − P(CT)
```

它**不依赖 .dem / demoparser**、不读取磁盘上的录像，模型启动时加载一次后常驻内存，
之后每个切面只做一次推理，适合实时接入（约 2–3ms / 次）。

## 模型库

本仓库**并列分发两个已训练模型**，特征空间相同（117 维），可按版本切换：

| 版本 | 构件 | 训练方式 | 校准 | 留出集 Acc | AUC | Brier |
|---|---|---|---|---|---:|---:|---:|
| **0820**（默认） | `model/winrate_model_0820.joblib` | XGBoost（hist，早停 162 轮）单次训练 | 3 折 isotonic | **0.7369** | **0.8371** | **0.1628** |
| **1010** | `model/winrate_model_1010.joblib` | **分阶段 XGBoost**（弱信号先训练 + 上一阶段增量续训），1,155 棵树 / 8 阶段 | 3 折 isotonic | 0.7307 | 0.8311 | 0.1653 |

- 两个模型都在 **3,300,482 条切面 / 4,006 场**（`data/crawl/parts`）上训练，按比赛分组防泄漏，后处理一致
  （isotonic cv=3 + 线性拉伸 + logit 限幅 ±9，T=1.0，输出严格在 (0,1) 开区间，永不输出 0/1）；
- **默认仍为 0820**（指标略优），以保证既有接入方行为不变；1010 用 `--model 1010` 选用；
- 1010 的差异在**训练方式**：把 117 维特征按「信息家族」分 8 个阶段，
  按家族独立判别力**升序（弱信号先训练）**逐阶段累积，每阶段用上一阶段 booster 做增量续训
  （零掩码实现：固定 117 列，未训练家族置 0）。
  这样每个参数的影响都能体现到最终胜率上（实测 8/8 阶段均有非零贡献）。

```bash
# 查看模型库
python -m src.live_predict --list-models

# 用默认模型（0820）预测
python -m src.live_predict --gsi examples/gsi_example.json --snapshot-second 35 --time-remaining 62

# 用新模型（1010）预测
python -m src.live_predict --model 1010 --gsi examples/gsi_example.json --snapshot-second 35 --time-remaining 62
```

## 模型与概率说明

- 特征空间：**117 维** = 17（回合上下文 + 团队聚合）+ 10 名玩家 × 10 维。
- 不含跨回合信息（无比分差 / Momentum / 金钱），只看「本回合当前可见状态」。
- 两个模型均含 isotonic 校准 + 线性拉伸 + **logit 限幅 ±9（T=1.0）**，
  输出范围约 **[0.0001, 0.9999]**，严格位于 (0,1) 开区间，**不会直接输出 0 或 1**。
- 单次推理约 **2–4 ms**；模型启动时加载一次后常驻内存。

### winrate_model_1010 的分阶段训练顺序

| 阶段 | 信息家族 | 维度 |
|---:|---|---:|
| 1 | `S08_flash` 致盲 | 10 |
| 2 | `S05_time` 回合时间 | 2 |
| 3 | `S04_bomb` 炸弹状态 | 13 |
| 4 | `S06_map_position` 位置与地图 | 25 |
| 5 | `S01_survival` 存活/减员 | 13 |
| 6 | `S07_height` 高度 | 10 |
| 7 | `S03_health` 血量 | 12 |
| 8 | `S02_loadout` 武器护甲 | 32 |

本仓库**只分发训练好的模型，不包含原始数据**；模型的训练数据来自主项目 CSAI 收集的以下公开数据集：

| 来源 | 数据集 | 说明 |
|---|---|---|
| `blanchon` | [blanchon/cs2_dataset_demo](https://huggingface.co/datasets/blanchon/cs2_dataset_demo) | HLTV 职业 CS2 全场录像（主源，CC-BY-4.0，经 hf-mirror 访问） |
| `aldigobbler` | [aldigobbler/cs2demos](https://huggingface.co/datasets/aldigobbler/cs2demos) | HLTV 职业 CS2 录像（177 场） |
| `cs2pov` | [cs2povarchive/cs2-demos](https://huggingface.co/datasets/cs2povarchive/cs2-demos) | 职业赛事全程/视角录像 |
| `vasy` | [Vasy7777/cs2-demo-archive](https://huggingface.co/datasets/Vasy7777/cs2-demo-archive) | 比赛录像存档 |
| `esta` | [pnxenopoulos/esta](https://github.com/pnxenopoulos/esta) | CS:GO 2021~2022 已解析数据（CC-BY-SA-4.0） |

## 目录结构

```
CounterStrikeAI/
├── README.md            # 本文件
├── requirements.txt     # 依赖
├── model/
│   ├── winrate_model_0820.joblib   # XGBoost 模型（117 维，v0820，默认）
│   ├── winrate_model_0820.json      # 模型卡片（版本/指标/后处理）
│   ├── winrate_model_1010.joblib   # 分阶段 XGBoost 模型（117 维，v1010）
│   └── winrate_model_1010.json      # 模型卡片（含 8 阶段训练顺序）
├── src/
│   ├── config.py        # 最小配置 + 模型库注册表（--model 0820 / 1010）
│   ├── features.py      # 切面 → 117 维特征
│   ├── model.py         # 模型加载 / 概率后处理 / booster 构件适配
│   └── live_predict.py  # GSI 单截面实时预测（入口，--list-models 查看模型库）
└── examples/
    └── gsi_example.json # GSI JSON 样例
```

## 安装

> 模型为 XGBoost 训练产物，运行时需要 `xgboost` 包参与反序列化；已在 requirements.txt 中加好。

```bash
cd CounterStrikeAI
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 快速开始（命令行）

用项目自带的 GSI 样例预测一个 CT 胜率：

```bash
# 在项目根目录执行
python -m src.live_predict --gsi examples/gsi_example.json \
    --snapshot-second 35 --time-remaining 62
```

输出示例：

```json
{"p_ct_win": 0.1575, "p_t_win": 0.8425, "latency_ms": 2.37}
```

- `p_ct_win`：CT 方回合胜率
- `p_t_win`：T 方回合胜率 = 1 − p_ct_win
- `latency_ms`：单次推理耗时（不含模型加载）

如果已经有内部切面 JSON（见 `src/live_predict.py` 的 `gsi_to_snapshot` 输出结构），也可用 `--json` 传入。

## Python API

```python
from src.live_predict import LivePredictor

pred = LivePredictor()            # 启动时加载一次模型，之后常驻内存

# 方式一：直接传 GSI JSON 字典
p_ct = pred.predict_gsi(gsi_payload, snapshot_second=35.0, time_remaining_sec=62.0)

# 方式二：传内部切面字典
p_ct = pred.predict(snapshot_dict)

# 方式三：批量
p_cts = pred.predict_many([snapshot_1, snapshot_2, ...])
```

## GSI 输入字段说明

`gsi_to_snapshot()` 会把 CS2 GSI JSON 映射为内部切面，主要字段：

| GSI 字段 | 含义 | 备注 |
|---|---|---|
| `map.name` | 地图名（de_dust2 / de_inferno …） | 映射为 map_code |
| `map.round` | 当前回合号 | 用作用于回合识别/重置 |
| `round.phase` | 回合阶段（freezetime / live …） | 缺失时兜底 |
| `round.bomb` | 炸弹状态（planted 等） | → bomb_planted |
| `allplayers.*.team` | T / CT | 归一化为 team_num |
| `allplayers.*.state.health` | 血量 | 死亡=0 |
| `allplayers.*.state.armor / helmet / money / flashed` | 护甲 / 头盔 / 金钱 / 致盲 | money 不入特征，仅兼容 |
| `allplayers.*.weapons` | 手持武器 | → weapon_class |
| `allplayers.*.position` | 位置 “x, y, z” | 归一化用 |

> GSI 没有 tick 概念：调用时优先传入 `snapshot_second` 与 `time_remaining_sec`；
> 缺省时按 `round.phase` / `phase_ends_in` 近似，最终退化为 115 秒回合时钟。

## 测试

测试文件位于 `tests/`（**本地使用，已在 `.gitignore` 中排除，不随仓库上传**）。提交前请先运行：

```bash
python -m pytest tests -q
```

> `tests/` 下的用例（模型契约、GSI 单截面预测、0820/1010 双模型对照）仅作本地校验；
> 仓库中不包含任何测试文件。

## 许可

本项目衍生自内部研究项目 CSAI；模型数据来源为公开 HLTV 职业比赛 demo（CC-BY-4.0 数据源）。
