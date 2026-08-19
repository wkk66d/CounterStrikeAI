# -*- coding: utf-8 -*-
'''独立 GSI 单截面预测项目的精简配置。

只保留「单个回合状态切面 → 特征 → 胜率预测」所需的最小路径与常量。
模型文件放在项目根目录 model/ 下。
'''

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
# 默认加载的模型：117 维 early-stopping MLP + isotonic + logit 限幅 ±9
MODEL_ARTIFACT = MODEL_DIR / "winrate_model.joblib"

# 回合切面常量（与训练端一致）
MAX_ROUND_SECONDS = 140.0
TEAM_T = 2
TEAM_CT = 3
TEAM_SIZE = 5
PLAYER_SLOTS = 10
