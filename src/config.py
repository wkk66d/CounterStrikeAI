# -*- coding: utf-8 -*-
'''独立 GSI 单截面预测项目的精简配置。

只保留「单个回合状态切面 → 特征 → 胜率预测」所需的最小路径与常量。
模型文件放在项目根目录 model/ 下。
'''

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"

# ---------------- 模型库 ----------------
# 默认加载的模型：winrate_model_0820（XGBoost，117 维，isotonic + logit 限幅 ±9）
# 保持默认不变以保证既有接入方行为稳定；新模型通过 --model / MODEL_CANDIDATES 选用。
DEFAULT_MODEL_VERSION = "0820"
MODEL_ARTIFACT = MODEL_DIR / "winrate_model_0820.joblib"

# 可用模型注册表：版本号 → 构件路径
#   0820：3,300,482 条切面 / 4,006 场；Acc 0.7369 / AUC 0.8371 / Brier 0.1628
#   1010：分阶段 XGBoost（弱信号先训练 + 增量续训），同 117 维特征空间；
#         共 1,155 棵树、8 个训练阶段；Acc 0.7307 / AUC 0.8311 / Brier 0.1653
MODEL_CANDIDATES = {
    "0820": MODEL_DIR / "winrate_model_0820.joblib",
    "1010": MODEL_DIR / "winrate_model_1010.joblib",
}


def resolve_model(version_or_path=None):
    """把「版本号 / 路径 / None」解析为模型构件路径。

    - None            → 默认模型（MODEL_ARTIFACT）
    - "0820"/"1010"   → 注册表中的对应构件
    - 其他字符串       → 视为文件路径（存在则使用）
    """
    if version_or_path is None:
        return MODEL_ARTIFACT
    key = str(version_or_path).strip()
    if key in MODEL_CANDIDATES:
        return MODEL_CANDIDATES[key]
    path = Path(key)
    return path


# 回合切面常量（与训练端一致）
MAX_ROUND_SECONDS = 140.0
TEAM_T = 2
TEAM_CT = 3
TEAM_SIZE = 5
PLAYER_SLOTS = 10
