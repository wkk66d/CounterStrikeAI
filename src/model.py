# -*- coding: utf-8 -*-
'''AI 模型模块：训练 / 评估 / 预测 / 持久化。

评估协议：按比赛（demo）分组划分训练/测试集，避免同场比赛回合间信息泄漏。
指标：准确率、ROC AUC、Brier 分数。
'''

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# 概率后处理：使用 logit 限幅而不是硬截断。
# 这样输出严格落在 (0, 1) 开区间，两端可接近 0/1（如 0.0001~0.9999），
# 又不会因为数值溢出而直接输出 0 或 1。
DEFAULT_LOGIT_LIMIT = 9.0
# 置信度温度：T>1 时在 logit 空间压缩输出、降低过度自信（非时序方法）。
# 构件中未记录时默认 1.0（不缩放，保留更宽的两端）。
DEFAULT_TEMPERATURE = 1.0

_MODEL_REGISTRY = {
    "logreg": lambda seed: LogisticRegression(max_iter=3000, random_state=seed),
    "gbt": lambda seed: HistGradientBoostingClassifier(max_iter=300, random_state=seed),
    "mlp": lambda seed: MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                                      max_iter=1200, random_state=seed),
}


def _split(X, y, groups, test_frac, seed):
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        return np.asarray(train_idx), np.asarray(test_idx)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_frac, random_state=seed, stratify=y)
    return train_idx, test_idx


def evaluate(y_true, p_ct):
    '''评估指标集合；单类标签时 AUC 置 None（避免大预训练留出集崩溃）。'''
    y = np.asarray(y_true)
    p = np.asarray(p_ct)
    pred = (p >= 0.5).astype(int)
    auc = float(roc_auc_score(y, p))
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "auc": None if not np.isfinite(auc) else auc,
    }
    return metrics


def train_model(X, y, groups=None, models=("logreg", "gbt", "mlp"),
                test_frac=0.25, seed=42, feature_names=None, calibrate=True):
    '''训练若干候选模型，按 Brier 分数取最优，返回构件字典。

    calibrate=True 时对最优模型做 sigmoid 概率校准（保序变换，AUC 不变，
    显著改善两端过度自信的问题）。
    '''
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if models is None:
        models = tuple(_MODEL_REGISTRY)
    train_idx, test_idx = _split(X, y, groups, test_frac, seed)

    scaler = StandardScaler().fit(X[train_idx])
    X_tr = scaler.transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])

    best_model, best_metric, best_name = None, None, None
    trained_metrics = {}
    for name in models:
        est = _MODEL_REGISTRY[name](seed).fit(X_tr, y[train_idx])
        p_ct = est.predict_proba(X_te)[:, 1]
        m = evaluate(y[test_idx], p_ct)
        trained_metrics[name] = m
        if best_metric is None or m["brier"] < best_metric:
            best_model, best_metric, best_name = est, m["brier"], name

    p_ct_best = best_model.predict_proba(X_te)[:, 1]
    metrics = evaluate(y[test_idx], p_ct_best)
    metrics["model"] = best_name
    # 训练侧指标（用于观察 train/test 差距 → 过拟合程度）
    metrics["train_metrics"] = evaluate(y[train_idx],
                                        best_model.predict_proba(X_tr)[:, 1])

    raw_model = best_model
    p_ct_final = p_ct_best
    if calibrate:
        cal = CalibratedClassifierCV(estimator=raw_model, method="sigmoid", cv=5)
        cal.fit(X_tr, y[train_idx])
        p_ct_final = cal.predict_proba(X_te)[:, 1]
        metrics = evaluate(y[test_idx], p_ct_final)
        metrics["model"] = best_name
        metrics["calibrated"] = True
        metrics["train_metrics"] = evaluate(
            y[train_idx], cal.predict_proba(X_tr)[:, 1])
        best_model = cal

    if feature_names is None:
        names = [f"f{i}" for i in range(X.shape[1])]
    else:
        names = list(feature_names)
        if len(names) != X.shape[1]:
            raise ValueError("feature_names 长度与 X 列数不一致")

    return {
        "model": best_model,
        "raw_model": raw_model,
        "feature_names": names,
        "scaler": scaler,
        "metrics": metrics,
        "trained_metrics": trained_metrics,
        "split": {"train_ids": train_idx, "test_ids": test_idx,
                  "y_test": y[test_idx], "p_ct": p_ct_final,
                  "p_ct_uncalibrated": p_ct_best},
        "seed": seed,
    }


def continue_training(est, X, y, extra_iters=150, seed=42):
    '''热启动续训：保留原 GBT 已训练树，在新数据上继续增加 extra_iters 棵树。

    注意：这是「在线增量 boosting」——保留全部旧树，仅用新批次数据拟合新增树。
    它适合 200G 级 demo 的流式预训练（每个 .dem/批次调用一次），
    也适合预训练模型在新数据上的微调。
    '''
    if not isinstance(est, HistGradientBoostingClassifier):
        raise TypeError("continue_training 仅支持 HistGradientBoostingClassifier")
    if not hasattr(est, "n_iter_") or not hasattr(est, "_predictors"):
        raise ValueError("传入的 GBT 必须先完成一次 fit 才能热启动续训")
    base_iter = int(est.n_iter_)
    target_iter = base_iter + int(extra_iters)
    if target_iter <= base_iter:
        raise ValueError("extra_iters 必须为正数")
    est.set_params(warm_start=True, max_iter=target_iter, random_state=seed)
    return est.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64))


def finetune_from(pretrained_artifact, X, y, groups=None, test_frac=0.25,
                  seed=42, extra_iters=150, calibrate=True, feature_names=None):
    '''微调：加载「大规模预训练」模型 → 热启动续训 → 重新校准 → 评估。

    返回与 train_model 相同结构的构件字典（模型为微调后的 GBT + 校准器）。
    '''
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    train_idx, test_idx = _split(X, y, groups, test_frac, seed)
    scaler = StandardScaler().fit(X[train_idx])
    X_tr, X_te = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])

    base = pretrained_artifact["raw_model"]
    raw = continue_training(base, X_tr, y[train_idx], extra_iters=extra_iters,
                            seed=seed)
    raw_p_ct = raw.predict_proba(X_te)[:, 1]
    metrics = evaluate(y[test_idx], raw_p_ct)
    metrics["model"] = "gbt"
    metrics["train_metrics"] = evaluate(y[train_idx],
                                        raw.predict_proba(X_tr)[:, 1])

    p_ct_final = raw_p_ct
    best_model = raw
    if calibrate:
        cal = CalibratedClassifierCV(estimator=raw, method="sigmoid", cv=5)
        cal.fit(X_tr, y[train_idx])
        p_ct_final = cal.predict_proba(X_te)[:, 1]
        best_model = cal
        metrics = evaluate(y[test_idx], p_ct_final)
        metrics["model"] = "gbt"
        metrics["calibrated"] = True
        metrics["train_metrics"] = evaluate(
            y[train_idx], cal.predict_proba(X_tr)[:, 1])

    if feature_names is None:
        names = list(pretrained_artifact.get("feature_names") or [])
        if len(names) != X.shape[1]:
            names = [f"f{i}" for i in range(X.shape[1])]
    else:
        names = list(feature_names)
        if len(names) != X.shape[1]:
            raise ValueError("feature_names 长度与 X 列数不一致")

    return {
        "model": best_model,
        "raw_model": raw,
        "feature_names": names,
        "scaler": scaler,
        "metrics": metrics,
        "trained_metrics": {"gbt_finetune": metrics},
        "split": {"train_ids": train_idx, "test_ids": test_idx,
                  "y_test": y[test_idx], "p_ct": p_ct_final,
                  "p_ct_uncalibrated": raw_p_ct},
        "seed": seed,
        "finetuned_from": True,
    }


def _identity_transform(X):
    '''对 X 做恒等变换（用于树模型，不需要特征缩放）。'''
    return np.asarray(X, dtype=np.float64)


def _continuous_feature_mask(feature_names):
    '''识别连续型特征：只对这些特征加测试期噪声（离散/类别特征不加）。

    用于 MC 特征扰动集成（非时序方法）：在不依赖时间平滑的前提下，
    用多次小扰动预测的平均来降低观察噪声导致的波动。
    '''
    continuous_tails = {
        "snapshot_second", "time_remaining_sec", "bomb_countdown_sec",
        "avg_health", "equip_value", "avg_x", "avg_y",
        "health", "armor", "flash", "x", "y", "z",
    }
    mask = []
    for name in feature_names:
        tail = name.rsplit(".", 1)[-1] if "." in name else name
        mask.append(1 if tail in continuous_tails else 0)
    return np.asarray(mask, dtype=bool)


def predict_proba(artifact, X, mc_samples=None, mc_noise=None, seed=0):
    '''预测 (n, 2)：P(非CT胜), P(CT胜)。

    兼容两类构件：
      - 既有构件（如 winrate_model_0820）：`model` 为 sklearn 估计器
        （CalibratedClassifierCV 已内含 isotonic 校准）；
      - 分阶段构件（如 winrate_model_1010）：`model` 为 `XGBStagedClassifier`
        （包装原生 booster），校准在**构件级**施加（artifact["calibration"]["isotonic"]）。

    概率后处理（全部为非时序方法）：
      1. 可选 MC 特征扰动集成（mc_samples>1）：对连续特征加小高斯噪声，
         多次推理取平均，降低观察噪声造成的波动（仅适用于 sklearn 估计器）；
      2. isotonic 校准（分阶段构件）与 stretch（拉伸）映射到接近 0~1；
      3. 有限 logit 限幅 + 温度缩放：输出严格落在 (0, 1) 开区间，
         不出现 0/1，且在 logit 空间压缩过度自信。

    mc_samples / mc_noise 未显式传入时，读取构件中的 mc_samples / mc_noise
    （默认 1 与 0.1，即默认不做 MC 集成）。
    '''
    X = np.asarray(X, dtype=np.float64)
    expected = int(artifact["model"].n_features_in_)
    if X.shape[1] != expected:
        raise ValueError(f"特征宽度不匹配：期望 {expected}，实际 {X.shape[1]}")
    scaler = artifact.get("scaler")
    Xs = X if scaler is None else scaler.transform(X)
    n_samples = int(artifact.get("mc_samples", 1) if mc_samples is None else mc_samples)
    noise_std = float(artifact.get("mc_noise", 0.1) if mc_noise is None else mc_noise)

    # ---- 分阶段构件：模型只给原始概率，由 isotonic 在此统一施加 ----
    if is_booster_artifact(artifact):
        cal = artifact.get("calibration") or {}
        folds = cal.get("isotonic_folds")
        if folds:
            # K 折交叉校准：对每折 isotonic 映射取均值
            # （等价于 CalibratedClassifierCV(method="isotonic") 的折平均）
            p_raw = np.asarray(artifact["model"].predict_raw(Xs), dtype=np.float64)
            p_pos = np.mean([np.asarray(f.predict(p_raw), dtype=np.float64)
                             for f in folds], axis=0)
        else:
            p_pos = np.asarray(artifact["model"].predict_raw(Xs), dtype=np.float64)
            if cal.get("isotonic") is not None:
                p_pos = np.asarray(cal["isotonic"].predict(p_pos), dtype=np.float64)
    elif n_samples <= 1:
        p_pos = artifact["model"].predict_proba(Xs)[:, 1]
    else:
        names = list(artifact.get("feature_names")
                     or [f"f{i}" for i in range(X.shape[1])])
        cont = _continuous_feature_mask(names)
        rng = np.random.default_rng(seed)
        acc = np.zeros(len(X), dtype=np.float64)
        for _ in range(int(n_samples)):
            Xj = Xs.copy()
            if cont.any():
                Xj[:, cont] += rng.normal(0.0, noise_std, size=(len(X), int(cont.sum())))
            acc += artifact["model"].predict_proba(Xj)[:, 1]
        p_pos = acc / float(n_samples)

    stretch = artifact.get("stretch")
    if stretch is not None:
        lo, hi = float(stretch[0]), float(stretch[1])
        if hi - lo > 1e-9:
            p_pos = (p_pos - lo) / (hi - lo)
    # 先夹到极小正数，避免 logit 除零；再用 [-limit, limit] 限幅。
    eps = 1e-12
    p_pos = np.clip(p_pos, eps, 1.0 - eps)
    logit = np.log(p_pos / (1.0 - p_pos))
    limit = float(artifact.get("logit_limit", DEFAULT_LOGIT_LIMIT))
    logit = np.clip(logit, -limit, limit)
    temperature = float(artifact.get("temperature", DEFAULT_TEMPERATURE))
    if temperature > 0:
        logit = logit / temperature
    p_pos = 1.0 / (1.0 + np.exp(-logit))
    return np.column_stack([1.0 - p_pos, p_pos])


def save_model(artifact, path):
    '''持久化模型构件。'''
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def load_model(path):
    '''加载模型构件。'''
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 分阶段 XGBoost（booster 构件）的推理适配
# ---------------------------------------------------------------------------

def is_booster_artifact(artifact):
    '''判断构件是否为「分阶段 XGBoost booster」类型。

    识别两类：
      - `XGBStagedClassifier` 包装类（由本模块提供，带显式标记）；
      - 原生 `xgboost.Booster`（历史构件，未包装）。

    注意：不能只靠类名判断——包装类没有 `num_boosted_rounds`，
    早期实现因此漏判，导致 isotonic 校准被静默跳过（概率停留在未校准状态）。
    '''
    if not isinstance(artifact, dict):
        return False
    model = artifact.get("model")
    if model is None:
        return False
    if getattr(model, "_staged_booster_wrapper", False):
        return True
    return type(model).__name__ == "Booster" or hasattr(model, "num_boosted_rounds")


class XGBStagedClassifier:
    '''把原生 xgboost Booster 包装成 sklearn 风格分类器（只读推理）。

    分参数训练（弱信号先训练 + 增量续训）产出的是 `xgb.Booster`，
    其 `predict()` 需要 `xgb.DMatrix` 且要限定 `iteration_range`。
    本包装类提供 `n_features_in_` 与 `predict_proba(X)`，
    使 booster 构件能与 CalibratedClassifierCV 走同一条推理路径。

    参数
    ----
    postprocess : None / "identity"
        - None：`predict_proba` 返回**未后处理**的原始概率。
          构件级 `predict_proba` 会统一施加 isotonic 校准 + 拉伸 + logit 限幅，
          因此这里必须保持原始输出，否则后处理会被重复施加。
        - "identity"：保留接口以兼容早期构件（不改变数值）。
    '''

    def __init__(self, booster, n_features, n_trees=None, postprocess=None):
        self.booster = booster
        self.n_features_in_ = int(n_features)
        self.n_trees = int(n_trees or booster.num_boosted_rounds())
        self.postprocess = postprocess
        # 显式标记：供 is_booster_artifact 稳定识别（不依赖类名/属性探测）
        self._staged_booster_wrapper = True
        self.classes_ = np.asarray([0, 1])

    def predict_raw(self, X):
        '''未后处理的正类概率 P(CT 胜)。'''
        import xgboost as xgb  # 延迟导入：仅 booster 构件需要
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"特征宽度不匹配：期望 {self.n_features_in_}，实际 {X.shape[1]}")
        p = self.booster.predict(xgb.DMatrix(X),
                                 iteration_range=(0, self.n_trees))
        return np.asarray(p, dtype=np.float64).reshape(-1)

    def predict_proba(self, X):
        p = self.predict_raw(X)
        return np.column_stack([1.0 - p, p])
