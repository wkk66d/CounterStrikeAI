# -*- coding: utf-8 -*-
'''实时单切面预测（GSI 接入预留）。

设计目标：
  1. 模型只加载一次（LivePredictor 内部缓存）；
  2. 每个切面只做「字典 → 117 维特征 → 一次 predict_proba」；
  3. 不依赖 .dem / demoparser，可被 GSI 每 tick 调用；
  4. 提供 gsi_to_snapshot 把 CS2 GSI JSON 载荷转换为内部切面字典。

典型用法：
  pred = LivePredictor()                # 启动时加载一次
  p_ct = pred.predict(snapshot)         # 内部切面字典
  p_ct = pred.predict_gsi(gsi_payload, snapshot_second=35.0,
                          time_remaining_sec=62.0)
'''

import argparse
import json
import time

import numpy as np

from src import config
from src.features import build_feature_vector, weapon_class_of
from src.model import load_model, predict_proba

# GSI 武器 type → 主武器选择优先级（C4 单独识别，不参与武器类别编码）
_WEAPON_TYPE_RANK = {
    "C4": 90, "Rifle": 80, "SniperRifle": 75, "Machine Gun": 70,
    "Submachine Gun": 60, "Shotgun": 50, "Pistol": 40,
    "Grenade": 10, "Knife": 5,
}
_NON_PRIMARY_TYPES = {"Grenade", "Knife", "C4"}


def _to_int_id(value, fallback):
    '''SteamID 归一化；GSI 缺失时用稳定的负递增 ID 兜底。'''
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return int(fallback)


def _team_num(team_value):
    team = str(team_value or "").strip().upper()
    if team in ("CT", "COUNTERTERRORIST", "COUNTER-TERRORIST", "3"):
        return 3
    if team in ("T", "TERRORIST", "2"):
        return 2
    return None


def _parse_position(raw_position):
    if isinstance(raw_position, (list, tuple)):
        parts = [float(v or 0.0) for v in raw_position[:3]]
    else:
        parts = [float(v or 0.0) for v in str(raw_position or "").split(",")[:3]]
    parts += [0.0] * (3 - len(parts))
    return parts[:3]


def _primary_weapon_name(weapons):
    '''从 GSI weapons 中挑出与 demoparser active_weapon 最接近的主武器。'''
    weapons = weapons or {}
    best_name, best_rank = "", -1
    for w in weapons.values():
        if not isinstance(w, dict):
            continue
        name = str(w.get("name") or "").lower()
        state = str(w.get("state") or "").lower()
        if "c4" in name:
            # C4 只在手持时充当 active_weapon；背在包位时不参与武器类别
            if state == "active":
                return name
            continue
        wtype = str(w.get("type") or "")
        rank = _WEAPON_TYPE_RANK.get(wtype, 0)
        # 优先使用当前 active 且不是投掷物/刀的主武器
        if state == "active" and wtype not in _NON_PRIMARY_TYPES:
            return name
        if rank > best_rank:
            best_name, best_rank = name, rank
    return best_name


def _first_value(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def gsi_to_snapshot(payload, snapshot_second=None, time_remaining_sec=None,
                    bomb_countdown_sec=None, coords=None, half=None,
                    tick_rate=None):
    '''把 CS2 GSI 载荷转换为 build_feature_vector 可消费的内部切面字典。

    GSI 没有 tick 概念：优先使用显式传入的 snapshot_second / time_remaining_sec；
    缺省时用 round.phase 和 round.phase_ends_in 做近似，再退化为 115 秒回合时钟。
    '''
    payload = payload or {}
    map_info = payload.get("map") or {}
    round_info = payload.get("round") or {}
    phase = str(round_info.get("phase") or "live").lower()

    if snapshot_second is None:
        snapshot_second = _first_value(
            payload, ("snapshot_second", "round_seconds", "round_time_elapsed"))
    if snapshot_second is None:
        snapshot_second = 0.0 if phase == "freezetime" else 15.0
    snapshot_second = float(snapshot_second)

    if time_remaining_sec is None:
        time_remaining_sec = payload.get("time_remaining_sec")
    if time_remaining_sec is None and phase == "live":
        raw = round_info.get("phase_ends_in")
        if raw is not None:
            time_remaining_sec = float(raw)
    if time_remaining_sec is not None:
        time_remaining_sec = float(time_remaining_sec)
    if bomb_countdown_sec is None:
        bomb_countdown_sec = payload.get("bomb_countdown_sec")
    if bomb_countdown_sec is not None:
        bomb_countdown_sec = float(bomb_countdown_sec)

    tick_rate = float(tick_rate or payload.get("tick_rate") or 64)
    snapshot_tick = int(round(snapshot_second * tick_rate))
    round_start_tick = 0
    round_num_raw = _first_value(map_info, ("round",),
                                 payload.get("round_num"))
    snap = {
        "map_name": map_info.get("name") or payload.get("map_name") or "unknown",
        "round_num": int(round_num_raw if round_num_raw is not None else 1),
        "half": 0,
        "snapshot_second": snapshot_second,
        "snapshot_tick": snapshot_tick,
        "round_start_tick": round_start_tick,
        "tick_rate": tick_rate,
        "bomb_planted_before": 1 if round_info.get("bomb") == "planted" else 0,
        "bomb_countdown_sec": float(bomb_countdown_sec or 0.0),
        "deaths_before": 0,
        "winner": 0,
        "label_ct_win": 0,
        "players": [],
    }
    if half is not None:
        snap["half"] = int(half)
    else:
        snap["half"] = 0 if snap["round_num"] <= 12 else 1
    if time_remaining_sec is not None:
        snap["time_remaining_sec"] = time_remaining_sec
        snap["round_end_tick"] = snapshot_tick + int(round(time_remaining_sec * tick_rate))
    if coords is not None:
        snap["coords"] = dict(coords)

    fallback_id = -1
    for key, p in (payload.get("allplayers") or {}).items():
        if not isinstance(p, dict):
            continue
        team_num = _team_num(p.get("team"))
        if team_num is None:
            continue
        state = p.get("state") or {}
        health = float(state.get("health") or 0)
        alive = 1 if health > 0 else 0
        weapons = p.get("weapons") or {}
        weapon_name = _primary_weapon_name(weapons)
        has_c4 = 1 if any(
            "c4" in str(w.get("name") or "").lower()
            or str(w.get("type") or "").upper() == "C4"
            for w in weapons.values() if isinstance(w, dict)
        ) else 0
        x, y, z = _parse_position(p.get("position"))
        id_source = p.get("steamid") or p.get("observer_slot") or key
        steamid = _to_int_id(id_source, fallback_id)
        fallback_id -= 1
        snap["players"].append({
            "steamid": steamid,
            "team_num": team_num,
            "alive": alive,
            "health": health if alive else 0.0,
            "armor": float(state.get("armor") or 0) if alive else 0.0,
            "helmet": int(bool(state.get("helmet"))) if alive else 0,
            "money": float(state.get("money") or 0) if alive else 0.0,
            "weapon_class": weapon_class_of(weapon_name),
            "weapon_name": weapon_name,
            "has_c4": has_c4,
            "flash": float(state.get("flashed") or 0) if alive else 0.0,
            "x": x if alive else 0.0,
            "y": y if alive else 0.0,
            "z": z if alive else 0.0,
        })
    snap["deaths_before"] = 10 - sum(1 for p in snap["players"] if p["alive"])
    return snap


class LivePredictor:
    '''单切面实时预测器：启动时加载模型，之后每次推理不触碰磁盘与 demo。'''

    def __init__(self, model_path=None):
        self.model_path = model_path or config.MODEL_ARTIFACT
        self.artifact = load_model(self.model_path)
        self._model = self.artifact["model"]
        self._scaler = self.artifact["scaler"]
        self.feature_names = list(self.artifact["feature_names"])

    def _predict_vector(self, vec):
        # 与 predict_demo 使用同一套概率后处理（stretch + logit 限幅 + 温度缩放），
        # 保证实时 GSI 与离线 demo 重放得到一致、处于 (0,1) 开区间的结果。
        return float(predict_proba(self.artifact, np.asarray(vec, dtype=np.float64)
                                   .reshape(1, -1))[0, 1])

    def predict(self, snapshot):
        '''内部切面字典 → P(CT 胜)。'''
        return self._predict_vector(build_feature_vector(snapshot))

    def predict_many(self, snapshots):
        '''批量单切面预测（GSI 多路/回放重算时使用）。'''
        if not snapshots:
            return np.empty((0,), dtype=np.float64)
        X = np.vstack([build_feature_vector(s) for s in snapshots])
        return np.asarray(predict_proba(self.artifact, X)[:, 1], dtype=np.float64)

    def predict_json(self, json_text):
        '''JSON 字符串切面 → P(CT 胜)。'''
        return self.predict(json.loads(json_text))

    def predict_gsi(self, gsi_payload, snapshot_second=None,
                    time_remaining_sec=None, **kwargs):
        '''GSI JSON 字典 → P(CT 胜)。'''
        return self.predict(gsi_to_snapshot(gsi_payload,
                                            snapshot_second=snapshot_second,
                                            time_remaining_sec=time_remaining_sec,
                                            **kwargs))


def _load_payload(path):
    return json.loads(open(path, encoding="utf-8").read())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CS2 实时单切面胜率预测（GSI 接入预留）")
    parser.add_argument("--model", default=None, help="模型构件路径")
    parser.add_argument("--json", default=None, help="内部切面 JSON 文件")
    parser.add_argument("--gsi", default=None, help="CS2 GSI JSON 文件")
    parser.add_argument("--snapshot-second", type=float, default=None,
                        help="回合内切面时刻（秒，GSI 载荷未提供时使用）")
    parser.add_argument("--time-remaining", type=float, default=None,
                        help="当前有效倒计时（秒，GSI 载荷未提供时使用）")
    parser.add_argument("--bomb-countdown", type=float, default=None,
                        help="炸弹爆炸剩余时间（秒，GSI 载荷未提供时使用）")
    args = parser.parse_args(argv)

    if not args.json and not args.gsi:
        parser.error("必须提供 --json 或 --gsi")
    pred = LivePredictor(args.model)
    start = time.perf_counter()
    if args.gsi:
        p_ct = pred.predict_gsi(_load_payload(args.gsi),
                                snapshot_second=args.snapshot_second,
                                time_remaining_sec=args.time_remaining,
                                bomb_countdown_sec=args.bomb_countdown)
    else:
        p_ct = pred.predict_json(open(args.json, encoding="utf-8").read())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(json.dumps({"p_ct_win": round(p_ct, 6),
                      "p_t_win": round(1.0 - p_ct, 6),
                      "latency_ms": round(elapsed_ms, 3)},
                     ensure_ascii=False))
    return p_ct


if __name__ == "__main__":
    main()
