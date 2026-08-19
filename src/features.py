# -*- coding: utf-8 -*-
'''状态切面 → 定长特征向量（117 维）。

布局：17 维全局/团队聚合 + 10 个玩家块 × 10 维。
玩家槽位顺序：T 方 5 人（SteamID 升序）→ CT 方 5 人（SteamID 升序）。
死亡玩家除 alive 外全部置 0；位置用整场 bbox 归一化。

设计约束：不引入「跨回合关系」特征——Momentum（近 3 回合胜负和）
与当前比分差均不进入特征向量，模型只看本回合当下状态。
'''

import numpy as np

from src.config import (
    MAX_ROUND_SECONDS,
    PLAYER_SLOTS,
    TEAM_CT,
    TEAM_SIZE,
    TEAM_T,
)

# ---------------- 特征布局 ----------------

# 注意：不含分数/比分等跨回合信息（用户要求）——只有本回合内的可观测状态
_TEAM_BLOCK = [
    ("round.map_code", "int"),
    ("round.snapshot_second", "float"),
    ("round.time_remaining_sec", "float"),
    ("round.bomb_countdown_sec", "float"),
    ("team.t_alive", "int"),
    ("team.ct_alive", "int"),
    ("team.t_avg_health", "float"),
    ("team.ct_avg_health", "float"),
    ("team.t_equip_value", "float"),
    ("team.ct_equip_value", "float"),
    ("team.t_has_c4", "int"),
    ("round.bomb_planted_before", "int"),
    ("round.deaths_before", "int"),
    ("team.t_avg_x", "float"),
    ("team.t_avg_y", "float"),
    ("team.ct_avg_x", "float"),
    ("team.ct_avg_y", "float"),
]

_PLAYER_BLOCK_NAMES = ["alive", "health", "armor", "helmet",
                       "weapon_class", "has_c4", "flash", "x", "y", "z"]


def _build_layout():
    layout = list(_TEAM_BLOCK)
    for i in range(PLAYER_SLOTS):
        layout.extend((f"p{i}.{n}", "float") for n in _PLAYER_BLOCK_NAMES)
    return layout


LAYOUT = _build_layout()
FEATURE_NAMES = [name for name, _ in LAYOUT]
FEATURE_DIM = len(FEATURE_NAMES)
_NAME_TO_INDEX = {name: i for i, (name, _) in enumerate(LAYOUT)}

# 玩家块起点偏移
_PLAYER_BLOCK_OFFSET = len(_TEAM_BLOCK)


def feature_index(name):
    '''按特征名取索引。'''
    try:
        return _NAME_TO_INDEX[name]
    except KeyError:
        raise KeyError(f"未知特征名: {name}") from None


# ---------------- 武器类别 ----------------

WEAPON_CLASS_NONE = 0
WEAPON_CLASS_PISTOL = 1
WEAPON_CLASS_SMG = 2
WEAPON_CLASS_SHOTGUN = 3
WEAPON_CLASS_HEAVY = 4
WEAPON_CLASS_RIFLE = 5
WEAPON_CLASS_AWP = 6
WEAPON_CLASS_SNIPER = 7

_PISTOLS = {"glock", "hkp2000", "usp_silencer", "p250", "deagle", "cz75a",
            "tec9", "fiveseven", "elite", "revolver", "dualberettas", "p2000",
            "usp"}
_SMGS = {"mac10", "mp9", "mp7", "mp5sd", "ump45", "bizon", "p90"}
_SHOTGUNS = {"x4", "sawedoff", "nova", "mag7"}
_HEAVIES = {"m249", "negev"}
_RIFLES = {"ak47", "m4a1", "m4a1_silencer", "galilar", "famas", "sg556", "aug",
           "m4a4"}
_AWPS = {"awp"}
_SNIPERS = {"ssg08", "scar20", "g3sg1"}

# demoparser2 / 部分 GSI 使用 CS2 显示名（空格、连字符、缩写）→ 内部规范名
_WEAPON_ALIASES = {
    "galil ar": "galilar", "ak-47": "ak47", "m4a1-s": "m4a1_silencer",
    "m4a1s": "m4a1_silencer", "m4a4": "m4a4", "famas": "famas",
    "sg 553": "sg556", "aug": "aug", "awp": "awp",
    "ssg 08": "ssg08", "scar-20": "scar20", "g3sg1": "g3sg1",
    "glock-18": "glock", "usp-s": "usp_silencer", "usps": "usp_silencer",
    "p2000": "p2000", "p250": "p250", "desert eagle": "deagle",
    "r8 revolver": "revolver", "cz75-auto": "cz75a",
    "tec-9": "tec9", "five-seven": "fiveseven",
    "dual berettas": "dualberettas",
    "mac-10": "mac10", "mp9": "mp9", "mp7": "mp7",
    "mp5-sd": "mp5sd", "ump-45": "ump45", "pp-bizon": "bizon", "p90": "p90",
    "xm1014": "x4", "sawed-off": "sawedoff", "nova": "nova",
    "mag-7": "mag7", "m249": "m249", "negev": "negev",
}


def weapon_class_of(weapon_name):
    '''主武器名 → 类别序数；兼容 demoparser2 / GSI 前缀与 CS2 显示名。'''
    name = (weapon_name or "").lower().strip()
    for prefix in ("weapon_", "item_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = _WEAPON_ALIASES.get(name, name)
    if name in _PISTOLS:
        return WEAPON_CLASS_PISTOL
    if name in _SMGS:
        return WEAPON_CLASS_SMG
    if name in _SHOTGUNS:
        return WEAPON_CLASS_SHOTGUN
    if name in _HEAVIES:
        return WEAPON_CLASS_HEAVY
    if name in _RIFLES:
        return WEAPON_CLASS_RIFLE
    if name in _AWPS:
        return WEAPON_CLASS_AWP
    if name in _SNIPERS:
        return WEAPON_CLASS_SNIPER
    return WEAPON_CLASS_NONE


# 装备经济价值（近似：主武器类别价值 + 护甲 650 + 头盔 350）
EQUIP_VALUES = {
    WEAPON_CLASS_PISTOL: 400, WEAPON_CLASS_SMG: 1350, WEAPON_CLASS_SHOTGUN: 1500,
    WEAPON_CLASS_HEAVY: 5500, WEAPON_CLASS_RIFLE: 3000, WEAPON_CLASS_AWP: 4750,
    WEAPON_CLASS_SNIPER: 2750,
}
ARMOR_COST = 650
HELMET_COST = 350

# 地图编码
MAP_CODES = {
    "de_dust2": 0, "de_mirage": 1, "de_inferno": 2, "de_nuke": 3,
    "de_overpass": 4, "de_train": 5, "de_vertigo": 6, "de_ancient": 7,
    "de_anubis": 8, "de_cache": 9,
}
UNKNOWN_MAP_CODE = 99

# 坐标归一化兜底半宽（无 bbox 时）
_FALLBACK_SCALE = 2048.0


def _pick_coords(snapshot):
    coords = snapshot.get("coords") or {}
    min_x = coords.get("min_x", -_FALLBACK_SCALE)
    max_x = coords.get("max_x", _FALLBACK_SCALE)
    min_y = coords.get("min_y", -_FALLBACK_SCALE)
    max_y = coords.get("max_y", _FALLBACK_SCALE)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    sx = max(1e-6, (max_x - min_x) / 2.0)
    sy = max(1e-6, (max_y - min_y) / 2.0)
    return cx, cy, sx, sy


def _norm_position(player, alive, cx, cy, sx, sy):
    '''把玩家位置归一化；死亡玩家返回 (0,0)。'''
    if not alive:
        return 0.0, 0.0
    return (player.get("x", 0.0) - cx) / sx, (player.get("y", 0.0) - cy) / sy


def _aggregate(players, cx, cy, sx, sy):
    '''团队聚合：存活数 / 金钱 / 平均血量 / 装备价值 / C4 / 质心。'''
    n = t_money = avg_hp = equip = 0
    has_c4 = 0
    sx_avg = sy_avg = 0.0
    for p in players:
        alive = bool(p["alive"])
        if alive:
            n += 1
            t_money += float(p["money"])
            avg_hp += float(p["health"])
            equip += (EQUIP_VALUES.get(int(p["weapon_class"]), 0)
                      + (ARMOR_COST if float(p["armor"]) > 0 else 0.0)
                      + (HELMET_COST if int(p["helmet"]) else 0.0))
            px, py = _norm_position(p, alive, cx, cy, sx, sy)
            sx_avg += px
            sy_avg += py
        if int(p.get("has_c4", 0)):
            has_c4 = 1
    avg_hp = avg_hp / n if n else 0.0
    sx_avg = sx_avg / n if n else 0.0
    sy_avg = sy_avg / n if n else 0.0
    return {"alive": n, "money": t_money, "avg_hp": avg_hp, "equip": equip,
            "has_c4": has_c4, "avg_x": sx_avg, "avg_y": sy_avg}


def _empty_player(team_num, slot):
    return {"steamid": 0, "team_num": team_num, "alive": 0, "health": 0.0,
            "armor": 0.0, "helmet": 0, "money": 0.0, "weapon_class": 0,
            "has_c4": 0, "flash": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}


def build_feature_vector(snapshot):
    '''把一条状态切面编码为 132 维特征向量。'''
    players = snapshot.get("players") or []
    t_players = sorted((p for p in players if p["team_num"] == TEAM_T),
                       key=lambda p: p["steamid"])
    ct_players = sorted((p for p in players if p["team_num"] == TEAM_CT),
                        key=lambda p: p["steamid"])
    slots = t_players[:TEAM_SIZE] + ct_players[:TEAM_SIZE]
    # 若某方人数不足（异常 demo），用空玩家补齐槽位
    if len(slots) < PLAYER_SLOTS:
        slots = slots + [_empty_player(3, i) for i in range(PLAYER_SLOTS - len(slots))]

    cx, cy, sx, sy = _pick_coords(snapshot)
    agg_t = _aggregate(t_players, cx, cy, sx, sy)
    agg_ct = _aggregate(ct_players, cx, cy, sx, sy)

    if snapshot.get("time_remaining_sec") is not None:
        time_remaining = float(snapshot["time_remaining_sec"])
    elif ("round_end_tick" in snapshot and "snapshot_tick" in snapshot):
        tick_rate = float(snapshot.get("tick_rate", 128))
        time_remaining = (float(snapshot["round_end_tick"])
                          - float(snapshot["snapshot_tick"])) / tick_rate
    else:
        # GSI 实时切面可能没有 tick 信息：按 CS2 常规回合时钟兜底
        time_remaining = max(0.0, 115.0 - float(snapshot.get("snapshot_second", 0.0)))
    # 异常 demo 的回合边界可能跨越暂停/半场：统一夹紧到物理上限
    time_remaining = min(max(0.0, time_remaining), float(MAX_ROUND_SECONDS))
    # 炸弹爆炸剩余时间独立编码：未埋包/已爆炸时为 0，上限 40s
    bomb_countdown = min(max(0.0, float(snapshot.get("bomb_countdown_sec", 0.0))), 40.0)

    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx = feature_index

    # 回合上下文 + 团队聚合
    vec[idx("round.map_code")] = MAP_CODES.get(snapshot.get("map_name"), UNKNOWN_MAP_CODE)
    # 注：V2 约束——不使用比分差 / Momentum 等跨回合特征；同时移除 round.half / round.num
    vec[idx("round.snapshot_second")] = float(snapshot.get("snapshot_second", 0.0))
    vec[idx("round.time_remaining_sec")] = time_remaining
    vec[idx("round.bomb_countdown_sec")] = bomb_countdown
    vec[idx("team.t_alive")] = agg_t["alive"]
    vec[idx("team.ct_alive")] = agg_ct["alive"]
    vec[idx("team.t_avg_health")] = agg_t["avg_hp"]
    vec[idx("team.ct_avg_health")] = agg_ct["avg_hp"]
    vec[idx("team.t_equip_value")] = agg_t["equip"]
    vec[idx("team.ct_equip_value")] = agg_ct["equip"]
    vec[idx("team.t_has_c4")] = agg_t["has_c4"]
    vec[idx("round.bomb_planted_before")] = int(snapshot.get("bomb_planted_before", 0))
    vec[idx("round.deaths_before")] = int(snapshot.get("deaths_before", 0))
    vec[idx("team.t_avg_x")] = agg_t["avg_x"]
    vec[idx("team.t_avg_y")] = agg_t["avg_y"]
    vec[idx("team.ct_avg_x")] = agg_ct["avg_x"]
    vec[idx("team.ct_avg_y")] = agg_ct["avg_y"]

    # 玩家块
    for slot_i, p in enumerate(slots[:PLAYER_SLOTS]):
        base = _PLAYER_BLOCK_OFFSET + slot_i * len(_PLAYER_BLOCK_NAMES)
        alive = int(p.get("alive", 0))
        vec[base + 0] = alive
        if not alive:
            continue   # 死亡玩家整块置 0
        vec[base + 1] = float(p.get("health", 0.0))
        vec[base + 2] = float(p.get("armor", 0.0))
        vec[base + 3] = int(p.get("helmet", 0))
        vec[base + 4] = int(p.get("weapon_class", 0))
        vec[base + 5] = int(p.get("has_c4", 0))
        vec[base + 6] = float(p.get("flash", 0.0))
        px, py = _norm_position(p, True, cx, cy, sx, sy)
        vec[base + 7] = px
        vec[base + 8] = py
        vec[base + 9] = float(p.get("z", 0.0))
    return vec
