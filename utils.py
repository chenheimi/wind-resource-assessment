import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

# ── 方向标签 ──────────────────────────────────
DIRECTION_LABELS = [
    "N","NNE","NE","ENE",
    "E","ESE","SE","SSE",
    "S","SSW","SW","WSW",
    "W","WNW","NW","NNW",
]

# ── 风机型号库 ─────────────────────────────────
TURBINE_MODELS = {
    "WTG-1500 (1.5MW)": {
        "rated_power":  1500,
        "cut_in":       3,
        "cut_out":      25,
        "rated_speed":  12,
    },
    "WTG-2000 (2.0MW)": {
        "rated_power":  2000,
        "cut_in":       3,
        "cut_out":      25,
        "rated_speed":  13,
    },
    "WTG-3000 (3.0MW)": {
        "rated_power":  3000,
        "cut_in":       3,
        "cut_out":      25,
        "rated_speed":  14,
    },
    "WTG-5000 (5.0MW)": {
        "rated_power":  5000,
        "cut_in":       3,
        "cut_out":      25,
        "rated_speed":  13,
    },
}


def generate_wind_data(n_days: int, k: float, c: float, seed: int) -> pd.DataFrame:
    """生成模拟逐小时风速/风向数据"""
    rng   = np.random.default_rng(seed)
    n     = n_days * 24
    speed = c * rng.weibull(k, n)               # 威布尔分布风速
    direc = rng.uniform(0, 360, n)              # 均匀风向

    ts = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp":      ts,
        "wind_speed":     np.round(speed, 2),
        "wind_direction": np.round(direc, 1),
    })
    df["month"] = df["timestamp"].dt.month
    df["hour"]  = df["timestamp"].dt.hour
    return df


def basic_statistics(df: pd.DataFrame) -> dict:
    """计算基础统计指标"""
    ws = df["wind_speed"]
    return {
        "平均风速 (m/s)":    round(ws.mean(), 2),
        "中位风速 (m/s)":    round(ws.median(), 2),
        "最大风速 (m/s)":    round(ws.max(), 2),
        "最小风速 (m/s)":    round(ws.min(), 2),
        "标准差 (m/s)":      round(ws.std(), 2),
        "湍流强度 (%)":      round(ws.std() / ws.mean() * 100, 1),
        "有效风时率 (%)":    round((ws >= 3).sum() / len(ws) * 100, 1),
        "数据总量 (条)":     len(ws),
    }


def fit_weibull(ws: np.ndarray):
    """MLE 拟合威布尔参数，返回 (k, c)"""
    ws_pos = ws[ws > 0]
    shape, _, scale = weibull_min.fit(ws_pos, floc=0)
    return round(shape, 3), round(scale, 3)


def weibull_pdf(x: np.ndarray, k: float, c: float) -> np.ndarray:
    """威布尔概率密度函数"""
    return (k / c) * (x / c) ** (k - 1) * np.exp(-(x / c) ** k)


def weibull_energy_density(k: float, c: float, rho: float = 1.225) -> float:
    """理论风能密度 W/m²"""
    import math
    return 0.5 * rho * c**3 * math.gamma(1 + 3 / k)


def wind_rose_data(df: pd.DataFrame) -> pd.DataFrame:
    """统计各方向各风速段频率"""
    sector_width = 22.5
    bins   = [0, 3, 5, 7, 10, 15, np.inf]
    labels = ["0–3","3–5","5–7","7–10","10–15",">15"]

    tmp = df.copy()
    tmp["sector"] = (
        ((tmp["wind_direction"] + sector_width / 2) % 360) // sector_width
    ).astype(int) % 16
    tmp["angle"]     = tmp["sector"] * sector_width
    tmp["speed_bin"] = pd.cut(tmp["wind_speed"], bins=bins, labels=labels, right=False)

    rose = (
        tmp.groupby(["angle","speed_bin"], observed=True)
        .size()
        .reset_index(name="count")
    )
    rose["frequency"] = rose["count"] / len(df) * 100
    return rose


def power_curve(ws: np.ndarray, turbine: dict) -> np.ndarray:
    """根据功率曲线计算输出功率 (kW)"""
    cut_in  = turbine["cut_in"]
    cut_out = turbine["cut_out"]
    rated_s = turbine["rated_speed"]
    rated_p = turbine["rated_power"]

    power = np.zeros_like(ws, dtype=float)
    mask_ramp  = (ws >= cut_in) & (ws < rated_s)
    mask_rated = (ws >= rated_s) & (ws <= cut_out)

    power[mask_ramp]  = rated_p * ((ws[mask_ramp] - cut_in) /
                                    (rated_s - cut_in)) ** 3
    power[mask_rated] = rated_p
    return power


def annual_energy(df: pd.DataFrame, turbine: dict) -> dict:
    """估算年发电量"""
    p      = power_curve(df["wind_speed"].values, turbine)
    aep_kw = p.sum()                        # kWh（每条记录 = 1 小时）
    aep_mw = aep_kw / 1000                  # MWh

    hours        = len(df)
    rated_power  = turbine["rated_power"]
    cf           = aep_kw / (rated_power * hours) * 100
    full_hours   = aep_kw / rated_power

    return {
        "年发电量 AEP (MWh)": round(aep_mw, 1),
        "容量因子 CF (%)":    round(cf, 2),
        "满负荷小时数 (h)":   round(full_hours, 0),
    }


