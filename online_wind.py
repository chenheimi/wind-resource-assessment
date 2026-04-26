# online_wind.py - 在线风资源数据获取模块（完整版）
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from scipy import stats
from datetime import date
import warnings
warnings.filterwarnings('ignore')

from app_part2 import render_tab5
from utils import basic_statistics, TURBINE_MODELS
from wind_grade import show_wind_grade_page
from turbine_design import show_turbine_design_page

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def fetch_wind_data(lat, lon, start_date, end_date, height):
    """从Open-Meteo免费API获取风速数据"""
    height_params = {
        10:  "windspeed_10m,winddirection_10m",
        100: "windspeed_100m,winddirection_100m",
    }

    if height in [10, 100]:
        param = height_params[height]
    else:
        param = "windspeed_10m,winddirection_10m,windspeed_100m,winddirection_100m"

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={param}"
        f"&wind_speed_unit=ms"
        f"&timezone=auto"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        st.error("⏱️ 请求超时，请检查网络连接后重试")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 网络请求失败：{e}")
        return None

    hourly = data.get("hourly", {})
    df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})

    if height == 10:
        df["wind_speed"] = hourly["windspeed_10m"]
        df["wind_direction"] = hourly["winddirection_10m"]
    elif height == 100:
        df["wind_speed"] = hourly["windspeed_100m"]
        df["wind_direction"] = hourly["winddirection_100m"]
    else:
        ws10 = np.array(hourly["windspeed_10m"], dtype=float)
        ws100 = np.array(hourly["windspeed_100m"], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = np.where(
                ws10 > 0,
                np.log(ws100 / np.where(ws10 > 0, ws10, np.nan)) / np.log(100 / 10),
                0.143
            )
        alpha = np.clip(np.nan_to_num(alpha, nan=0.143), 0.05, 0.5)
        ws_h = ws10 * (height / 10) ** alpha
        df["wind_speed"] = ws_h
        df["wind_direction"] = hourly["winddirection_10m"]

    df = df.dropna(subset=["wind_speed", "wind_direction"])
    df["wind_speed"] = df["wind_speed"].astype(float)
    df["wind_direction"] = df["wind_direction"].astype(float)
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    return df

def show_online_wind_page():
    st.markdown("## 🌐 在线风资源评估")
    st.info("📡 数据来源：Open-Meteo ERA5 | 无需注册 | 免费使用")

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("纬度", min_value=-90.0, max_value=90.0, value=39.90, step=0.01, key="online_lat")
        height = st.selectbox("评估高度 (m)", [10, 30, 50, 70, 80, 100, 120, 150], index=4, key="online_height")
    with col2:
        lon = st.number_input("经度", min_value=-180.0, max_value=180.0, value=116.40, step=0.01, key="online_lon")
        rho = st.number_input("空气密度 (kg/m³)", value=1.225, step=0.001, format="%.3f", key="online_rho")

    col3, col4, col5 = st.columns(3)
    with col3:
        start_date = st.date_input("开始日期", value=date(2023, 1, 1), max_value=date(2024, 12, 31), key="online_start")
    with col4:
        end_date = st.date_input("结束日期", value=date(2023, 12, 31), max_value=date(2024, 12, 31), key="online_end")
    with col5:
        turbine_name = st.selectbox("机组选型", list(TURBINE_MODELS.keys()), index=0, key="online_turbine_select")
        turbine = TURBINE_MODELS[turbine_name]

    if st.button("🚀 获取数据并生成完整评估报告", type="primary", use_container_width=True, key="online_fetch_btn"):
        if start_date >= end_date:
            st.error("❌ 开始日期必须早于结束日期")
            return

        with st.spinner("🌐 正在获取数据..."):
            df = fetch_wind_data(
                lat, lon,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                height
            )

        if df is None or len(df) == 0:
            st.error("❌ 数据获取失败")
            return

        st.success(f"✅ 成功获取 {len(df):,} 条数据！")

        # ═══════════════════════════════════════════
        # 1. 数据概览
        # ═══════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📊 数据概览")
        stats_dict = basic_statistics(df)
        items = list(stats_dict.items())
        row1 = st.columns(4)
        row2 = st.columns(4)
        for i, (label, value) in enumerate(items):
            (row1 + row2)[i].metric(label, value)

        with st.expander("📄 查看原始数据"):
            st.dataframe(df.head(500), use_container_width=True)

        # ═══════════════════════════════════════════
        # 2. 风速分析（内联实现，避免key冲突）
        # ═══════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 📈 风速分析")

        # 月平均风速
        monthly_avg = df.groupby("month")["wind_speed"].mean()
        month_labels = ["1月", "2月", "3月", "4月", "5月", "6月",
                        "7月", "8月", "9月", "10月", "11月", "12月"]
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=[month_labels[m-1] for m in monthly_avg.index],
            y=monthly_avg.values,
            mode="lines+markers",
            line=dict(color="#1e88e5", width=2.5),
            marker=dict(size=8),
            fill="tozeroy",
            name="月平均风速"
        ))
        fig_monthly.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="月份", yaxis_title="平均风速 (m/s)",
            margin=dict(t=20)
        )
        st.plotly_chart(fig_monthly, use_container_width=True, key="online_monthly_chart")

        # 日变化
        hourly_avg = df.groupby("hour")["wind_speed"].mean()
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(
            x=hourly_avg.index,
            y=hourly_avg.values,
            mode="lines+markers",
            line=dict(color="#ff7043", width=2.5),
            marker=dict(size=6),
            fill="tozeroy",
            name="小时平均风速"
        ))
        fig_hourly.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="小时", yaxis_title="平均风速 (m/s)",
            margin=dict(t=20)
        )
        st.plotly_chart(fig_hourly, use_container_width=True, key="online_hourly_chart")

        # ═══════════════════════════════════════════
        # 3. 发电量估算
        # ═══════════════════════════════════════════
        st.markdown("---")
        st.markdown("## ⚡ 发电量估算")
        render_tab5(df, turbine, turbine_name, key_prefix="online_tab5_")

        # ═══════════════════════════════════════════
        # 4. 风资源等级评定
        # ═══════════════════════════════════════════
        st.markdown("---")
        show_wind_grade_page(df=df, rho=rho, height=height, key_prefix="online_grade_")

        # ═══════════════════════════════════════════
        # 5. 机组设计
        # ═══════════════════════════════════════════
        st.markdown("---")
        show_turbine_design_page(df=df, rho=rho, hub_height=height, key_prefix="online_turbine_")

        st.markdown("---")
        st.success("✅ 在线风资源评估报告生成完成！")





