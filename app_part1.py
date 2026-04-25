import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    generate_wind_data, basic_statistics,
    power_curve, TURBINE_MODELS, DIRECTION_LABELS,
)


def setup_page():
    st.set_page_config(
        page_title="风资源评估系统",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.4rem; font-weight: 800;
            background: linear-gradient(135deg, #1e88e5, #26c6da);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .subtitle { color: #90a4ae; font-size: 1rem; margin-top: 0; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/wind.png", width=64)
        st.markdown("## ⚙️ 参数配置")
        st.markdown("### 📂 数据来源")
        data_source = st.radio("选择数据来源", ["🎲 模拟数据", "📁 上传 CSV"])

        df = None

        if data_source == "🎲 模拟数据":
            st.markdown("### 🌀 威布尔参数")
            k_input = st.slider("形状参数 k", 1.0, 4.0, 2.0, 0.1)
            c_input = st.slider("尺度参数 c (m/s)", 3.0, 15.0, 8.0, 0.5)
            n_days  = st.slider("模拟天数", 30, 365, 365, 30)
            seed    = st.number_input("随机种子", value=42, step=1)

            if st.button("🔄 生成数据", type="primary", use_container_width=True):
                df = generate_wind_data(n_days, k_input, c_input, int(seed))
                st.session_state["df"] = df
                st.success(f"✅ 已生成 {len(df):,} 条数据")
        else:
            uploaded = st.file_uploader(
                "上传 CSV（需含 wind_speed / wind_direction 列）",
                type=["csv"],
            )
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    required = {"wind_speed", "wind_direction"}
                    if not required.issubset(df.columns):
                        st.error(f"❌ 缺少列：{required - set(df.columns)}")
                        df = None
                    else:
                        if "timestamp" not in df.columns:
                            df["timestamp"] = pd.date_range(
                                "2025-01-01", periods=len(df), freq="h")
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df["month"] = df["timestamp"].dt.month
                        df["hour"]  = df["timestamp"].dt.hour
                        st.session_state["df"] = df
                        st.success(f"✅ 上传成功，共 {len(df):,} 条记录")
                except Exception as e:
                    st.error(f"解析失败：{e}")

        if df is None and "df" in st.session_state:
            df = st.session_state["df"]

        st.markdown("---")
        st.markdown("### 🏭 风机选型")
        turbine_name = st.selectbox("选择风机型号", list(TURBINE_MODELS.keys()))
        turbine = TURBINE_MODELS[turbine_name]
        st.caption(
            f"额定功率：{turbine['rated_power']} kW | "
            f"切入：{turbine['cut_in']} m/s | "
            f"切出：{turbine['cut_out']} m/s | "
            f"额定风速：{turbine['rated_speed']} m/s"
        )
        st.markdown("---")
        st.markdown("### 📍 站点信息")
        site_name  = st.text_input("站点名称", "示例测风塔")
        hub_height = st.number_input("轮毂高度 (m)", 10, 200, 80, 10)
        rho        = st.number_input("空气密度 (kg/m³)", 1.0, 1.3, 1.225, 0.005)
        st.markdown("---")
        st.caption("🎓 风资源评估演示系统 v1.0")

    return df, turbine, turbine_name, site_name, hub_height, rho


def render_tab1(df):
    st.subheader("📊 基础统计指标")
    stats_dict = basic_statistics(df)
    items      = list(stats_dict.items())
    row1       = st.columns(4)
    row2       = st.columns(4)
    for i, (label, value) in enumerate(items):
        (row1 + row2)[i].metric(label, value)

    st.divider()
    st.subheader("⏱️ 风速时序（前 30 天）")
    t0      = df["timestamp"].iloc[0]
    preview = df[df["timestamp"] < t0 + pd.Timedelta(days=30)]

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=preview["timestamp"], y=preview["wind_speed"],
        mode="lines", line=dict(color="#1e88e5", width=1),
        fill="tozeroy", fillcolor="rgba(30,136,229,0.15)", name="风速",
    ))
    fig_ts.add_hline(
        y=stats_dict["平均风速 (m/s)"], line_dash="dash", line_color="#ff7043",
        annotation_text=f"均值 {stats_dict['平均风速 (m/s)']} m/s",
        annotation_font_color="#ff7043",
    )
    fig_ts.update_layout(
        template="plotly_dark", height=300,
        xaxis_title="时间", yaxis_title="风速 (m/s)",
        hovermode="x unified", margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    with st.expander("🗂️ 查看原始数据（前 200 行）"):
        st.dataframe(df.head(200), use_container_width=True)

    st.download_button(
        label="⬇️ 下载完整数据 CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="wind_data.csv",
        mime="text/csv",
    )


def render_tab2(df, turbine):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 风速频率分布")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["wind_speed"], nbinsx=40,
            histnorm="probability density",
            marker_color="#1e88e5", opacity=0.8, name="实测频率",
        ))
        fig_hist.update_layout(
            template="plotly_dark", height=360,
            xaxis_title="风速 (m/s)", yaxis_title="概率密度",
            bargap=0.05, margin=dict(t=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("📅 各月平均风速")
        monthly = df.groupby("month")["wind_speed"].agg(["mean","std"]).reset_index()
        mlabels = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["label"] = monthly["month"].apply(lambda m: mlabels[m - 1])

        fig_month = go.Figure()
        fig_month.add_trace(go.Bar(
            x=monthly["label"], y=monthly["mean"].round(2),
            error_y=dict(type="data", array=monthly["std"].round(2)),
            marker=dict(color=monthly["mean"], colorscale="Blues", showscale=False),
            text=monthly["mean"].round(2), textposition="outside",
            name="月均风速",
        ))
        fig_month.update_layout(
            template="plotly_dark", height=360,
            xaxis_title="月份", yaxis_title="风速 (m/s)", margin=dict(t=20),
        )
        st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("🕐 风速日变化规律（均值 ± 四分位区间）")
    hourly = (df.groupby("hour")["wind_speed"]
              .agg(mean="mean",
                   p25=lambda x: x.quantile(0.25),
                   p75=lambda x: x.quantile(0.75))
              .reset_index())

    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=list(hourly["hour"]) + list(hourly["hour"])[::-1],
        y=list(hourly["p75"]) + list(hourly["p25"])[::-1],
        fill="toself", fillcolor="rgba(30,136,229,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name="25%–75% 区间",
    ))
    fig_h.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean"].round(2),
        mode="lines+markers",
        line=dict(color="#1e88e5", width=2.5), marker=dict(size=6), name="均值",
    ))
    fig_h.update_layout(
        template="plotly_dark", height=320,
        xaxis=dict(title="小时", tickvals=list(range(0, 24, 3))),
        yaxis_title="风速 (m/s)", hovermode="x unified", margin=dict(t=20),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("📈 风速超越概率曲线")
    sorted_ws  = np.sort(df["wind_speed"].values)[::-1]
    exceedance = np.arange(1, len(sorted_ws) + 1) / len(sorted_ws) * 100

    fig_exc = go.Figure()
    fig_exc.add_trace(go.Scatter(
        x=sorted_ws, y=exceedance, mode="lines",
        line=dict(color="#26c6da", width=2),
        fill="tozeroy", fillcolor="rgba(38,198,218,0.1)", name="超越概率",
    ))
    fig_exc.add_vline(
        x=turbine["cut_in"], line_dash="dot", line_color="#ffa726",
        annotation_text=f"切入 {turbine['cut_in']} m/s",
        annotation_font_color="#ffa726",
    )
    fig_exc.add_vline(
        x=turbine["cut_out"], line_dash="dot", line_color="#ef5350",
        annotation_text=f"切出 {turbine['cut_out']} m/s",
        annotation_font_color="#ef5350",
    )
    fig_exc.update_layout(
        template="plotly_dark", height=320,
        xaxis_title="风速 (m/s)", yaxis_title="超越概率 (%)", margin=dict(t=20),
    )
    st.plotly_chart(fig_exc, use_container_width=True)



