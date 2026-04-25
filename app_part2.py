import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    fit_weibull, weibull_pdf, weibull_energy_density,
    wind_rose_data, power_curve, annual_energy,
    DIRECTION_LABELS,
)


# ══════════════════════════════════════════════
#  Tab 3 · 风玫瑰图
# ══════════════════════════════════════════════
def render_tab3(df):
    st.subheader("🌹 风玫瑰图（频率 × 风速区间）")
    rose = wind_rose_data(df)
    speed_colors = {
        "0–3":   "#bbdefb",
        "3–5":   "#64b5f6",
        "5–7":   "#1e88e5",
        "7–10":  "#1565c0",
        "10–15": "#0d47a1",
        ">15":   "#01579b",
    }

    fig_rose = go.Figure()
    for speed_bin, color in speed_colors.items():
        subset = rose[rose["speed_bin"] == speed_bin]
        if subset.empty:
            continue
        full_angles = pd.DataFrame({
            "angle":     np.arange(16) * 22.5,
            "direction": DIRECTION_LABELS,
        })
        full_angles = full_angles.merge(
            subset[["angle", "frequency"]], on="angle", how="left"
        ).fillna(0)
        fig_rose.add_trace(go.Barpolar(
            r=full_angles["frequency"],
            theta=full_angles["angle"],
            name=f"{speed_bin} m/s",
            marker_color=color,
            opacity=0.88,
        ))

    fig_rose.update_layout(
        template="plotly_dark", height=540,
        polar=dict(
            angularaxis=dict(
                tickvals=[i * 22.5 for i in range(16)],
                ticktext=DIRECTION_LABELS,
                direction="clockwise",
                rotation=90,
            ),
            radialaxis=dict(
                ticksuffix="%",
                showticklabels=True,
                gridcolor="rgba(255,255,255,0.15)",
            ),
        ),
        legend=dict(
            title="风速区间", orientation="h",
            yanchor="bottom", y=-0.18,
            xanchor="center", x=0.5,
        ),
        margin=dict(t=40, b=100),
    )
    st.plotly_chart(fig_rose, use_container_width=True)

    with st.expander("📋 各方向风频详细统计"):
        sw  = 22.5
        tmp = df.copy()
        tmp["sector"] = (
            ((tmp["wind_direction"] + sw / 2) % 360) // sw
        ).astype(int) % 16

        tbl = (
            tmp.groupby("sector")["wind_speed"]
            .agg(样本数="count", 平均风速="mean", 最大风速="max")
            .reset_index()
        )
        tbl["方向"]          = tbl["sector"].apply(lambda s: DIRECTION_LABELS[s])
        tbl["频率(%)"]       = (tbl["样本数"] / len(df) * 100).round(2)
        tbl["平均风速(m/s)"] = tbl["平均风速"].round(2)
        tbl["最大风速(m/s)"] = tbl["最大风速"].round(2)
        st.dataframe(
            tbl[["方向", "频率(%)", "平均风速(m/s)", "最大风速(m/s)", "样本数"]],
            use_container_width=True,
        )


# ══════════════════════════════════════════════
#  Tab 4 · 威布尔拟合
# ══════════════════════════════════════════════
def render_tab4(df, rho):
    st.subheader("📐 威布尔分布拟合分析")

    ws_data      = df["wind_speed"].values
    k_fit, c_fit = fit_weibull(ws_data)
    wpd          = weibull_energy_density(k_fit, c_fit, rho)
    mean_t       = c_fit * math.gamma(1 + 1 / k_fit)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("形状参数 k",          k_fit)
    c2.metric("尺度参数 c (m/s)",    c_fit)
    c3.metric("理论平均风速 (m/s)",  round(mean_t, 2))
    c4.metric("理论风能密度 (W/m²)", round(wpd, 1))
    st.divider()

    x_range  = np.linspace(0.01, df["wind_speed"].max() + 3, 400)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**概率密度函数（PDF）拟合对比**")
        fig_wb = go.Figure()
        fig_wb.add_trace(go.Histogram(
            x=ws_data, nbinsx=40,
            histnorm="probability density",
            marker_color="#1e88e5", opacity=0.55,
            name="实测频率",
        ))
        fig_wb.add_trace(go.Scatter(
            x=x_range,
            y=weibull_pdf(x_range, k_fit, c_fit),
            mode="lines",
            line=dict(color="#ff7043", width=2.5),
            name=f"威布尔拟合 (k={k_fit}, c={c_fit})",
        ))
        fig_wb.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="风速 (m/s)", yaxis_title="概率密度",
            legend=dict(x=0.4, y=0.95),
            margin=dict(t=20),
        )
        st.plotly_chart(fig_wb, use_container_width=True)

    with col_r:
        st.markdown("**累积分布函数（CDF）拟合对比**")
        sorted_ws = np.sort(ws_data)
        ecdf      = np.arange(1, len(sorted_ws) + 1) / len(sorted_ws)
        cdf_t     = 1 - np.exp(-(x_range / c_fit) ** k_fit)

        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=sorted_ws, y=ecdf,
            mode="lines",
            line=dict(color="#1e88e5", width=1.5),
            name="实测 ECDF",
        ))
        fig_cdf.add_trace(go.Scatter(
            x=x_range, y=cdf_t,
            mode="lines",
            line=dict(color="#ff7043", width=2.5, dash="dash"),
            name="威布尔 CDF",
        ))
        fig_cdf.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="风速 (m/s)", yaxis_title="累积概率",
            legend=dict(x=0.55, y=0.15),
            margin=dict(t=20),
        )
        st.plotly_chart(fig_cdf, use_container_width=True)

    st.divider()
    st.subheader("🔬 参数敏感性分析")
    s1, s2 = st.columns(2)

    with s1:
        st.markdown("**改变形状参数 k（固定 c = 8 m/s）**")
        fig_k    = go.Figure()
        colors_k = ["#90caf9", "#64b5f6", "#1e88e5", "#1565c0", "#0d47a1"]
        for i, kv in enumerate([1.2, 1.6, 2.0, 2.5, 3.0]):
            fig_k.add_trace(go.Scatter(
                x=x_range, y=weibull_pdf(x_range, kv, 8.0),
                mode="lines",
                line=dict(color=colors_k[i], width=2),
                name=f"k = {kv}",
            ))
        fig_k.update_layout(
            template="plotly_dark", height=310,
            xaxis_title="风速 (m/s)", yaxis_title="概率密度",
            margin=dict(t=20),
        )
        st.plotly_chart(fig_k, use_container_width=True)

    with s2:
        st.markdown("**改变尺度参数 c（固定 k = 2）**")
        fig_c    = go.Figure()
        colors_c = ["#a5d6a7", "#66bb6a", "#43a047", "#2e7d32", "#1b5e20"]
        for i, cv in enumerate([5.0, 6.5, 8.0, 10.0, 12.0]):
            fig_c.add_trace(go.Scatter(
                x=x_range, y=weibull_pdf(x_range, 2.0, cv),
                mode="lines",
                line=dict(color=colors_c[i], width=2),
                name=f"c = {cv} m/s",
            ))
        fig_c.update_layout(
            template="plotly_dark", height=310,
            xaxis_title="风速 (m/s)", yaxis_title="概率密度",
            margin=dict(t=20),
        )
        st.plotly_chart(fig_c, use_container_width=True)


# ══════════════════════════════════════════════
#  Tab 5 · 发电量估算
# ══════════════════════════════════════════════
def render_tab5(df, turbine, turbine_name, key_prefix=""):
    st.subheader("⚡ 年发电量估算")

    ws_curve = np.linspace(0, 32, 320)
    pc_vals  = power_curve(ws_curve, turbine)

    st.markdown(f"**当前选型：{turbine_name}**")
    fig_pc = go.Figure()
    fig_pc.add_trace(go.Scatter(
        x=ws_curve, y=pc_vals,
        mode="lines",
        line=dict(color="#66bb6a", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(102,187,106,0.15)",
        name="输出功率 (kW)",
    ))
    fig_pc.add_vline(
        x=turbine["cut_in"], line_dash="dot", line_color="#ffa726",
        annotation_text=f"切入 {turbine['cut_in']} m/s",
        annotation_font_color="#ffa726",
    )
    fig_pc.add_vline(
        x=turbine["rated_speed"], line_dash="dot", line_color="#26c6da",
        annotation_text=f"额定 {turbine['rated_speed']} m/s",
        annotation_font_color="#26c6da",
    )
    fig_pc.add_vline(
        x=turbine["cut_out"], line_dash="dot", line_color="#ef5350",
        annotation_text=f"切出 {turbine['cut_out']} m/s",
        annotation_font_color="#ef5350",
    )
    fig_pc.add_hline(
        y=turbine["rated_power"],
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text=f"额定 {turbine['rated_power']} kW",
    )
    fig_pc.update_layout(
        template="plotly_dark", height=340,
        xaxis_title="风速 (m/s)", yaxis_title="输出功率 (kW)",
        margin=dict(t=20),
    )
    st.plotly_chart(fig_pc, use_container_width=True, key=f"{key_prefix}tab5_pc")

    st.divider()

    # 年发电量指标
    aep_result = annual_energy(df, turbine)
    st.subheader("📋 年发电量计算结果")
    m1, m2, m3 = st.columns(3)
    m1.metric("年发电量 AEP",  f"{aep_result['年发电量 AEP (MWh)']:,.1f} MWh")
    m2.metric("容量因子 CF",   f"{aep_result['容量因子 CF (%)']:.1f} %")
    m3.metric("满负荷小时数",  f"{aep_result['满负荷小时数 (h)']:,.0f} h")

    st.divider()

    # 各月发电量
    st.subheader("📅 各月发电量分布")
    monthly_aep = []
    for month, grp in df.groupby("month"):
        p = power_curve(grp["wind_speed"].values, turbine)
        monthly_aep.append({
            "月份":        month,
            "发电量(MWh)": round(p.sum() / 1000, 1),
        })
    df_maep    = pd.DataFrame(monthly_aep)
    mlabels_cn = ["1月","2月","3月","4月","5月","6月",
                  "7月","8月","9月","10月","11月","12月"]
    df_maep["月份标签"] = df_maep["月份"].apply(lambda m: mlabels_cn[m - 1])

    fig_aep = go.Figure()
    fig_aep.add_trace(go.Bar(
        x=df_maep["月份标签"],
        y=df_maep["发电量(MWh)"],
        marker=dict(
            color=df_maep["发电量(MWh)"],
            colorscale="Greens",
            showscale=True,
            colorbar=dict(title="MWh"),
        ),
        text=df_maep["发电量(MWh)"],
        textposition="outside",
        name="月发电量",
    ))
    fig_aep.update_layout(
        template="plotly_dark", height=380,
        xaxis_title="月份", yaxis_title="发电量 (MWh)",
        margin=dict(t=20),
    )
    st.plotly_chart(fig_aep, use_container_width=True, key=f"{key_prefix}tab5_aep")

    st.divider()

    # 风速 vs 功率散点图
    st.subheader("🔵 风速 vs 瞬时功率散点图（随机采样 3000 点）")
    sample       = df.sample(min(3000, len(df)), random_state=42)
    sample_power = power_curve(sample["wind_speed"].values, turbine)

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=sample["wind_speed"], y=sample_power,
        mode="markers",
        marker=dict(
            color=sample_power, colorscale="Greens",
            size=4, opacity=0.6, showscale=False,
        ),
        name="实测点",
    ))
    fig_sc.add_trace(go.Scatter(
        x=ws_curve, y=pc_vals,
        mode="lines",
        line=dict(color="#ff7043", width=2),
        name="理论功率曲线",
    ))
    fig_sc.update_layout(
        template="plotly_dark", height=360,
        xaxis_title="风速 (m/s)", yaxis_title="功率 (kW)",
        margin=dict(t=20),
    )
    st.plotly_chart(fig_sc, use_container_width=True, key=f"{key_prefix}tab5_sc")

    st.divider()

    # 导出报告
    st.subheader("⬇️ 导出发电量报告")
    report_df = df_maep[["月份标签", "发电量(MWh)"]].copy()
    report_df.columns = ["月份", "发电量(MWh)"]
    report_df.loc[len(report_df)] = [
        "全年合计",
        round(report_df["发电量(MWh)"].sum(), 1),
    ]
    st.dataframe(report_df, use_container_width=True)
    st.download_button(
        label="⬇️ 下载发电量报告 CSV",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="aep_report.csv",
        mime="text/csv",
        key=f"{key_prefix}tab5_download",    # ← 本次新增，唯一修改处
    )




