# wind_grade.py - 风资源等级评估模块（GB/T 18710-2002）
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import fit_weibull, weibull_energy_density


# ══════════════════════════════════════════════
#  颜色工具函数
# ══════════════════════════════════════════════
def hex_to_rgba(hex_color, alpha=1.0):
    """将 hex 颜色（#rrggbb）转换为 rgba(r,g,b,a) 字符串"""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ══════════════════════════════════════════════
#  国标风资源等级参数表
#  依据：GB/T 18710-2002《风电场风能资源评估方法》
# ══════════════════════════════════════════════
WIND_GRADE_TABLE = [
    {"等级": "1级", "功率密度下限": 0, "功率密度上限": 100, "参考风速": "< 4.4", "资源描述": "贫乏", "颜色": "#b0bec5"},
    {"等级": "2级", "功率密度下限": 100, "功率密度上限": 150, "参考风速": "4.4–5.1", "资源描述": "较差", "颜色": "#81d4fa"},
    {"等级": "3级", "功率密度下限": 150, "功率密度上限": 200, "参考风速": "5.1–5.6", "资源描述": "可利用", "颜色": "#a5d6a7"},
    {"等级": "4级", "功率密度下限": 200, "功率密度上限": 250, "参考风速": "5.6–6.0", "资源描述": "较丰富", "颜色": "#fff176"},
    {"等级": "5级", "功率密度下限": 250, "功率密度上限": 300, "参考风速": "6.0–6.4", "资源描述": "丰富", "颜色": "#ffb74d"},
    {"等级": "6级", "功率密度下限": 300, "功率密度上限": 400, "参考风速": "6.4–7.0", "资源描述": "很丰富", "颜色": "#f48fb1"},
    {"等级": "7级", "功率密度下限": 400, "功率密度上限": 9999, "参考风速": "> 7.0", "资源描述": "极丰富", "颜色": "#ce93d8"},
]

SUITABILITY_THRESHOLD = 200   # W/m²：国标推荐的适合开发下限


def calc_wind_power_density(ws_array, rho=1.225):
    """计算实测风功率密度 (W/m²)"""
    return 0.5 * rho * np.mean(ws_array ** 3)


def get_wind_grade(wpd):
    """根据风功率密度判定国标等级"""
    for g in WIND_GRADE_TABLE:
        if g["功率密度下限"] <= wpd < g["功率密度上限"]:
            return g
    return WIND_GRADE_TABLE[-1]


def calc_effective_hours(ws_array, v_in=3.0, v_out=25.0):
    """计算有效风速（切入~切出）小时数及占比"""
    mask = (ws_array >= v_in) & (ws_array <= v_out)
    eff_h = int(mask.sum())
    total_h = len(ws_array)
    ratio = eff_h / total_h * 100 if total_h > 0 else 0
    return eff_h, total_h, round(ratio, 2)


def calc_utilization_coeff(ws_array, rho=1.225, v_rated=10.0, v_in=3.0, v_out=25.0):
    """
    风能利用系数（实际可用风能 / 总风能）
    = 有效风速段风能 / 全风速段风能
    """
    total_energy = np.sum(ws_array ** 3)
    if total_energy == 0:
        return 0.0
    mask = (ws_array >= v_in) & (ws_array <= v_out)
    useful_energy = np.sum(ws_array[mask] ** 3)
    return round(useful_energy / total_energy * 100, 2)


def calc_turbulence_intensity(ws_array, bin_size=1.0):
    """
    湍流强度：各风速段（以 1m/s 为组距）的 σ/μ
    返回 DataFrame，便于绘图
    """
    records = []
    bins = np.arange(0, ws_array.max() + bin_size, bin_size)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        seg = ws_array[(ws_array >= lo) & (ws_array < hi)]
        if len(seg) >= 5:
            mu = seg.mean()
            if mu > 0:
                ti = seg.std() / mu
                records.append({"风速区间中值": round((lo + hi) / 2, 1), "湍流强度": round(ti, 4)})
    return pd.DataFrame(records)


def show_wind_grade_page(df=None, rho=1.225, height=70, key_prefix="grade_"):
    """
    风资源等级评估主页面
    df        : 包含 wind_speed 列的 DataFrame（可来自上传或在线获取）
    rho       : 空气密度 kg/m³
    height    : 评估高度 m
    key_prefix: 避免重复 key
    """
    st.markdown("## 🏅 风资源等级评估")
    st.caption("依据：GB/T 18710-2002《风电场风能资源评估方法》")

    if df is None or len(df) == 0:
        st.warning("⚠️ 请先上传风资源数据或通过在线评估获取数据后再使用本模块。")
        return

    ws = df["wind_speed"].dropna().values

    # ══════════════════════════════
    #  评估参数
    # ══════════════════════════════
    st.markdown("### ⚙️ 评估参数")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        rho_input = st.number_input(
            "空气密度 (kg/m³)", value=float(rho),
            min_value=0.9, max_value=1.4, step=0.001,
            format="%.3f", key=f"{key_prefix}rho"
        )
    with col_p2:
        v_in = st.number_input(
            "切入风速 (m/s)", value=3.0,
            min_value=1.0, max_value=6.0, step=0.5,
            key=f"{key_prefix}vin"
        )
    with col_p3:
        v_out = st.number_input(
            "切出风速 (m/s)", value=25.0,
            min_value=15.0, max_value=35.0, step=1.0,
            key=f"{key_prefix}vout"
        )

    st.divider()

    # ══════════════════════════════
    #  核心指标计算
    # ══════════════════════════════
    wpd = calc_wind_power_density(ws, rho_input)
    grade_info = get_wind_grade(wpd)
    eff_h, tot_h, eff_ratio = calc_effective_hours(ws, v_in, v_out)
    util_coeff = calc_utilization_coeff(ws, rho_input, v_in=v_in, v_out=v_out)
    mean_ws = round(float(np.mean(ws)), 2)
    max_ws = round(float(np.max(ws)), 2)

    # 威布尔拟合
    k_fit, c_fit = fit_weibull(ws)
    wpd_weibull = weibull_energy_density(k_fit, c_fit, rho_input)

    # ══════════════════════════════
    #  等级展示横幅
    # ══════════════════════════════
    grade_color = grade_info["颜色"]
    grade_label = grade_info["等级"]
    grade_desc = grade_info["资源描述"]
    suitable = wpd >= SUITABILITY_THRESHOLD

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {grade_color}33, {grade_color}11);
            border: 2px solid {grade_color};
            border-radius: 12px;
            padding: 20px 28px;
            margin-bottom: 8px;
        ">
            <h2 style="margin:0; color:{grade_color};">
                🏅 {grade_label} &nbsp;·&nbsp; 风能资源{grade_desc}
            </h2>
            <p style="margin:6px 0 0 0; font-size:16px; color:#ccc;">
                实测风功率密度：<b style="color:{grade_color}; font-size:20px;">{wpd:.1f} W/m²</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                评估高度：<b>{height} m</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                开发适宜性：
                <b style="color:{'#66bb6a' if suitable else '#ef5350'};">
                    {'✅ 适合开发' if suitable else '❌ 暂不适合开发'}
                </b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ══════════════════════════════
    #  核心指标卡片
    # ══════════════════════════════
    st.subheader("📊 核心评估指标")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("风功率密度 (实测)", f"{wpd:.1f} W/m²", help="0.5ρV³ 的时序均值")
    r1c2.metric("风功率密度 (威布尔)", f"{wpd_weibull:.1f} W/m²", help="基于威布尔拟合的理论值")
    r1c3.metric("年平均风速", f"{mean_ws} m/s")
    r1c4.metric("最大风速", f"{max_ws} m/s")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("有效风速小时数", f"{eff_h:,} h", help=f"风速在 {v_in}~{v_out} m/s 的小时数")
    r2c2.metric("有效风速占比", f"{eff_ratio} %")
    r2c3.metric("风能利用系数", f"{util_coeff} %", help="有效风速段风能 / 总风能")
    r2c4.metric("威布尔形状参数 k", k_fit)

    st.divider()

    # ══════════════════════════════
    #  国标等级对照表 + 当前位置标注
    # ══════════════════════════════
    st.subheader("📋 国标风资源等级对照表")

    grade_df = pd.DataFrame(WIND_GRADE_TABLE)[
        ["等级", "资源描述", "功率密度下限", "功率密度上限", "参考风速"]
    ].copy()
    grade_df.columns = ["等级", "资源描述", "功率密度下限 (W/m²)",
                         "功率密度上限 (W/m²)", "参考年均风速 (m/s)"]
    grade_df["功率密度上限 (W/m²)"] = grade_df["功率密度上限 (W/m²)"].replace(9999, "—")
    grade_df["当前场址"] = grade_df["等级"].apply(
        lambda g: "◀ 当前场址" if g == grade_label else ""
    )

    def highlight_current(row):
        if row["当前场址"] == "◀ 当前场址":
            return [f"background-color:{grade_color}33; color:white"] * len(row)
        return [""] * len(row)

    st.dataframe(
        grade_df.style.apply(highlight_current, axis=1),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ══════════════════════════════
    #  风功率密度等级仪表盘（进度条）
    # ══════════════════════════════
    st.subheader("🎯 风功率密度区间分布图")

    fig_gauge = go.Figure()

    boundaries = [0, 100, 150, 200, 250, 300, 400, 500]
    colors_bar = [g["颜色"] for g in WIND_GRADE_TABLE]
    labels_bar = [g["等级"] + " " + g["资源描述"] for g in WIND_GRADE_TABLE]

    for i in range(len(boundaries) - 1):
        fig_gauge.add_trace(go.Bar(
            x=[boundaries[i + 1] - boundaries[i]],
            y=["风功率密度"],
            orientation="h",
            base=boundaries[i],
            marker_color=colors_bar[i],
            opacity=0.75,
            name=labels_bar[i],
            text=labels_bar[i],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"{labels_bar[i]}<br>{boundaries[i]}–{boundaries[i+1]} W/m²",
        ))

    fig_gauge.add_vline(
        x=min(wpd, 490),
        line_color="white", line_width=3, line_dash="solid",
        annotation_text=f"当前：{wpd:.1f} W/m²",
        annotation_font_color="white",
        annotation_font_size=13,
        annotation_position="top",
    )

    fig_gauge.update_layout(
        template="plotly_dark",
        barmode="stack",
        height=180,
        xaxis=dict(title="风功率密度 (W/m²)", range=[0, 500]),
        yaxis=dict(showticklabels=False),
        showlegend=True,
        legend=dict(orientation="h", y=-0.5, x=0),
        margin=dict(t=40, b=80, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, key=f"{key_prefix}gauge")

    st.divider()

    # ══════════════════════════════
    #  雷达图：多维度评分
    # ══════════════════════════════
    st.subheader("🕸️ 风资源多维度综合评分")

    score_wpd = min(wpd / 400 * 100, 100)
    score_ws = min(mean_ws / 10 * 100, 100)
    score_eff = min(eff_ratio / 80 * 100, 100)
    score_util = min(util_coeff / 90 * 100, 100)
    score_k = min((float(k_fit) - 1.0) / (3.5 - 1.0) * 100, 100)

    dimensions = ["风功率密度", "平均风速", "有效风时占比", "风能利用系数", "风速稳定性(k)"]
    scores = [score_wpd, score_ws, score_eff, score_util, score_k]
    scores_closed = scores + [scores[0]]
    dims_closed = dimensions + [dimensions[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=dims_closed,
        fill="toself",
        fillcolor=hex_to_rgba(grade_color, 0.27),
        line=dict(color=grade_color, width=2.5),
        name="综合评分",
        hovertemplate="%{theta}<br>评分：%{r:.1f}/100",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[60] * (len(dimensions) + 1),
        theta=dims_closed,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dash"),
        name="参考基准(60分)",
        hoverinfo="skip",
    ))
    fig_radar.update_layout(
        template="plotly_dark", height=420,
        polar=dict(
            radialaxis=dict(range=[0, 100], tickvals=[20, 40, 60, 80, 100]),
        ),
        legend=dict(x=0.8, y=1.1),
        margin=dict(t=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}radar")

    with st.expander("📌 各维度评分说明"):
        score_df = pd.DataFrame({
            "维度": dimensions,
            "得分": [round(s, 1) for s in scores],
            "计算基准": [
                "满分基准 400 W/m²",
                "满分基准 10 m/s",
                "满分基准 80%",
                "满分基准 90%",
                "k 值 1.0~3.5 线性映射",
            ],
        })
        st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.divider()

    # ══════════════════════════════
    #  湍流强度分析
    # ══════════════════════════════
    st.subheader("💨 湍流强度分析")
    ti_df = calc_turbulence_intensity(ws)

    if not ti_df.empty:
        fig_ti = go.Figure()
        fig_ti.add_trace(go.Bar(
            x=ti_df["风速区间中值"],
            y=ti_df["湍流强度"],
            marker=dict(
                color=ti_df["湍流强度"],
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="TI"),
            ),
            name="湍流强度",
        ))
        fig_ti.add_hline(
            y=0.16, line_dash="dash", line_color="#ffa726",
            annotation_text="IEC A类 TI=0.16",
            annotation_font_color="#ffa726",
        )
        fig_ti.add_hline(
            y=0.14, line_dash="dot", line_color="#66bb6a",
            annotation_text="IEC B类 TI=0.14",
            annotation_font_color="#66bb6a",
        )
        fig_ti.update_layout(
            template="plotly_dark", height=340,
            xaxis_title="风速 (m/s)", yaxis_title="湍流强度 TI",
            margin=dict(t=20),
        )
        st.plotly_chart(fig_ti, use_container_width=True, key=f"{key_prefix}ti")

        avg_ti = ti_df["湍流强度"].mean()
        ti_level = (
            "A类（高湍流）" if avg_ti >= 0.16 else
            "B类（中湍流）" if avg_ti >= 0.14 else
            "C类（低湍流）"
        )
        st.info(f"📌 场址平均湍流强度：**{avg_ti:.4f}**，对应 IEC 湍流等级：**{ti_level}**")
    else:
        st.warning("数据量不足，无法计算湍流强度。")

    st.divider()

    # ══════════════════════════════
    #  综合评价结论  ← 本次修改处
    # ══════════════════════════════
    st.subheader("📝 综合评价结论")

    avg_score = round(np.mean(scores), 1)

    if suitable:
        conclusion_color = "#66bb6a"
        conclusion_icon = "✅"
        conclusion_text = (
            f"该场址风功率密度为 {wpd:.1f} W/m²，达到国标 {grade_label}（{grade_desc}）标准，"
            f"年有效风速小时数约 {eff_h:,} 小时（占比 {eff_ratio}%），"
            f"综合评分 {avg_score}/100，"
            f"具备商业化风电开发条件，建议进一步开展微观选址和机组选型工作。"
        )
    else:
        conclusion_color = "#ef5350"
        conclusion_icon = "⚠️"
        conclusion_text = (
            f"该场址风功率密度为 {wpd:.1f} W/m²，属于国标 {grade_label}（{grade_desc}）标准，"
            f"年有效风速小时数约 {eff_h:,} 小时（占比 {eff_ratio}%），"
            f"综合评分 {avg_score}/100，"
            f"当前风能资源条件尚不满足商业化开发要求（建议功率密度 ≥ {SUITABILITY_THRESHOLD} W/m²），"
            f"可考虑更换场址或等待更长观测期再行评估。"
        )

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {conclusion_color};
            background: #ffffff;
            border-radius: 8px;
            padding: 16px 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        ">
            <h4 style="color:{conclusion_color}; margin:0 0 10px 0; font-size:17px;">
                {conclusion_icon} 评估结论
            </h4>
            <p style="color:#1a1a1a; margin:0; line-height:2.0; font-size:15px;">
                {conclusion_text}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ══════════════════════════════
    #  导出评估摘要
    # ══════════════════════════════
    st.subheader("⬇️ 导出评估摘要")
    summary = {
        "评估指标": [
            "评估高度 (m)", "空气密度 (kg/m³)",
            "年平均风速 (m/s)", "最大风速 (m/s)",
            "实测风功率密度 (W/m²)", "威布尔风功率密度 (W/m²)",
            "国标等级", "资源描述",
            "有效风速小时数 (h)", "有效风速占比 (%)",
            "风能利用系数 (%)", "综合评分 (/100)",
            "开发适宜性",
        ],
        "结果": [
            height, rho_input,
            mean_ws, max_ws,
            round(wpd, 1), round(wpd_weibull, 1),
            grade_label, grade_desc,
            eff_h, eff_ratio,
            util_coeff, avg_score,
            "适合开发" if suitable else "暂不适合",
        ],
    }
    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="⬇️ 下载评估摘要 CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="wind_grade_summary.csv",
        mime="text/csv",
        key=f"{key_prefix}download",
    )


