# turbine_design.py - 风力发电机组设计参数计算模块
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from utils import fit_weibull
except:
    def fit_weibull(ws_data):
        from scipy.stats import weibull_min
        ws = ws_data[~np.isnan(ws_data)]
        params = weibull_min.fit(ws, floc=0)
        k, loc, c = params
        return k, c


def calculate_design_wind_speeds(ws_data, safety_factor=1.4):
    """根据风资源数据计算机组设计风速"""
    ws = ws_data[~np.isnan(ws_data)]

    mean_ws = np.mean(ws)
    std_ws = np.std(ws)
    max_ws = np.max(ws)

    k, c = fit_weibull(ws)

    # 1. 切入风速
    cut_in = 3.0

    # 2. 额定风速
    if k < 2.0:
        rated = mean_ws * 2.2
    else:
        rated = mean_ws * 1.8
    rated = min(max(rated, 9.0), 15.0)

    # 3. 切出风速
    p995 = np.percentile(ws, 99.5)
    cut_out_method1 = p995 * safety_factor

    T = 50 * 365 * 24 / len(ws)
    extreme_ws = c * (-np.log(1 / T)) ** (1 / k)
    cut_out_method2 = extreme_ws * 0.8

    cut_out = np.mean([cut_out_method1, cut_out_method2])
    cut_out = min(max(cut_out, 20.0), 25.0)

    survival_ws = extreme_ws

    return {
        'cut_in': round(cut_in, 1),
        'rated': round(rated, 1),
        'cut_out': round(cut_out, 1),
        'survival': round(survival_ws, 1),
        'mean_ws': round(mean_ws, 2),
        'weibull_k': round(k, 3),
        'weibull_c': round(c, 2),
        'p995_ws': round(p995, 1),
    }


def calculate_rotor_diameter(rated_power_mw, rated_ws, rho=1.225, cp=0.45):
    """根据额定功率和额定风速反推叶轮直径"""
    rated_power_w = rated_power_mw * 1e6
    area = rated_power_w / (0.5 * rho * rated_ws ** 3 * cp)
    diameter = 2 * np.sqrt(area / np.pi)
    return round(diameter, 1)


def generate_power_curve(cut_in, rated, cut_out, rated_power_mw):
    """生成功率曲线"""
    ws_range = np.linspace(0, 30, 300)
    power = np.zeros_like(ws_range)

    for i, v in enumerate(ws_range):
        if v < cut_in:
            power[i] = 0
        elif v < rated:
            power[i] = rated_power_mw * ((v - cut_in) / (rated - cut_in)) ** 3
        elif v < cut_out:
            power[i] = rated_power_mw
        else:
            power[i] = 0

    return ws_range, power


def calculate_annual_energy(ws_data, cut_in, rated, cut_out, rated_power_mw, rho=1.225):
    """计算年发电量"""
    ws = ws_data[~np.isnan(ws_data)]
    ws_curve, power_curve = generate_power_curve(cut_in, rated, cut_out, rated_power_mw)
    power_interp = np.interp(ws, ws_curve, power_curve)
    mean_power_mw = np.mean(power_interp)
    annual_energy_mwh = mean_power_mw * 8760
    capacity_factor = mean_power_mw / rated_power_mw
    full_load_hours = annual_energy_mwh / rated_power_mw

    return {
        'annual_energy_mwh': round(annual_energy_mwh, 1),
        'capacity_factor': round(capacity_factor, 3),
        'full_load_hours': int(full_load_hours),
        'mean_power_mw': round(mean_power_mw, 3),
    }


def optimize_hub_height(ws_ground, alpha=0.14, ground_height=10,
                        test_heights=[70, 80, 90, 100, 110, 120]):
    """轮毂高度优化"""
    results = []
    for h in test_heights:
        ws_h = ws_ground * (h / ground_height) ** alpha
        mean_ws_h = np.mean(ws_h)
        wpd = 0.5 * 1.225 * mean_ws_h ** 3
        results.append({'height': h, 'mean_ws': round(mean_ws_h, 2), 'wpd': round(wpd, 1)})

    df = pd.DataFrame(results)
    optimal_idx = df['wpd'].idxmax()
    return df, df.iloc[optimal_idx]['height']


def show_turbine_design_page(df=None, rho=1.225, hub_height=70, key_prefix="turbine_"):
    """主页面函数"""

    st.markdown("## ⚙️ 风力发电机组设计参数计算")
    st.caption("基于场址风资源数据，智能推荐3.4MW直驱式风力发电机组设计参数")

    if df is None or len(df) == 0:
        st.warning("⚠️ 请先在其他页面生成或上传风速数据")
        return

    # 识别风速列
    ws_col = None
    for col in ['wind_speed', 'ws', 'speed', 'Wind Speed', '风速']:
        if col in df.columns:
            ws_col = col
            break

    if ws_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            ws_col = numeric_cols[0]
        else:
            st.error("❌ 未找到风速数据列")
            return

    ws_data = df[ws_col].dropna().values

    st.divider()

    # ── 参数输入 ──────────────────────────────
    st.subheader("📝 机组基本参数设置")
    col1, col2, col3 = st.columns(3)

    with col1:
        rated_power = st.number_input(
            "额定功率 (MW)",
            value=3.4,
            min_value=0.5,
            max_value=10.0,
            step=0.1,
            key=f"{key_prefix}power"
        )
    with col2:
        turbine_type = st.selectbox(
            "机组类型",
            ["直驱式", "双馈式"],
            index=0,
            key=f"{key_prefix}type"
        )
    with col3:
        safety_factor = st.slider(
            "安全系数",
            min_value=1.2,
            max_value=1.6,
            value=1.4,
            step=0.1,
            key=f"{key_prefix}safety"
        )

    st.divider()

    # ── 计算设计风速 ──────────────────────────
    design_params = calculate_design_wind_speeds(ws_data, safety_factor)

    # 提前计算叶轮直径（供公式展示用）
    cp_default = 0.45
    rotor_d = calculate_rotor_diameter(rated_power, design_params['rated'], rho, cp_default)

    st.subheader("🎯 推荐设计风速参数")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("切入风速", f"{design_params['cut_in']} m/s", help="开始发电的最低风速")
    with col_b:
        st.metric("额定风速", f"{design_params['rated']} m/s", help="达到额定功率的风速")
    with col_c:
        st.metric("切出风速", f"{design_params['cut_out']} m/s", help="保护停机风速")
    with col_d:
        st.metric("生存风速", f"{design_params['survival']} m/s", help="50年一遇极端风速")

    # ── 详细计算公式展示 ──────────────────────
    with st.expander("📊 计算依据与详细公式"):
        st.markdown(f"""
## 一、场址风资源特征参数

| 参数 | 符号 | 数值 | 说明 |
|------|------|------|------|
| 年平均风速 | $\\bar{{v}}$ | {design_params['mean_ws']} m/s | 实测数据统计平均 |
| 威布尔形状参数 | $k$ | {design_params['weibull_k']} | 描述风速分布形态 |
| 威布尔尺度参数 | $c$ | {design_params['weibull_c']} m/s | 特征风速 |
| 99.5%分位风速 | $v_{{99.5}}$ | {design_params['p995_ws']} m/s | 极端风速统计 |

---

## 二、威布尔分布理论基础

### 📐 概率密度函数 (PDF)

$$
f(v) = \\frac{{k}}{{c}} \\left( \\frac{{v}}{{c}} \\right)^{{k-1}} \\exp\\left[ -\\left( \\frac{{v}}{{c}} \\right)^k \\right]
$$

### 📊 累积分布函数 (CDF)

$$
F(v) = 1 - \\exp\\left[ -\\left( \\frac{{v}}{{c}} \\right)^k \\right]
$$

### 🎯 平均风速与威布尔参数的关系

$$
\\bar{{v}} = c \\cdot \\Gamma\\left(1 + \\frac{{1}}{{k}}\\right)
$$

其中 $\\Gamma$ 为伽马函数。

---

## 三、设计风速计算公式

### 1️⃣ 切入风速 $v_{{in}}$

$$
v_{{in}} = 3.0 \\ \\text{{m/s}}
$$

**说明**：行业标准值，保证低风速时能启动发电。

---

### 2️⃣ 额定风速 $v_{{rated}}$ ⭐

**自适应优化算法：**

$$
v_{{rated}} = \\begin{{cases}}
\\bar{{v}} \\times 2.2, & k < 2.0 \\ \\text{{（风速波动大）}} \\\\
\\bar{{v}} \\times 1.8, & k \\geq 2.0 \\ \\text{{（风速稳定）}}
\\end{{cases}}
$$

**约束条件：**

$$
9.0 \\leq v_{{rated}} \\leq 15.0 \\ \\text{{m/s}}
$$

**本场址计算过程：**

$$
v_{{rated}} = {design_params['mean_ws']} \\times {2.2 if design_params['weibull_k'] < 2.0 else 1.8} = {design_params['rated']} \\ \\text{{m/s}}
$$

---

### 3️⃣ 切出风速 $v_{{out}}$ ⭐⭐

**方法1：统计分位数法**

$$
v_{{out,1}} = v_{{99.5}} \\times \\eta_{{safe}} = {design_params['p995_ws']} \\times {safety_factor} = {round(design_params['p995_ws'] * safety_factor, 1)} \\ \\text{{m/s}}
$$

其中 $\\eta_{{safe}}$ 为安全系数。

**方法2：极端风速法**

基于威布尔分布的重现期风速公式：

$$
v_{{T}} = c \\cdot \\left[ -\\ln\\left( \\frac{{1}}{{T}} \\right) \\right]^{{\\frac{{1}}{{k}}}}
$$

**50年一遇极端风速重现期：**

$$
T_{{50}} = \\frac{{50 \\times 365 \\times 24}}{{N_{{sample}}}}
$$

**极端风速计算：**

$$
v_{{extreme}} = {design_params['weibull_c']} \\times \\left[ -\\ln\\left( \\frac{{1}}{{T_{{50}}}} \\right) \\right]^{{\\frac{{1}}{{{design_params['weibull_k']}}}}} = {design_params['survival']} \\ \\text{{m/s}}
$$

$$
v_{{out,2}} = 0.8 \\times v_{{extreme}} = {round(0.8 * design_params['survival'], 1)} \\ \\text{{m/s}}
$$

**最终综合取值：**

$$
v_{{out}} = \\frac{{v_{{out,1}} + v_{{out,2}}}}{{2}} = {design_params['cut_out']} \\ \\text{{m/s}}
$$

**约束条件：** $20.0 \\leq v_{{out}} \\leq 25.0$ m/s

---

### 4️⃣ 生存风速 $v_{{survival}}$

$$
v_{{survival}} = v_{{extreme}} = {design_params['survival']} \\ \\text{{m/s}}
$$

**用途**：结构强度设计的极限载荷工况，机组停机但需保证结构安全。

---

## 四、叶轮直径计算公式 ⭐⭐⭐

### 风力机基本功率方程

$$
P = \\frac{{1}}{{2}} \\rho A v^3 C_p
$$

其中：

| 符号 | 含义 | 数值 |
|------|------|------|
| $P$ | 额定功率 | {rated_power} MW |
| $\\rho$ | 空气密度 | {rho} kg/m³ |
| $A$ | 扫掠面积 | 待求 m² |
| $v$ | 额定风速 | {design_params['rated']} m/s |
| $C_p$ | 风能利用系数 | ≤ 0.593（贝兹极限） |

### 叶轮直径反推公式

$$
A = \\frac{{P}}{{0.5 \\cdot \\rho \\cdot v_{{rated}}^3 \\cdot C_p}} = \\pi \\left( \\frac{{D}}{{2}} \\right)^2
$$

$$
D = 2 \\sqrt{{ \\frac{{P}}{{0.5 \\cdot \\pi \\cdot \\rho \\cdot v_{{rated}}^3 \\cdot C_p}} }}
$$

**本机组计算结果（$C_p = 0.45$）：**

$$
D = 2 \\sqrt{{ \\frac{{{rated_power} \\times 10^6}}{{0.5 \\times \\pi \\times {rho} \\times {design_params['rated']}^3 \\times 0.45}} }} \\approx {rotor_d} \\ \\text{{m}}
$$

---

## 五、年发电量计算公式 💡

### 功率曲线分段模型

$$
P(v) = \\begin{{cases}}
0, & v < v_{{in}} \\\\
P_{{rated}} \\cdot \\left( \\dfrac{{v - v_{{in}}}}{{v_{{rated}} - v_{{in}}}} \\right)^3, & v_{{in}} \\leq v < v_{{rated}} \\\\
P_{{rated}}, & v_{{rated}} \\leq v < v_{{out}} \\\\
0, & v \\geq v_{{out}}
\\end{{cases}}
$$

### 平均输出功率（离散求和）

$$
\\bar{{P}} = \\frac{{1}}{{N}} \\sum_{{i=1}}^{{N}} P(v_i)
$$

### 年发电量

$$
E_{{annual}} = \\bar{{P}} \\times 8760 \\ \\text{{(MWh/年)}}
$$

### 容量系数

$$
CF = \\frac{{\\bar{{P}}}}{{P_{{rated}}}} \\times 100\\%
$$

### 等效满发小时数

$$
h_{{full}} = \\frac{{E_{{annual}}}}{{P_{{rated}}}}
$$

---

## 六、轮毂高度优化公式 📏

### 风切变幂律公式

$$
\\frac{{v_2}}{{v_1}} = \\left( \\frac{{h_2}}{{h_1}} \\right)^\\alpha
$$

其中 $\\alpha$ 为风切变指数（地形越复杂，$\\alpha$ 值越大）。

### 各高度风功率密度

$$
WPD(h) = \\frac{{1}}{{2}} \\rho \\left[ v_1 \\cdot \\left( \\frac{{h}}{{h_1}} \\right)^\\alpha \\right]^3
$$

**优化目标**：选择 $WPD(h)$ 最大的高度作为推荐轮毂高度。

---

## 📚 参考标准

- **GB/T 18709-2002** 《风电场风能资源评估方法》
- **IEC 61400-1:2019** 《风力发电机组 设计要求》
- **GB/T 18451.1-2012** 《风力发电机组 功率特性测试》
- **GB 50009-2012** 《建筑结构荷载规范》（极端风速参考）
        """)

    st.divider()

    # ── 叶轮设计 ──────────────────────────────
    st.subheader("🔄 叶轮设计参数")
    col_e, col_f = st.columns(2)

    with col_e:
        cp = st.slider(
            "风能利用系数 Cp",
            min_value=0.35,
            max_value=0.50,
            value=0.45,
            step=0.01,
            help="理论最大值0.593（贝兹极限），实际机组通常0.40-0.48",
            key=f"{key_prefix}cp"
        )
    with col_f:
        rotor_d = calculate_rotor_diameter(rated_power, design_params['rated'], rho, cp)
        st.metric("推荐叶轮直径", f"{rotor_d} m",
                  help=f"根据额定功率{rated_power}MW和额定风速{design_params['rated']}m/s反推")

    sweep_area = np.pi * (rotor_d / 2) ** 2
    tip_speed_ratio = 7.0
    rated_rpm = tip_speed_ratio * design_params['rated'] * 60 / (np.pi * rotor_d)

    col_g, col_h = st.columns(2)
    col_g.metric("扫掠面积", f"{sweep_area:.1f} m²")
    col_h.metric("额定转速", f"{rated_rpm:.2f} rpm",
                 help=f"假设叶尖速比={tip_speed_ratio}")

    st.divider()

    # ── 功率曲线 ──────────────────────────────
    st.subheader("📈 功率曲线")
    ws_curve, power_curve = generate_power_curve(
        design_params['cut_in'],
        design_params['rated'],
        design_params['cut_out'],
        rated_power
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ws_curve,
        y=power_curve,
        mode='lines',
        name='功率曲线',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    fig.add_vline(x=design_params['cut_in'], line_dash="dash",
                  line_color="green", annotation_text="切入")
    fig.add_vline(x=design_params['rated'], line_dash="dash",
                  line_color="orange", annotation_text="额定")
    fig.add_vline(x=design_params['cut_out'], line_dash="dash",
                  line_color="red", annotation_text="切出")
    fig.add_vline(x=design_params['mean_ws'], line_dash="dot",
                  line_color="purple",
                  annotation_text=f"场址平均({design_params['mean_ws']}m/s)")
    fig.update_layout(
        xaxis_title="风速 (m/s)",
        yaxis_title="输出功率 (MW)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 发电量估算 ──────────────────────────────
    st.subheader("💡 年发电量估算")
    energy_result = calculate_annual_energy(
        ws_data,
        design_params['cut_in'],
        design_params['rated'],
        design_params['cut_out'],
        rated_power,
        rho
    )

    col_i, col_j, col_k, col_l = st.columns(4)
    with col_i:
        st.metric("年发电量", f"{energy_result['annual_energy_mwh']:,.1f} MWh")
    with col_j:
        st.metric("容量系数", f"{energy_result['capacity_factor'] * 100:.1f}%")
    with col_k:
        st.metric("等效满发小时", f"{energy_result['full_load_hours']:,} h")
    with col_l:
        annual_income = energy_result['annual_energy_mwh'] * 0.35
        st.metric("年收益估算", f"{annual_income:,.0f} 万元",
                  help="假设上网电价0.35元/kWh")

    st.divider()

    # ── 轮毂高度优化 ──────────────────────────
    st.subheader("📏 轮毂高度优化分析")
    alpha = st.slider(
        "风切变指数 α",
        min_value=0.10,
        max_value=0.25,
        value=0.14,
        step=0.01,
        help="地形越复杂，α值越大",
        key=f"{key_prefix}alpha"
    )
    height_df, optimal_h = optimize_hub_height(ws_data, alpha, hub_height)

    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("不同高度的平均风速", "不同高度的风功率密度")
    )
    fig2.add_trace(
        go.Bar(x=height_df['height'], y=height_df['mean_ws'],
               name='平均风速', marker_color='lightblue'),
        row=1, col=1
    )
    fig2.add_trace(
        go.Bar(x=height_df['height'], y=height_df['wpd'],
               name='风功率密度', marker_color='lightcoral'),
        row=1, col=2
    )
    fig2.update_xaxes(title_text="高度 (m)", row=1, col=1)
    fig2.update_xaxes(title_text="高度 (m)", row=1, col=2)
    fig2.update_yaxes(title_text="风速 (m/s)", row=1, col=1)
    fig2.update_yaxes(title_text="功率密度 (W/m²)", row=1, col=2)
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.success(f"✅ 推荐轮毂高度：**{int(optimal_h)} m**（测试高度范围内风功率密度最大）")

    st.divider()

    # ── 汇总表格 ──────────────────────────────
    st.subheader("📋 机组设计参数汇总")
    summary_data = {
        "参数类别": ["机组基本参数"] * 4 + ["设计风速"] * 4 + ["叶轮参数"] * 4 + ["发电性能"] * 4,
        "参数名称": [
            "额定功率", "机组类型", "轮毂高度", "叶轮直径",
            "切入风速", "额定风速", "切出风速", "生存风速",
            "扫掠面积", "额定转速", "叶尖速比", "风能利用系数",
            "年发电量", "容量系数", "等效满发小时", "年收益"
        ],
        "数值": [
            f"{rated_power} MW", turbine_type, f"{int(optimal_h)} m", f"{rotor_d} m",
            f"{design_params['cut_in']} m/s", f"{design_params['rated']} m/s",
            f"{design_params['cut_out']} m/s", f"{design_params['survival']} m/s",
            f"{sweep_area:.1f} m²", f"{rated_rpm:.2f} rpm",
            f"{tip_speed_ratio:.1f}", f"{cp:.2f}",
            f"{energy_result['annual_energy_mwh']:,.1f} MWh",
            f"{energy_result['capacity_factor'] * 100:.1f}%",
            f"{energy_result['full_load_hours']:,} h",
            f"{annual_income:,.0f} 万元"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    csv = summary_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 导出设计参数表（CSV）",
        data=csv,
        file_name=f"机组设计参数_{rated_power}MW.csv",
        mime="text/csv",
        key=f"{key_prefix}download"
    )

