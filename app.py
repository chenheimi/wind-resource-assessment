import streamlit as st
from app_part1 import setup_page, render_sidebar, render_tab1, render_tab2
from app_part2 import render_tab3, render_tab4, render_tab5
from online_wind import show_online_wind_page
from wind_grade import show_wind_grade_page
from report_export import show_report_export_page
from turbine_design import show_turbine_design_page    # ← 新增这一行

setup_page()

df, turbine, turbine_name, site_name, hub_height, rho = render_sidebar()

st.markdown('<p class="main-title">🌬️ 风资源评估系统</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">站点：{site_name} ｜ '
    f'轮毂高度：{hub_height} m ｜ '
    f'空气密度：{rho} kg/m³</p>',
    unsafe_allow_html=True,
)

if df is None:
    st.info("👈 请在左侧生成或上传数据后开始分析", icon="ℹ️")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([    # ← tab8改为tab8, tab9
    "📊 数据概览",
    "📈 风速分析",
    "🌹 风玫瑰图",
    "📐 威布尔拟合",
    "⚡ 发电量估算",
    "🌐 在线评估",
    "🏅 风资源等级",
    "📄 报告导出",
    "⚙️ 机组设计",    # ← 新增
])

with tab1: render_tab1(df)
with tab2: render_tab2(df, turbine)
with tab3: render_tab3(df)
with tab4: render_tab4(df, rho)
with tab5: render_tab5(df, turbine, turbine_name)
with tab6: show_online_wind_page()
with tab7: show_wind_grade_page(df=df, rho=rho, height=hub_height)
with tab8: show_report_export_page(df=df, rho=rho, height=hub_height,
                                    key_prefix="main_report_")
with tab9: show_turbine_design_page(df=df, rho=rho, hub_height=hub_height,    # ← 新增
                                     key_prefix="main_turbine_")


