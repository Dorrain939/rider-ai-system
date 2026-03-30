import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk

# --- 1. 数据持久化：生成 200 名骑手的静态数据 ---
@st.cache_data
def generate_riders_data(n=200):
    np.random.seed(88) # 锁定种子，确保演示时画像永久固定
    # 上海市中心坐标范围
    lats = np.random.uniform(31.20, 31.26, n)
    lons = np.random.uniform(121.44, 121.50, n)
    
    data = []
    for i in range(n):
        gender = np.random.choice(["男", "女"], p=[0.88, 0.12])
        age = np.random.randint(19, 52)
        personality = np.random.choice(["稳健型", "激进型", "经验型"])
        rider_type = np.random.choice(["全职", "兼职"])
        
        # 初始动态指标（仅作为基准）
        base_fatigue = np.random.randint(10, 50)
        base_hr = np.random.randint(70, 85)
        
        data.append({
            "工号": f"RID-{2000 + i}",
            "姓名": f"骑手_{i+1}",
            "性别": gender,
            "年龄": age,
            "性格": personality,
            "类型": rider_type,
            "历史违章": np.random.randint(0, 8),
            "lat": lats[i],
            "lon": lons[i],
            "base_fatigue": base_fatigue,
            "base_hr": base_hr,
            "区域熟悉度": np.random.uniform(0.6, 1.0)
        })
    return pd.DataFrame(data)

# --- 2. AI 算法引擎 ---
class AICoreEngine:
    @staticmethod
    def calculate_k(rider_static, dynamic_input, env_params):
        k = 1.0
        logs = []
        # 生理偏移 (RNN/LSTM逻辑)
        if dynamic_input['hr'] > 125: 
            k += 0.12; logs.append("心率异常偏移 +0.12")
        if dynamic_input['fatigue'] > 75: 
            k += 0.15; logs.append("高疲劳度补偿 +0.15")
        
        # 静态画像偏移
        if rider_static['年龄'] > 45: 
            k += 0.05; logs.append("年龄生理补偿 +0.05")
        if rider_static['性格'] == "激进型" and rider_static['历史违章'] > 3:
            k += 0.08; logs.append("驾驶风格风险控制 +0.08")
            
        # 外部环境 (GNN/环境逻辑)
        if env_params['weather'] == "暴雨": 
            k += 0.20; logs.append("极端天气溢价 +0.20")
        if env_params['terrain'] == "复杂老旧小区": 
            k += 0.08; logs.append("地形复杂度补偿 +0.08")
            
        return round(min(max(k, 0.8), 1.5), 2), logs

# --- 3. 页面配置与初始化 ---
st.set_page_config(page_title="全域调度指挥大脑", layout="wide")
st.title("🛡️ 全链路生态调度指挥系统 v5.0")

df_riders = generate_riders_data(200)

# 侧边栏：全局动态变量
st.sidebar.header("🌍 外部因子实时模拟")
weather = st.sidebar.selectbox("天气状态", ["晴朗", "多云", "暴雨", "高温"])
terrain = st.sidebar.selectbox("区域地形", ["现代楼宇区", "复杂老旧小区"])
traffic_density = st.sidebar.slider("路网拥堵度 (GNN输入)", 0, 100, 30)

# --- 4. 双层看板设计 ---
tab_global, tab_personal = st.tabs(["🌐 全域数据大屏", "👤 骑手精准干预"])

# --- Tab 1: 全域看板 (Macro) ---
with tab_global:
    # 顶部指标
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("在线骑手总数", "200", "实时")
    m2.metric("全城平均疲劳度", f"{df_riders['base_fatigue'].mean():.1f}", "安全范围内")
    m3.metric("异常预警(心率/违章)", "5 名", "-2", delta_color="inverse")
    m4.metric("平均调度补偿系数 K", "1.08")

    st.markdown("### 📍 实时骑手分布与风险热力图 (GNN 动态映射)")
    
    # 模拟 GNN 压力颜色：疲劳值越高越红
    df_riders['color_r'] = df_riders['base_fatigue'] * 2.5
    df_riders['color_g'] = 255 - (df_riders['base_fatigue'] * 2.5)
    
    # 使用 Pydeck 绘制地图
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_riders,
        get_position=["lon", "lat"],
        get_color="[color_r, color_g, 100, 160]",
        get_radius=120,
        pickable=True,
    )
    
    view_state = pdk.ViewState(latitude=31.23, longitude=121.47, zoom=12, pitch=45)
    
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "工号: {工号}\n性格: {性格}\n疲劳基准: {base_fatigue}"}
    ))

    # 统计图表
    c1, c2 = st.columns(2)
    with c1:
        fig_age = px.histogram(df_riders, x="年龄", title="骑手年龄分布", color_discrete_sequence=['#3366CC'])
        st.plotly_chart(fig_age, use_container_width=True)
    with c2:
        fig_type = px.pie(df_riders, names='类型', title="用工结构比例", hole=0.4)
        st.plotly_chart(fig_type, use_container_width=True)

# --- Tab 2: 个人干预 (Micro) ---
with tab_personal:
    st.subheader("🎯 骑手个体精准干预与仿真")
    
    # 搜索与选择
    search_col1, search_col2 = st.columns([1, 3])
    with search_col1:
        selected_id = st.selectbox("搜索/选择骑手工号", df_riders['工号'].tolist())
    
    r_static = df_riders[df_riders['工号'] == selected_id].iloc[0]

    # 展示锁定画像
    st.info(f"正在锁定骑手：**{r_static['姓名']}** ({r_static['工号']})")
    
    col_static, col_dynamic, col_ai = st.columns([1, 1.2, 1.5])
    
    with col_static:
        st.markdown("#### 1. 固化画像 (Locked)")
        st.write(f"**年龄/性别:** {r_static['年龄']} 岁 / {r_static['性别']}")
        st.write(f"**性格标签:** {r_static['性格']}")
        st.write(f"**用工类型:** {r_static['类型']}")
        st.write(f"**历史违章次数:** {r_static['历史违章']}")
        st.write(f"**区域熟练度:** {r_static['区域熟悉度']:.2%}")
        st.progress(r_static['区域熟悉度'])

    with col_dynamic:
        st.markdown("#### 2. 实时生理仿真输入")
        # 这里的滑块仅作为输入参数，不改变 df_riders 里的静态值
        sim_hr = st.slider("模拟心率 (BPM)", 60, 160, int(r_static['base_hr']))
        sim_fatigue = st.slider("模拟疲劳度 (LSTM预测值)", 0, 100, int(r_static['base_fatigue']))
        sim_merchant = st.slider("商家出餐延迟补偿 (min)", 0, 20, 0)

    with col_ai:
        st.markdown("#### 3. AI 决策干预结果")
        
        # 调用算法
        dyn_in = {'hr': sim_hr, 'fatigue': sim_fatigue}
        env_in = {'weather': weather, 'terrain': terrain}
        
        k_val, k_logs = AICoreEngine.calculate_k(r_static, dyn_in, env_in)
        
        # 结果展示
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("动态补偿系数 K", f"{k_val}")
        res_col2.write(f"**建议最终时限:** \n {int(25 * k_val)} 分钟 (原25min)")
        
        st.markdown("---")
        st.write("**决策链路依据 (Explainable AI):**")
        if not k_logs:
            st.caption("环境平稳，无异常偏移项")
        for log in k_logs:
            st.caption(f"✅ {log}")
            
        # 状态干预动作
        if sim_fatigue > 85:
            st.error("🚨 警告：骑手已进入生理极限，系统已将其标记为‘强制休整’，暂停派单。")
        elif sim_hr > 130:
            st.warning("⚠️ 提醒：骑手心率过高，建议 HR 发起语音关怀。")

    # 底部：时序趋势 (RNN/LSTM模拟)
    st.markdown("#### 4. 骑手生理指标趋势 (LSTM 实时推演)")
    time_series = pd.DataFrame({
        "过去30分钟心率": np.random.normal(sim_hr, 2, 30),
        "预测未来10分钟疲劳度": np.linspace(sim_fatigue, sim_fatigue + 5, 30)
    })
    st.line_chart(time_series)

st.sidebar.markdown("---")
st.sidebar.caption("系统内核：Torch + GNN_Cluster_v2 | 韧性调度逻辑已加载")