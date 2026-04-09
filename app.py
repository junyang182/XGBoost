import streamlit as st
import pandas as pd
import xgboost as xgb
import os

# 1. 页面基础配置（必须放在脚本最开始）
st.set_page_config(
    page_title="老年失能风险评估系统",
    page_icon="⚕️",
    layout="centered"
)

# 2. 数据与模型加载模块
# 使用 @st.cache_resource 装饰器，确保模型只需在网站启动时训练一次，加快后续用户的访问速度
@st.cache_resource
def build_model():
    file_path = "XGBoost.csv"
    if not os.path.exists(file_path):
        return None

    # 读取数据集
    df = pd.read_csv(file_path)

    # 提取特征 (X) 和标签 (y) 
    # 此处特征名称与您提供的 R 代码和 CSV 保持一致
    features = ['年龄', '受教育水平', '自评健康', '运动情况', '睡眠时长', '工作状况']
    X = df[features]
    y = df['ADL']

    # 初始化并训练 XGBoost 分类模型
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X, y)
    return model

# 3. 网站主界面 UI 设计
st.title("⚕️ 老年失能 (ADL) 风险在线评估工具")
st.markdown("""
> **项目背景**：本工具基于真实的健康数据集，采用先进的 **XGBoost 机器学习算法** 构建。
> 医生或用户可以通过输入下方的各项生活及健康指标，系统将实时计算并反馈日常活动能力（ADL）受损的风险概率。
""")

st.divider()

# 加载模型
model = build_model()

if model is None:
    st.error("🚨 **系统初始化失败**：未找到 `XGBoost.csv` 文件。请确保您的数据文件与本程序处于同一文件夹内。")
else:
    # 4. 侧边栏：用于收集用户的预测因子输入
    st.sidebar.header("📋 请输入预测因子")
    st.sidebar.markdown("请根据实际情况滑动或选择以下参数：")

    age = st.sidebar.slider("年龄 (岁)", min_value=50, max_value=120, value=70, step=1)
    
    edu = st.sidebar.selectbox(
        "受教育水平",
        options=[1, 2, 3, 4, 5],
        help="1-5 级，请根据您的原始数据编码习惯选择"
    )
    
    health = st.sidebar.selectbox(
        "自评健康状况",
        options=[1, 2, 3, 4, 5],
        help="请选择自我评估的健康评分"
    )
    
    exercise = st.sidebar.selectbox(
        "运动情况",
        options=[0, 1, 2, 3],
        help="级别越高代表运动频率或强度越高"
    )
    
    sleep = st.sidebar.slider(
        "睡眠时长 (小时/天)",
        min_value=2.0, max_value=16.0, value=7.0, step=0.5
    )
    
    work = st.sidebar.radio(
        "工作状况",
        options=[0, 1],
        format_func=lambda x: "在职 (1)" if x == 1 else "非在职 (0)"
    )

    # 5. 预测按钮与结果反馈
    if st.sidebar.button("⚡ 计算风险值", type="primary"):
        
        # 组织输入数据为 DataFrame 格式
        input_df = pd.DataFrame({
            '年龄': [age],
            '受教育水平': [edu],
            '自评健康': [health],
            '运动情况': [exercise],
            '睡眠时长': [sleep],
            '工作状况': [work]
        })

        # 调用模型进行概率预测
        # predict_proba 返回 [类别0概率, 类别1概率]，[0][1] 即获取发生失能(类别1)的概率
        proba = model.predict_proba(input_df)[0][1]

        # 展示评估结果
        st.subheader("📊 评估结果")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="ADL失能风险发生概率", value=f"{proba * 100:.1f}%")
        with col2:
            st.write("**风险水平指示：**")
            st.progress(float(proba))

        # 根据概率给出具体的健康建议
        if proba < 0.3:
            st.success("✅ **低风险**：当前预测显示失能风险较低。请继续保持良好的生活和作息习惯！")
        elif proba < 0.6:
            st.warning("⚠️ **中等风险**：当前预测显示存在一定的失能风险。建议多关注身体变化，适度增加活动量。")
        else:
            st.error("🚨 **高风险**：当前预测显示失能风险较高！建议尽快寻求专业医疗人员的帮助，进行针对性的评估与干预。")

        st.markdown("---")
        st.caption("免责声明：本评估结果由基于数据的机器学习模型自动生成，仅供参考，不具备临床医学诊断效力。")
