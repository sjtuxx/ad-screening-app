import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. 语言和文本内容 (LANG_STRINGS) ---
# 包含所有界面文本的双语字典
# [V5 变更] 移除了选项中的 (0/1)，并为中文添加了 'has_none' 映射
LANG_STRINGS = {
    'zh': {
        'page_title': "AD筛选工具",
        'app_title': "👨‍⚕️ 阿尔茨海默病 (AD) 机器学习筛选工具",
        'model_info': "**模型版本:** `{model_name}` (源自 V4.43 脚本) | **分类阈值 (Youden 指数):** `{threshold:.4f}`",
        'lang_select': "语言 (Language)",
        'sidebar_header': "患者信息输入",
        'sidebar_help': "请输入以下 12 项特征：",
        'subheader_continuous': "连续特征",
        'subheader_binary': "二分类特征",
        'features': {
            'age': '年龄 (岁)',
            'BMI': 'BMI (kg/m²)',
            'ABO': '血清Aβ寡聚体 (ABO)',
            'edu': '教育年限 (年)',
            'gender': '性别', # [V5] 移除了 (1=男性)
            'alcohol': '饮酒史',
            'dementia_family_history': '痴呆家族史',
            'hypertension': '高血压',
            'diabetes': '糖尿病',
            'hyperlipidemia': '高血脂',
            'APOE4_carrier': 'APOE ε4 携带状态',
            'GDS_DIA': '抑郁症状 (GDS)'
        },
        'gender_map': {'女性': 0, '男性': 1}, # [V5]
        'binary_map_status': {'否': 0, '是': 1}, # [V5]
        'binary_map_history': {'无': 0, '有': 1}, # [V5]
        'input_summary': "模型输入总览",
        'input_help': "请在左侧侧边栏中修改值。",
        'input_table_cols': {'feature': "特征", 'value': "输入值"},
        'results_header': "📈 预测结果",
        'predict_button': "运行模型预测",
        'results_recommendation': "**诊断建议:**",
        'results_risk_high': "高风险",
        'results_risk_low': "低风险",
        'results_delta_high': "高于阈值 {threshold:.4f}",
        'results_delta_low': "低于阈值 {threshold:.4f}",
        'results_metric_label': "MCI/AD 预测概率",
        'results_caption': "该概率值 ({probability:.4f}) 表示模型预测个体为认知受损 (MCI/AD) 的可能性。",
        'errors': {
            'load_fail_header': "❌ 模型加载失败",
            'load_fail_help': "请检查下方的错误信息并确保模型文件存在。",
            'file_not_found': "❌ 错误：在 {path} 未找到模型文件。",
            'file_not_found_help': "请确保您已运行 V4.43 脚本的步骤 25，并且 'ad_screening_model_v4_43.joblib' 文件与此 app.py 在同一个文件夹中。",
            'load_error': "加载模型时出错： {e}",
            'predict_error': "预测过程中发生错误：",
            'predict_error_help': "请检查输入数据。"
        }
    },
    'en': {
        'page_title': "AD Screening Tool",
        'app_title': "👨‍⚕️ Alzheimer's Disease (AD) ML Screening Tool",
        'model_info': "**Model Version:** `{model_name}` (from V4.43 Script) | **Classification Threshold (Youden Index):** `{threshold:.4f}`",
        'lang_select': "语言 (Language)",
        'sidebar_header': "Patient Information Input",
        'sidebar_help': "Please enter the following 12 features:",
        'subheader_continuous': "Continuous Features",
        'subheader_binary': "Binary Features",
        'features': {
            'age': 'Age (years)',
            'BMI': 'BMI (kg/m²)',
            'ABO': 'Serum Aβ Oligomers (ABO)',
            'edu': 'Education (years)',
            'gender': 'Sex',
            'alcohol': 'Alcohol Use',
            'dementia_family_history': 'Family History of Dementia',
            'hypertension': 'Hypertension',
            'diabetes': 'Diabetes Mellitus',
            'hyperlipidemia': 'Hyperlipidemia',
            'APOE4_carrier': 'APOE ε4 Carrier Status',
            'GDS_DIA': 'Depressive Symptoms (GDS)'
        },
        'gender_map': {'Female': 0, 'Male': 1}, # [V5]
        'binary_map_status': {'No': 0, 'Yes': 1}, # [V5]
        'binary_map_history': {'No': 0, 'Yes': 1}, # [V5]
        'input_summary': "Model Input Overview",
        'input_help': "Please modify values in the left sidebar.",
        'input_table_cols': {'feature': "Feature", 'value': "Input Value"},
        'results_header': "📈 Prediction Results",
        'predict_button': "Run Model Prediction",
        'results_recommendation': "**Recommendation:**",
        'results_risk_high': "High Risk",
        'results_risk_low': "Low Risk",
        'results_delta_high': "Above threshold {threshold:.4f}",
        'results_delta_low': "Below threshold {threshold:.4f}",
        'results_metric_label': "MCI/AD Predicted Probability",
        'results_caption': "This probability ({probability:.4f}) represents the model's predicted likelihood of cognitive impairment (MCI/AD).",
        'errors': {
            'load_fail_header': "❌ Model Load Failed",
            'load_fail_help': "Please check the error message above and ensure the model file exists.",
            'file_not_found': "❌ Error: Model file not found at {path}.",
            'file_not_found_help': "Please ensure you have run Step 25 of the V4.43 script, and the 'ad_screening_model_v4_43.joblib' file is in the same folder as this app.py.",
            'load_error': "Error loading model: {e}",
            'predict_error': "An error occurred during prediction:",
            'predict_error_help': "Please check the input data."
        }
    }
}

# --- 2. 初始化会话状态 (Session State) ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'zh' # 默认语言设置为中文

# --- 3. 加载模型和预处理工件 ---
try:
    MODEL_PATH = Path(__file__).parent / "ad_screening_model_v4_43.joblib"
except NameError:
    MODEL_PATH = Path(".") / "ad_screening_model_v4_43.joblib"

@st.cache_resource
def load_artifacts(path):
    """加载 joblib 文件。"""
    T = LANG_STRINGS[st.session_state.lang]['errors'] 
    try:
        artifacts = joblib.load(path)
        return artifacts
    except FileNotFoundError:
        st.error(T['file_not_found'].format(path=path.resolve()))
        st.error(T['file_not_found_help'])
        return None
    except Exception as e:
        st.error(T['load_error'].format(e=e))
        return None

artifacts = load_artifacts(MODEL_PATH)

# --- 4. 定义预测函数 ---
def preprocess_and_predict(input_data, artifacts):
    """
    使用加载的工件对新输入数据进行完整的预处理和预测。
    (此函数内部逻辑不变)
    """
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]
    continuous_cols = artifacts["continuous_cols"]
    binary_cols = artifacts["binary_cols"]
    
    input_df = pd.DataFrame([input_data])
    X_raw = input_df[feature_cols] 
    
    X_imputed_values = imputer.transform(X_raw)
    X_imputed = pd.DataFrame(X_imputed_values, columns=feature_cols, index=X_raw.index)
    
    for col in binary_cols:
        if col in X_imputed.columns:
            X_imputed[col] = X_imputed[col].round().astype(int)
            
    X_scaled = X_imputed.copy()
    if continuous_cols:
        cols_to_scale = [col for col in continuous_cols if col in X_scaled.columns]
        if cols_to_scale:
            X_scaled[cols_to_scale] = scaler.transform(X_imputed[cols_to_scale])
    
    probability = artifacts["model"].predict_proba(X_scaled)[:, 1]
    
    return probability[0] 

# --- 5. 构建 Streamlit 用户界面 ---
def main_app():
    # --- 5.1 设置语言 ---
    st.sidebar.radio(
        label=LANG_STRINGS['zh']['lang_select'], 
        options=['zh', 'en'],
        format_func=lambda x: "中文" if x == 'zh' else "English",
        key='lang', 
        horizontal=True
    )
    
    lang = st.session_state.lang
    T = LANG_STRINGS[lang]

    # --- 5.2 检查模型是否加载成功 ---
    if artifacts is None:
        st.header(T['errors']['load_fail_header'])
        st.write(T['errors']['load_fail_help'])
        return

    threshold = artifacts["optimal_threshold"]
    model_name = artifacts["model_name"]
    
    st.set_page_config(layout="wide", page_title=T['page_title'])
    st.title(T['app_title'])
    st.markdown(T['model_info'].format(model_name=model_name, threshold=threshold))
    st.markdown("---")

    # --- 5.3 侧边栏输入 [V5 变更] ---
    st.sidebar.header(T['sidebar_header'])
    st.sidebar.markdown(T['sidebar_help'])
    
    input_features = {} 
    T_FEATURES = T['features'] 

    # --- 连续特征 (4) ---
    st.sidebar.subheader(T['subheader_continuous'])
    input_features['age'] = st.sidebar.number_input(label=T_FEATURES['age'], min_value=18, max_value=100, value=65)
    input_features['BMI'] = st.sidebar.number_input(label=T_FEATURES['BMI'], min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    input_features['ABO'] = st.sidebar.number_input(label=T_FEATURES['ABO'], min_value=0.0, value=50.0, step=1.0)
    input_features['edu'] = st.sidebar.number_input(label=T_FEATURES['edu'], min_value=0, max_value=30, value=12)

    # --- 二分类特征 (8) [V5 逻辑更新] ---
    st.sidebar.subheader(T['subheader_binary'])
    
    # 性别
    gender_map = T['gender_map'] 
    gender_choice = st.sidebar.selectbox(T_FEATURES['gender'], options=gender_map.keys())
    input_features['gender'] = gender_map[gender_choice]
    
    # 经历类特征 (有/无)
    map_history = T['binary_map_history']
    
    choice_alcohol = st.sidebar.selectbox(T_FEATURES['alcohol'], options=map_history.keys())
    input_features['alcohol'] = map_history[choice_alcohol]
    
    choice_dementia = st.sidebar.selectbox(T_FEATURES['dementia_family_history'], options=map_history.keys())
    input_features['dementia_family_history'] = map_history[choice_dementia]

    # 状态类特征 (是/否)
    map_status = T['binary_map_status']
    
    choice_hypertension = st.sidebar.selectbox(T_FEATURES['hypertension'], options=map_status.keys())
    input_features['hypertension'] = map_status[choice_hypertension]
    
    choice_diabetes = st.sidebar.selectbox(T_FEATURES['diabetes'], options=map_status.keys())
    input_features['diabetes'] = map_status[choice_diabetes]
    
    choice_hyperlipidemia = st.sidebar.selectbox(T_FEATURES['hyperlipidemia'], options=map_status.keys())
    input_features['hyperlipidemia'] = map_status[choice_hyperlipidemia]
    
    choice_apoe = st.sidebar.selectbox(T_FEATURES['APOE4_carrier'], options=map_status.keys())
    input_features['APOE4_carrier'] = map_status[choice_apoe]
    
    choice_gds = st.sidebar.selectbox(T_FEATURES['GDS_DIA'], options=map_status.keys())
    input_features['GDS_DIA'] = map_status[choice_gds]
    
    # --- 5.4 主面板显示 ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(T['input_summary'])
        st.markdown(T['input_help'])
        
        display_labels = T['features']
        
        # [V5] 优化显示逻辑，以正确显示中文选项
        display_data = []
        for key, value in input_features.items():
            label = display_labels[key]
            # 特殊处理二分类的显示值
            display_value = value
            if key == 'gender':
                # 反向查找 map 的键
                display_value = next((k for k, v in gender_map.items() if v == value), value)
            elif key in ['alcohol', 'dementia_family_history']:
                display_value = next((k for k, v in map_history.items() if v == value), value)
            elif key in ['hypertension', 'diabetes', 'hyperlipidemia', 'APOE4_carrier', 'GDS_DIA']:
                display_value = next((k for k, v in map_status.items() if v == value), value)
            
            display_data.append({
                T['input_table_cols']['feature']: label,
                T['input_table_cols']['value']: display_value
            })
            
        input_df_display = pd.DataFrame(display_data).set_index(T['input_table_cols']['feature'])
        st.dataframe(input_df_display)

    with col2:
        st.subheader(T['results_header'])
        
        # --- 5.5 预测按钮 ---
        if st.button(T['predict_button'], type="primary", use_container_width=True):
            
            try:
                probability = preprocess_and_predict(input_features, artifacts)
                
                if probability >= threshold:
                    classification = T['results_risk_high']
                    delta_text = T['results_delta_high'].format(threshold=threshold)
                    st.error(f"{T['results_recommendation']} {classification}")
                else:
                    classification = T['results_risk_low']
                    delta_text = T['results_delta_low'].format(threshold=threshold)
                    st.success(f"{T['results_recommendation']} {classification}")

                st.metric(
                    label=T['results_metric_label'],
                    value=f"{probability:.2%}",
                    delta=delta_text,
                    delta_color="inverse" if probability >= threshold else "normal"
                )
                
                st.progress(probability)
                st.caption(T['results_caption'].format(probability=probability))
                
            except Exception as e:
                st.error(T['errors']['predict_error'])
                st.exception(e)
                st.error(T['errors']['predict_error_help'])

# --- 6. 运行 App ---
if __name__ == "__main__":
    main_app()
