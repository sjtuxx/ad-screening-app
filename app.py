import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. 语言和文本内容 (LANG_STRINGS) ---
# 包含所有界面文本的双语字典
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
            'gender': '性别',
            'alcohol': '饮酒史',
            'dementia_family_history': '痴呆家族史',
            'hypertension': '高血压',
            'diabetes': '糖尿病',
            'hyperlipidemia': '高血脂',
            'APOE4_carrier': 'APOE ε4 携带状态',
            'GDS_DIA': '抑郁症状 (GDS)'
        },
        'gender_map': {'女性 (0)': 0, '男性 (1)': 1},
        'binary_map': {'否 (0)': 0, '是 (1)': 1},
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
        'gender_map': {'Female (0)': 0, 'Male (1)': 1},
        'binary_map': {'No (0)': 0, 'Yes (1)': 1},
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
# 这必须在顶部，以便 'load_artifacts' 可以访问它
if 'lang' not in st.session_state:
    st.session_state.lang = 'zh' # 默认语言设置为中文

# --- 3. 加载模型和预处理工件 ---

# (重要!) 使用相对路径
try:
    MODEL_PATH = Path(__file__).parent / "ad_screening_model_v4_43.joblib"
except NameError:
    MODEL_PATH = Path(".") / "ad_screening_model_v4_43.joblib"

@st.cache_resource
def load_artifacts(path):
    """加载 joblib 文件。"""
    # 在函数内部获取语言，以便错误信息是正确的语言
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

# 加载工件
artifacts = load_artifacts(MODEL_PATH)

# --- 4. 定义预测函数 ---
# (此函数内部逻辑不变，无需翻译)
def preprocess_and_predict(input_data, artifacts):
    """
    使用加载的工件对新输入数据进行完整的预处理和预测。
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
    # (注意: 'lang' 已经在脚本顶部被 st.session_state 初始化)
    
    # 在侧边栏顶部添加语言选择器
    # `key='lang'` 将此小部件直接绑定到会话状态
    st.sidebar.radio(
        label=LANG_STRINGS['zh']['lang_select'], # 标签始终显示双语
        options=['zh', 'en'],
        format_func=lambda x: "中文" if x == 'zh' else "English",
        key='lang', # 关键！
        horizontal=True
    )
    
    # 获取当前选择的语言文本
    lang = st.session_state.lang
    T = LANG_STRINGS[lang]

    # --- 5.2 检查模型是否加载成功 ---
    if artifacts is None:
        st.header(T['errors']['load_fail_header'])
        st.write(T['errors']['load_fail_help'])
        return

    # 从工件中获取关键信息
    threshold = artifacts["optimal_threshold"]
    model_name = artifacts["model_name"]
    
    # 页面设置 (使用翻译)
    st.set_page_config(layout="wide", page_title=T['page_title'])
    st.title(T['app_title'])
    st.markdown(T['model_info'].format(model_name=model_name, threshold=threshold))
    st.markdown("---")

    # --- 5.3 侧边栏输入 ---
    st.sidebar.header(T['sidebar_header'])
    st.sidebar.markdown(T['sidebar_help'])
    
    input_features = {} # 用于存储输入的字典
    T_FEATURES = T['features'] # 获取特征标签字典

    # --- 连续特征 (4) ---
    st.sidebar.subheader(T['subheader_continuous'])
    input_features['age'] = st.sidebar.number_input(label=T_FEATURES['age'], min_value=18, max_value=100, value=65)
    input_features['BMI'] = st.sidebar.number_input(label=T_FEATURES['BMI'], min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    input_features['ABO'] = st.sidebar.number_input(label=T_FEATURES['ABO'], min_value=0.0, value=50.0, step=1.0)
    input_features['edu'] = st.sidebar.number_input(label=T_FEATURES['edu'], min_value=0, max_value=30, value=12)

    # --- 二分类特征 (8) ---
    st.sidebar.subheader(T['subheader_binary'])
    
    gender_map = T['gender_map'] # 使用翻译后的 map
    gender_choice = st.sidebar.selectbox(T_FEATURES['gender'], options=gender_map.keys())
    input_features['gender'] = gender_map[gender_choice]
    
    binary_map = T['binary_map'] # 使用翻译后的 map
    
    def create_binary_input(key):
        """辅助函数，用于创建二分类的 selectbox"""
        label = T_FEATURES[key] 
        choice = st.sidebar.selectbox(label, options=binary_map.keys())
        return binary_map[choice]

    input_features['alcohol'] = create_binary_input('alcohol')
    input_features['dementia_family_history'] = create_binary_input('dementia_family_history')
    input_features['hypertension'] = create_binary_input('hypertension')
    input_features['diabetes'] = create_binary_input('diabetes')
    input_features['hyperlipidemia'] = create_binary_input('hyperlipidemia')
    input_features['APOE4_carrier'] = create_binary_input('APOE4_carrier')
    input_features['GDS_DIA'] = create_binary_input('GDS_DIA')
    
    # --- 5.4 主面板显示 ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(T['input_summary'])
        st.markdown(T['input_help'])
        
        # 将输入特征 (英文键) 映射到中文标签
        display_labels = T['features']
        input_df_display = pd.DataFrame([input_features])
        input_df_display = input_df_display.rename(columns=display_labels).T
        input_df_display.columns = [T['input_table_cols']['value']]
        input_df_display.index.name = T['input_table_cols']['feature']
        st.dataframe(input_df_display)

    with col2:
        st.subheader(T['results_header'])
        
        # --- 5.5 预测按钮 ---
        if st.button(T['predict_button'], type="primary", use_container_width=True):
            
            try:
                # 1. 调用预测函数
                probability = preprocess_and_predict(input_features, artifacts)
                
                # 2. 根据阈值确定分类 (使用翻译)
                if probability >= threshold:
                    classification = T['results_risk_high']
                    delta_text = T['results_delta_high'].format(threshold=threshold)
                    st.error(f"{T['results_recommendation']} {classification}")
                else:
                    classification = T['results_risk_low']
                    delta_text = T['results_delta_low'].format(threshold=threshold)
                    st.success(f"{T['results_recommendation']} {classification}")

                # 3. 显示概率计量表 (使用翻译)
                st.metric(
                    label=T['results_metric_label'],
                    value=f"{probability:.2%}",
                    delta=delta_text,
                    delta_color="inverse" if probability >= threshold else "normal"
                )
                
                # 4. 显示概率条
                st.progress(probability)
                st.caption(T['results_caption'].format(probability=probability))
                
            except Exception as e:
                st.error(T['errors']['predict_error'])
                st.exception(e) # (可选) 打印详细的错误堆栈
                st.error(T['errors']['predict_error_help'])

# --- 6. 运行 App ---
if __name__ == "__main__":
    main_app()
