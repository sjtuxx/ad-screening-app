import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import shap # 导入 SHAP

# --- 1. 语言和文本内容 (LANG_STRINGS) ---
# [V7] 更新了错误提示
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
            'age': '年龄 (岁)', 'BMI': 'BMI (kg/m²)', 'ABO': '血清Aβ寡聚体 (ABO)', 'edu': '教育年限 (年)',
            'gender': '性别', 'alcohol': '饮酒史', 'dementia_family_history': '痴呆家族史',
            'hypertension': '高血压', 'diabetes': '糖尿病', 'hyperlipidemia': '高血脂',
            'APOE4_carrier': 'APOE ε4 携带状态', 'GDS_DIA': '抑郁症状 (GDS)'
        },
        'gender_map': {'女性': 0, '男性': 1},
        'binary_map_status': {'否': 0, '是': 1}, 
        'binary_map_history': {'无': 0, '有': 1}, 
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
        'shap_expander': "📊 显示/隐藏 个体预测归因 (SHAP 分析)",
        'shap_help': "下图显示了每个特征如何将预测概率从基线值（{base_value:.2f}）推动到最终值（{probability:.2f}）。",
        'shap_help_red': "**红色特征** (如 年龄) 推动预测**增加**风险。",
        'shap_help_blue': "**蓝色特征** (如 教育年限) 推动预测**降低**风险。",
        'errors': {
            'load_fail_header': "❌ 模型加载失败",
            'load_fail_help': "请检查下方的错误信息并确保模型文件存在。",
            'file_not_found': "❌ 错误：在 {path} 未找到模型文件。",
            'file_not_found_help': "请确保您已运行 V4.43 脚本的步骤 25 (V7版)，并且 'ad_screening_model_v4_43_with_shap_data.joblib' 文件与此 app.py 在同一个文件夹中。", # [V7] 更新了文件名
            'load_error': "加载模型时出错： {e}",
            'predict_error': "预测过程中发生错误：",
            'predict_error_help': "请检查输入数据。",
            'shap_error': "SHAP 背景数据加载失败。请确保您使用了 V7 版本的步骤 25 来重新生成 .joblib 文件。", # [V7] 更新了错误
            'shap_create_error': "创建 SHAP 分析器时出错："
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
            'age': 'Age (years)', 'BMI': 'BMI (kg/m²)', 'ABO': 'Serum Aβ Oligomers (ABO)', 'edu': 'Education (years)',
            'gender': 'Sex', 'alcohol': 'Alcohol Use', 'dementia_family_history': 'Family History of Dementia',
            'hypertension': 'Hypertension', 'diabetes': 'Diabetes Mellitus', 'hyperlipidemia': 'Hyperlipidemia',
            'APOE4_carrier': 'APOE ε4 Carrier Status', 'GDS_DIA': 'Depressive Symptoms (GDS)'
        },
        'gender_map': {'Female': 0, 'Male': 1},
        'binary_map_status': {'No': 0, 'Yes': 1},
        'binary_map_history': {'No': 0, 'Yes': 1},
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
        'shap_expander': "📊 Show/Hide Individual Prediction Attribution (SHAP Analysis)",
        'shap_help': "The plot below shows how each feature pushed the prediction from the base value ({base_value:.2f}) to the final value ({probability:.2f}).",
        'shap_help_red': "**Red features** (e.g., Age) pushed the prediction to **increase** risk.",
        'shap_help_blue': "**Blue features** (e.g., Education) pushed the prediction to **decrease** risk.",
        'errors': {
            'load_fail_header': "❌ Model Load Failed",
            'load_fail_help': "Please check the error message above and ensure the model file exists.",
            'file_not_found': "❌ Error: Model file not found at {path}.",
            'file_not_found_help': "Please ensure you have run Step 25 (V7) of the V4.43 script, and 'ad_screening_model_v4_43_with_shap_data.joblib' is in the same folder as app.py.", # [V7]
            'load_error': "Error loading model: {e}",
            'predict_error': "An error occurred during prediction:",
            'predict_error_help': "Please check the input data.",
            'shap_error': "SHAP background data failed to load. Please ensure you regenerated the .joblib file using Step 25 (V7).", # [V7]
            'shap_create_error': "Error creating SHAP Explainer:"
        }
    }
}

# --- 2. 初始化会话状态 (Session State) ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'zh' 

# --- 3. 加载模型和预处理工件 ---
# [V7] 更改了模型文件名
try:
    MODEL_PATH = Path(__file__).parent / "ad_screening_model_v4_43_with_shap_data.joblib"
except NameError:
    MODEL_PATH = Path(".") / "ad_screening_model_v4_43_with_shap_data.joblib"

@st.cache_resource
def load_artifacts(path):
    T = LANG_STRINGS[st.session_state.lang]['errors'] 
    try:
        artifacts = joblib.load(path)
        # [V7] 检查 'shap_background_data' 是否存在
        if 'shap_background_data' not in artifacts:
             st.error(T['shap_error'])
             return None
        return artifacts
    except FileNotFoundError:
        st.error(T['file_not_found'].format(path=path.resolve()))
        st.error(T['file_not_found_help'])
        return None
    except Exception as e:
        st.error(T['load_error'].format(e=e))
        return None

artifacts = load_artifacts(MODEL_PATH)

# --- 4. [V7 新增] 实时创建并缓存 Explainer ---
@st.cache_resource
def create_explainer_and_base_value(_artifacts):
    """
    在应用启动时运行一次，使用云端的 SHAP 库版本创建 Explainer。
    """
    T = LANG_STRINGS[st.session_state.lang]['errors']
    try:
        model = _artifacts['model']
        background_data = _artifacts['shap_background_data']
        
        # [V7 关键变更] 在此实时创建 explainer
        explainer = shap.TreeExplainer(model, background_data)
        
        # [V7] 在此获取基线值
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value_class1 = explainer.expected_value[1]
        else:
            base_value_class1 = explainer.expected_value 
            
        return explainer, base_value_class1
    except Exception as e:
        st.error(f"{T['shap_create_error']} {e}")
        return None, None

# --- 5. 定义预测函数 ---
@st.cache_data(show_spinner=False)
def preprocess_data(input_data, _artifacts):
    """
    仅执行预处理，返回可用于模型和 SHAP 的 X_scaled。
    """
    imputer = _artifacts["imputer"]
    scaler = _artifacts["scaler"]
    feature_cols = _artifacts["feature_cols"]
    continuous_cols = _artifacts["continuous_cols"]
    binary_cols = _artifacts["binary_cols"]
    
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
    
    return X_scaled

# --- 6. 构建 Streamlit 用户界面 ---
def main_app():
    # --- 6.1 设置语言 ---
    st.sidebar.radio(
        label=LANG_STRINGS['zh']['lang_select'], 
        options=['zh', 'en'],
        format_func=lambda x: "中文" if x == 'zh' else "English",
        key='lang', 
        horizontal=True
    )
    lang = st.session_state.lang
    T = LANG_STRINGS[lang]

    # --- 6.2 检查模型是否加载成功 ---
    if artifacts is None:
        st.header(T['errors']['load_fail_header'])
        st.write(T['errors']['load_fail_help'])
        return

    # [V7 新增] 加载 Explainer 和 Base Value
    explainer, base_value_class1 = create_explainer_and_base_value(artifacts)
    if explainer is None:
        return # 如果 explainer 创建失败，则停止

    threshold = artifacts["optimal_threshold"]
    model_name = artifacts["model_name"]
    
    st.set_page_config(layout="wide", page_title=T['page_title'])
    st.title(T['app_title'])
    st.markdown(T['model_info'].format(model_name=model_name, threshold=threshold))
    st.markdown("---")

    # --- 6.3 侧边栏输入 ---
    st.sidebar.header(T['sidebar_header'])
    st.sidebar.markdown(T['sidebar_help'])
    
    input_features = {} 
    T_FEATURES = T['features'] 

    # 连续特征
    st.sidebar.subheader(T['subheader_continuous'])
    input_features['age'] = st.sidebar.number_input(label=T_FEATURES['age'], min_value=18, max_value=100, value=65)
    input_features['BMI'] = st.sidebar.number_input(label=T_FEATURES['BMI'], min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    input_features['ABO'] = st.sidebar.number_input(label=T_FEATURES['ABO'], min_value=0.0, value=50.0, step=1.0)
    input_features['edu'] = st.sidebar.number_input(label=T_FEATURES['edu'], min_value=0, max_value=30, value=12)

    # 二分类特征
    st.sidebar.subheader(T['subheader_binary'])
    gender_map = T['gender_map'] 
    gender_choice = st.sidebar.selectbox(T_FEATURES['gender'], options=gender_map.keys())
    input_features['gender'] = gender_map[gender_choice]
    
    map_history = T['binary_map_history']
    choice_alcohol = st.sidebar.selectbox(T_FEATURES['alcohol'], options=map_history.keys())
    input_features['alcohol'] = map_history[choice_alcohol]
    choice_dementia = st.sidebar.selectbox(T_FEATURES['dementia_family_history'], options=map_history.keys())
    input_features['dementia_family_history'] = map_history[choice_dementia]

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
    
    # --- 6.4 主面板显示 ---
    col1, col2 = st.columns([1, 2])
    
    display_labels = T['features']
    display_data_list = []
    
    for key, value in input_features.items():
        label = display_labels[key]
        display_value = value
        if key == 'gender':
            display_value = next((k for k, v in gender_map.items() if v == value), value)
        elif key in ['alcohol', 'dementia_family_history']:
            display_value = next((k for k, v in map_history.items() if v == value), value)
        elif key in ['hypertension', 'diabetes', 'hyperlipidemia', 'APOE4_carrier', 'GDS_DIA']:
            display_value = next((k for k, v in map_status.items() if v == value), value)
        display_data_list.append({
            'label': label,
            'value': display_value,
            'original_value': value
        })
    
    display_df_for_table = pd.DataFrame(display_data_list).set_index('label')[['value']]
    display_df_for_table.index.name = T['input_table_cols']['feature']
    display_df_for_table.columns = [T['input_table_cols']['value']]
    
    # [V7] 为 SHAP 创建有序的输入 (原始值 和 标签)
    shap_features = pd.Series([d['original_value'] for d in display_data_list], index=[d['label'] for d in display_data_list])
    
    with col1:
        st.subheader(T['input_summary'])
        st.markdown(T['input_help'])
        st.dataframe(display_df_for_table)

    with col2:
        st.subheader(T['results_header'])
        
        # --- 6.5 预测按钮和 SHAP 分析 [V8 修复] ---
        if st.button(T['predict_button'], type="primary", use_container_width=True):
            
            try:
                # --- A. 预处理 ---
                X_scaled = preprocess_data(input_features, artifacts)
                
                # --- B. 模型预测 ---
                model = artifacts["model"]
                probability = model.predict_proba(X_scaled)[:, 1][0]
                
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
                
                # --- C. [V8 修复] SHAP 分析 ---
                with st.expander(T['shap_expander']):
                    st.markdown("---")
                    
                    # 1. [V7] explainer 已经加载
                    shap_values = explainer.shap_values(X_scaled)
                    
                    # 2. [V8 修复] 检查 shap_values 是列表(size 2)还是单个数组
                    #    X_scaled 是单个样本, shape (1, 12)
                    
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        # 正常情况：返回 [shap_class_0, shap_class_1]
                        # shap_values[1] 是 class 1 的数组, shape (1, 12)
                        # 我们需要第一个 (也是唯一一个) 样本 [0]
                        shap_values_class1_single_sample = shap_values[1][0]
                    
                    elif isinstance(shap_values, np.ndarray) and shap_values.shape[0] == 1:
                        # 异常情况：只返回了一个数组, shape (1, 12)
                        # 我们假设这就是 class 1, 并获取第一个 (也是唯一一个) 样本 [0]
                        shap_values_class1_single_sample = shap_values[0]
                    
                    else:
                        # 捕获其他意外格式, 例如 list[1]
                        try:
                            # 尝试假设它是一个单元素列表
                            st.warning("SHAP analysis returned an unexpected list format. Attempting to parse.")
                            shap_values_class1_single_sample = shap_values[0][0]
                        except Exception:
                            st.error(f"SHAP analysis returned an unhandled format: {type(shap_values)}")
                            raise # 重新引发错误，停止执行

                    st.markdown(T['shap_help'].format(base_value=base_value_class1, probability=probability))
                    st.markdown(T['shap_help_red'])
                    st.markdown(T['shap_help_blue'])
                    
                    # 3. 绘制 SHAP 力图 (Force Plot)
                    st.shap(shap.force_plot(
                        base_value=base_value_class1,
                        shap_values=shap_values_class1_single_sample, # <--- [V8] 使用修复后的变量
                        features=shap_features.values, 
                        feature_names=shap_features.index 
                    ), height=150, width=800)
                    
            except Exception as e:
                st.error(T['errors']['predict_error'])
                st.exception(e)
                st.error(T['errors']['predict_error_help'])

# --- 7. 运行 App ---
if __name__ == "__main__":
    main_app()
