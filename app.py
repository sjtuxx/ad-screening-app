import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. 加载模型和预处理工件 ---

# 定义模型路径 (请根据需要调整)
# 假设 'app.py' 和 'ad_screening_model_v4_43.joblib' 都在 'E:/ABO_ML/ML' 文件夹中
MODEL_PATH = Path("E:/ABO_ML/ML/ad_screening_model_v4_43.joblib") 
# 或者使用相对路径 (如果 app.py 在 ML 文件夹下):
# MODEL_PATH = Path(__file__).parent / "ad_screening_model_v4_43.joblib"

@st.cache_resource
def load_artifacts(path):
    """加载 joblib 文件。使用 Streamlit 缓存以提高效率。"""
    try:
        artifacts = joblib.load(path)
        return artifacts
    except FileNotFoundError:
        st.error(f"错误: 未在 {path} 找到模型文件。")
        st.error("请确保您已经运行了 V4.43 脚本的步骤 25 (模型永久化)。")
        return None
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None

# 加载工件
artifacts = load_artifacts(MODEL_PATH)

# --- 2. 定义预测函数 ---

def preprocess_and_predict(input_data, artifacts):
    """
    使用加载的工件对新输入数据进行完整的预处理和预测。
    这必须 *精确* 匹配 V4.43 脚本中的 步骤 4 和 步骤 7.9。
    """
    
    # 从工件中解包
    imputer = artifacts["imputer"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]
    continuous_cols = artifacts["continuous_cols"]
    binary_cols = artifacts["binary_cols"]
    
    # 1. 将单一样本的输入字典转换为 DataFrame (只有一行)
    # 确保列的顺序与训练时 *完全* 一致
    input_df = pd.DataFrame([input_data])
    X_raw = input_df[feature_cols] # 保证顺序
    
    # --- 开始重现 V4.43 脚本中的 步骤 4 ---
    
    # 2. 插补 (Imputation)
    X_imputed_values = imputer.transform(X_raw)
    X_imputed = pd.DataFrame(X_imputed_values, columns=feature_cols, index=X_raw.index)
    
    # 3. [关键] 对二分类变量进行四舍五入
    for col in binary_cols:
        if col in X_imputed.columns:
            X_imputed[col] = X_imputed[col].round().astype(int)
            
    # 4. 标准化 (Standardization)
    X_scaled = X_imputed.copy()
    if continuous_cols:
        cols_to_scale = [col for col in continuous_cols if col in X_scaled.columns]
        if cols_to_scale:
            X_scaled[cols_to_scale] = scaler.transform(X_imputed[cols_to_scale])
    
    # --- 预处理结束 ---

    # 5. 预测概率 (来自 步骤 7.9)
    # model = artifacts["model"] (即 Random Forest)
    probability = model.predict_proba(X_scaled)[:, 1]
    
    return probability[0] # 返回单个概率值

# --- 3. 构建 Streamlit 用户界面 ---

def main_app():
    if artifacts is None:
        st.header("❌ 模型加载失败")
        st.write("请检查控制台错误信息并确保模型文件存在。")
        return

    # 从工件中获取显示名称和阈值
    display_names = artifacts["feature_display_names"]
    threshold = artifacts["optimal_threshold"]
    model_name = artifacts["model_name"]
    
    st.set_page_config(layout="wide")
    st.title(f"👨‍⚕️ 阿尔茨海默病 (AD) 机器学习筛选工具")
    st.markdown(f"**模型版本:** `{model_name}` (基于 V4.43 脚本) | **分类阈值 (Youden Index):** `{threshold:.4f}`")
    st.markdown("---")

    # --- 3.1 侧边栏输入 ---
    st.sidebar.header("患者信息输入")
    st.sidebar.markdown("请输入以下 12 项特征：")
    
    input_features = {}

    # --- 连续特征 (4) ---
    st.sidebar.subheader("连续特征")
    input_features['age'] = st.sidebar.number_input(label=display_names['age'], min_value=18, max_value=100, value=65)
    input_features['BMI'] = st.sidebar.number_input(label=display_names['BMI'], min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    input_features['ABO'] = st.sidebar.number_input(label=display_names['ABO'], min_value=0.0, value=50.0, step=1.0)
    input_features['edu'] = st.sidebar.number_input(label=display_names['edu'], min_value=0, max_value=30, value=12)

    # --- 二分类特征 (8) ---
    st.sidebar.subheader("二分类特征")
    
    # (注意: 'gender' 在您的脚本中是 1=Male)
    gender_map = {'女性 (0)': 0, '男性 (1)': 1}
    gender_choice = st.sidebar.selectbox(display_names['gender'], options=gender_map.keys())
    input_features['gender'] = gender_map[gender_choice]
    
    binary_map = {'否 (0)': 0, '是 (1)': 1}
    
    def create_binary_input(key):
        """辅助函数，用于创建二分类的 selectbox"""
        label = display_names[key]
        choice = st.sidebar.selectbox(label, options=binary_map.keys())
        return binary_map[choice]

    input_features['alcohol'] = create_binary_input('alcohol')
    input_features['dementia_family_history'] = create_binary_input('dementia_family_history')
    input_features['hypertension'] = create_binary_input('hypertension')
    input_features['diabetes'] = create_binary_input('diabetes')
    input_features['hyperlipidemia'] = create_binary_input('hyperlipidemia')
    input_features['APOE4_carrier'] = create_binary_input('APOE4_carrier')
    input_features['GDS_DIA'] = create_binary_input('GDS_DIA')
    
    # --- 3.2 主面板显示 ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("模型输入总览")
        st.markdown("请在左侧侧边栏中修改值。")
        # 将输入特征转换为带有显示名称的 DataFrame 以便查看
        input_df_display = pd.DataFrame([input_features])
        input_df_display = input_df_display.rename(columns=display_names).T
        input_df_display.columns = ["输入值"]
        st.dataframe(input_df_display)

    with col2:
        st.subheader("📈 预测结果")
        
        # --- 3.3 预测按钮 ---
        if st.button("运行模型预测", type="primary", use_container_width=True):
            
            # 1. 调用预测函数
            try:
                probability = preprocess_and_predict(input_features, artifacts)
                
                # 2. 根据阈值确定分类
                if probability >= threshold:
                    classification = "高风险 (High Risk)"
                    delta_text = f"高于阈值 {threshold:.4f}"
                    st.error(f"**诊断建议:** {classification}")
                else:
                    classification = "低风险 (Low Risk)"
                    delta_text = f"低于阈值 {threshold:.4f}"
                    st.success(f"**诊断建议:** {classification}")

                # 3. 显示概率计量表
                st.metric(
                    label=f"MCI/AD 预测概率",
                    value=f"{probability:.2%}",
                    delta=delta_text,
                    delta_color="inverse" if classification == "高风险 (High Risk)" else "normal"
                )
                
                # 4. 显示概率条
                st.progress(probability)
                st.caption(f"该概率值 ({probability:.4f}) 表示模型预测个体为认知受损 (MCI/AD) 的可能性。")
                
                # 5. [V4.43 修复] 显示正确的混淆矩阵图
                st.markdown("---")
                st.subheader("模型性能参考 (来自 V4.43, 步骤 13)")
                st.markdown(f"以下是 `{model_name}` 模型在*测试集*上使用*最佳阈值* ({threshold:.2f}) 时的混淆矩阵。")
                
                # (您需要将 V4.43 脚本生成的 'confusion_matrix_OPTIMAL_Random_Forest.pdf' 转换为 .png 格式)
                # (并将该 .png 文件放在与 app.py 相同的文件夹中)
                cm_image_path = Path("E:/ABO_ML/ML/confusion_matrix_OPTIMAL_Random_Forest.png")
                if cm_image_path.exists():
                    st.image(str(cm_image_path), caption="图 3A：基于最佳阈值的混淆矩阵 (测试集)")
                else:
                    st.warning("未找到 'confusion_matrix_OPTIMAL_Random_Forest.png' 图像文件。")
                    st.markdown(f"请将 V4.43 脚本在 `{output_dir}` 中生成的 `confusion_matrix_OPTIMAL_Random_Forest.pdf` 转换为 **PNG** 格式并保存在 `{cm_image_path.parent}` 文件夹中。")

            except Exception as e:
                st.error(f"预测过程中发生错误: {e}")
                st.error("请检查输入数据，特别是缺失值。")


# --- 4. 运行 App ---
if __name__ == "__main__":
    main_app()