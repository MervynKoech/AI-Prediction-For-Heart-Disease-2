# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Heart Disease AI Dashboard", page_icon="❤️", layout="wide")

# --- Cache Resource Loading ---
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    metrics = joblib.load("metrics.pkl")
    test_artifacts = joblib.load("test_artifacts.pkl")
    
    # --- CRITICAL FIX: Recursively patch SimpleImputer to fix scikit-learn 1.4+ pickle compatibility ---
    def patch_sklearn_objects(obj):
        if isinstance(obj, (list, tuple)):
            for item in obj:
                patch_sklearn_objects(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                patch_sklearn_objects(value)
        elif hasattr(obj, '__dict__'):
            # Fix for SimpleImputer missing _fill_dtype in scikit-learn 1.4+
            if obj.__class__.__name__ == 'SimpleImputer' and not hasattr(obj, '_fill_dtype'):
                if hasattr(obj, 'statistics_'):
                    obj._fill_dtype = obj.statistics_.dtype
                else:
                    obj._fill_dtype = np.dtype('float64')
            
            for key, value in obj.__dict__.items():
                patch_sklearn_objects(value)

    # Apply the patch to all loaded models and the standalone preprocessor
    patch_sklearn_objects(rf_model)
    patch_sklearn_objects(xgb_model)
    patch_sklearn_objects(preprocessor)
    
    # Force numpy output to prevent pandas dtype bugs
    try:
        rf_model.named_steps['preprocessor'].set_output(transform="default")
        xgb_model.named_steps['preprocessor'].set_output(transform="default")
        preprocessor.set_output(transform="default")
    except Exception:
        pass
        
    return rf_model, xgb_model, preprocessor, metrics, test_artifacts

rf_model, xgb_model, preprocessor, metrics, test_artifacts = load_artifacts()

# --- Sidebar Navigation ---
st.sidebar.title("❤️ CardioAI Navigator")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "🏠 Dashboard Overview",
    "📈 Model Performance",
    "🩺 New Patient Prediction",
    "🔍 Explainable AI (XAI)",
    "📖 Dataset & Feature Info"
])

# --- Page 1: Dashboard Overview ---
if page == "🏠 Dashboard Overview":
    st.title("Welcome to CardioAI Dashboard")
    st.markdown("""
    This professional dashboard leverages advanced Machine Learning and Explainable AI (XAI) to assess the risk of Atherosclerotic Heart Disease (AHD). 
    
    ### 🚀 Key Features:
    - **Dual Model Support**: Choose between optimized **Random Forest** and **XGBoost** models.
    - **Real-time Prediction**: Input patient vitals to get instant risk assessments.
    - **Clinical Recommendations**: Actionable advice based on probability thresholds (0%-30%, 31%-50%, 51%-70%, 71%-100%).
    - **Explainable AI**: SHAP-based local and global feature importance to ensure transparent, trustworthy clinical decisions.
    """)
    st.image("https://img.freepik.com/free-vector/heart-rate-monitor-concept-illustration_114360-1524.jpg", width="stretch")

# --- Page 2: Model Performance ---
# --- Page 2: Model Performance ---
elif page == "📈 Model Performance":
    st.title("Model Performance & Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Performance Metrics")
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.format("{:.4f}").background_gradient(cmap="Blues"), width="stretch")
    
    with col2:
        st.subheader("📈 ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        
        # --- FIX: Strictly cast to int/float to eliminate all dtype ambiguities ---
        y_test_vals = np.ravel(test_artifacts["y_test"]).astype(int)
        y_prob_rf_vals = np.ravel(test_artifacts["y_prob_rf"]).astype(float)
        y_prob_xgb_vals = np.ravel(test_artifacts["y_prob_xgb"]).astype(float)
        
        # Use core roc_curve function to bypass RocCurveDisplay bugs in scikit-learn 1.4+
        from sklearn.metrics import roc_curve, auc
        
        fpr_rf, tpr_rf, _ = roc_curve(y_test_vals, y_prob_rf_vals)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        ax_roc.plot(fpr_rf, tpr_rf, color="blue", lw=2, label=f"Random Forest (AUC = {roc_auc_rf:.4f})")
        
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test_vals, y_prob_xgb_vals)
        roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
        ax_roc.plot(fpr_xgb, tpr_xgb, color="green", lw=2, label=f"XGBoost (AUC = {roc_auc_xgb:.4f})")
        
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic (ROC)")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.subheader("🎯 Confusion Matrix")
    model_cm = st.selectbox("Select Model for Confusion Matrix", ["Random Forest", "XGBoost"])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    
    # Strictly cast predictions to int
    y_pred_rf_vals = np.ravel(test_artifacts["y_pred_rf"]).astype(int)
    y_pred_xgb_vals = np.ravel(test_artifacts["y_pred_xgb"]).astype(int)
    
    # Use core confusion_matrix + Seaborn to bypass ConfusionMatrixDisplay bugs
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    if model_cm == "Random Forest":
        cm = confusion_matrix(y_test_vals, y_pred_rf_vals)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, cbar=False)
    else:
        cm = confusion_matrix(y_test_vals, y_pred_xgb_vals)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax_cm, cbar=False)
        
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title(f"Confusion Matrix - {model_cm}")
    st.pyplot(fig_cm)
# --- Page 3: New Patient Prediction ---
elif page == "🩺 New Patient Prediction":
    st.title("New Patient Risk Assessment")
    st.markdown("Enter the patient's clinical data below to generate a heart disease risk prediction.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type", ["typical", "nontypical", "nonanginal", "asymptomatic"])
            rest_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
            chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=500, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
            rest_ecg = st.selectbox("Resting ECG", [0, 1, 2])
            
        with col2:
            max_hr = st.number_input("Maximum Heart Rate (bpm)", min_value=60, max_value=220, value=150)
            ex_ang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("Peak Exercise ST Segment Slope", [1, 2, 3])
            ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thalassemia Stress Test", ["normal", "fixed", "reversable"])
            
        st.markdown("---")
        model_choice = st.selectbox("Select Prediction Model", ["Random Forest", "XGBoost"])
        submit = st.form_submit_button("🔮 Predict Risk", type="primary")

    if submit:
        input_data = pd.DataFrame({
            "Age": [age], "Sex": [1 if sex == "Male" else 0], "ChestPain": [chest_pain],
            "RestBP": [rest_bp], "Chol": [chol], "Fbs": [1 if fbs == "Yes" else 0],
            "RestECG": [rest_ecg], "MaxHR": [max_hr], "ExAng": [1 if ex_ang == "Yes" else 0],
            "Oldpeak": [oldpeak], "Slope": [slope], "Ca": [ca], "Thal": [thal]
        })
        
        feature_order = ["Age", "Sex", "ChestPain", "RestBP", "Chol", "Fbs", "RestECG", "MaxHR", "ExAng", "Oldpeak", "Slope", "Ca", "Thal"]
        input_data = input_data[feature_order]
        
        model = rf_model if model_choice == "Random Forest" else xgb_model
        pred_class = model.predict(input_data)[0]
        pred_prob = model.predict_proba(input_data)[0][1]
        
        st.session_state['latest_input'] = input_data
        st.session_state['latest_model'] = model
        st.session_state['latest_model_name'] = model_choice
        
        st.markdown("---")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Predicted Diagnosis", "🚨 Disease" if pred_class == 1 else "✅ No Disease")
        with col_res2:
            st.metric("Probability of Heart Disease", f"{pred_prob:.2%}")
            
        st.subheader("🩺 Clinical Recommendation")
        if pred_prob <= 0.30:
            st.success("🟢 **Low Risk (0%-30%)**: Your risk of heart disease is low. Maintain a healthy lifestyle with regular exercise and a balanced diet.")
        elif pred_prob <= 0.50:
            st.warning("🟡 **Moderate Risk (31%-50%)**: You have a moderate risk of heart disease. Consider lifestyle modifications such as improving your diet, increasing physical activity, and monitoring your blood pressure and cholesterol.")
        elif pred_prob <= 0.70:
            st.error("🟠 **High Risk (51%-70%)**: You have a high risk of heart disease. It is strongly recommended to consult a healthcare provider for a thorough evaluation and possible preventive measures.")
        else:
            st.error("🔴 **Very High Risk (71%-100%)**: You have a very high risk of heart disease. Immediate medical consultation is highly recommended for further diagnostic testing and intervention.")

# --- Page 4: Explainable AI (XAI) ---
elif page == "🔍 Explainable AI (XAI)":
    st.title("Explainable AI (XAI)")
    st.markdown("Understand how the model makes its predictions using **SHAP** (SHapley Additive exPlanations).")
    
    # ==========================================
    # LOCAL EXPLANATION (Single Patient)
    # ==========================================
    if 'latest_input' in st.session_state:
        st.subheader(f"🔎 Local Explanation for Latest Prediction ({st.session_state['latest_model_name']})")
        input_data = st.session_state['latest_input']
        model = st.session_state['latest_model']
        model_name = st.session_state['latest_model_name']
        
        input_trans = preprocessor.transform(input_data)
        feature_names = preprocessor.get_feature_names_out()
        
        estimator = model.named_steps['xgb'] if model_name == "XGBoost" else model.named_steps['rf']
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(input_trans)
        
        # Robustly extract positive class values and base value
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1][0]
            raw_base_val = explainer.expected_value
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values_pos = shap_values[0, :, 1]
            raw_base_val = explainer.expected_value
        else:
            shap_values_pos = shap_values[0]
            raw_base_val = explainer.expected_value
            
        if isinstance(raw_base_val, (np.ndarray, list)):
            arr = np.array(raw_base_val)
            if arr.size > 1:
                base_val = float(arr[1])
            else:
                base_val = float(arr.squeeze())
        else:
            base_val = float(raw_base_val)
            
        shap_exp = shap.Explanation(values=shap_values_pos, base_values=base_val, data=input_trans[0], feature_names=feature_names)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        st.pyplot(fig)
    else:
        st.info("💡 Please make a prediction in the '🩺 New Patient Prediction' tab to see the local SHAP explanation.")

    st.markdown("---")
    
    # ==========================================
    # GLOBAL EXPLANATION (Entire Test Set)
    # ==========================================
    st.subheader("🌍 Global Feature Importance (SHAP Summary)")
    st.markdown("Overall impact of each feature on the model's predictions across the entire dataset.")
    
    x_test = test_artifacts["x_test"]
    x_test_trans = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()
    
    estimator_rf = rf_model.named_steps['rf']
    explainer_rf = shap.TreeExplainer(estimator_rf)
    shap_values_test = explainer_rf.shap_values(x_test_trans)
    
    if isinstance(shap_values_test, list):
        shap_values_test = shap_values_test[1]
        raw_base_val = explainer_rf.expected_value
    elif isinstance(shap_values_test, np.ndarray) and len(shap_values_test.shape) == 3:
        shap_values_test = shap_values_test[:, :, 1]
        raw_base_val = explainer_rf.expected_value
    else:
        raw_base_val = explainer_rf.expected_value
        
    if isinstance(raw_base_val, (np.ndarray, list)):
        arr = np.array(raw_base_val)
        if arr.size > 1:
            base_val = float(arr[1])
        else:
            base_val = float(arr.squeeze())
    else:
        base_val = float(raw_base_val)
        
    base_values_array = np.full(x_test_trans.shape[0], base_val)
        
    shap_exp_global = shap.Explanation(
        values=shap_values_test, 
        base_values=base_values_array, 
        data=x_test_trans, 
        feature_names=feature_names
    )
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_exp_global, max_display=10, show=False)
    st.pyplot(fig2)
    
# --- Page 5: Dataset & Feature Info ---
elif page == "📖 Dataset & Feature Info":
    st.title("Dataset & Clinical Feature Definitions")
    st.markdown("This dashboard is built on a validated clinical dataset. Below is the clinical relevance of each feature used for prediction.")
    
    feature_info = {
        "Age": "Patient's age in years. Strong predictor of cardiovascular risk.",
        "Sex": "Biological sex (0 = Female, 1 = Male). Men have higher baseline CVD risk.",
        "ChestPain": "Type of chest pain (typical, nontypical, nonanginal, asymptomatic). Asymptomatic often indicates severe disease.",
        "RestBP": "Resting blood pressure in mmHg. Hypertension is a major CVD risk factor.",
        "Chol": "Serum cholesterol in mg/dL. Elevated LDL contributes to atherosclerosis.",
        "Fbs": "Fasting blood sugar > 120 mg/dL (0 = No, 1 = Yes). Indicator of diabetes/metabolic syndrome.",
        "RestECG": "Resting Electrocardiogram results (0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy).",
        "MaxHR": "Maximum heart rate achieved in bpm. Lower values may indicate poor cardiac function.",
        "ExAng": "Exercise-induced angina (0 = No, 1 = Yes). Suggests ischemia during exertion.",
        "Oldpeak": "ST depression induced by exercise. Marker of myocardial ischemia.",
        "Slope": "Peak exercise ST segment slope (1 = Upsloping, 2 = Flat, 3 = Downsloping). Downsloping = higher risk.",
        "Ca": "Number of major vessels colored by fluoroscopy (0-3). More vessels = more severe disease.",
        "Thal": "Thalassemia stress test result (normal, fixed, reversable). Defects indicate ischemia."
    }
    
    for feature, desc in feature_info.items():
        st.markdown(f"**{feature}**: {desc}")