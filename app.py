import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# App.py jis folder mein hai, wahi se pkl files load karein
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { color: #38bdf8; font-size: 2rem; }
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        width: 100%;
        font-weight: bold;
    }
    .churn-box {
        background: #7f1d1d;
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .safe-box {
        background: #14532d;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .action-box {
        background: #1e293b;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.8rem;
        border: 1px solid #334155;
    }
    .tip {
        font-size: 0.85rem;
        color: #94a3b8;
        font-style: italic;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL FILES (same folder as app.py)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    # ✅ FIXED: Paths corrected - no duplicate filenames
    model_path = os.path.join(BASE_DIR, "xgb_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    cols_path = os.path.join(BASE_DIR, "columns.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    cols = joblib.load(cols_path)
    return model, scaler, cols

try:
    xgb_model, scaler, saved_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(f"❌ File nahi mili: {e}")
    st.warning(f"App yahan dhoondh rahi hai: `{BASE_DIR}`")
    st.info("xgb_model.pkl, scaler.pkl, columns.pkl — ye teeno files app.py ke saath usi folder mein honi chahiye.")
    st.stop()
except Exception as e:
    st.error(f"❌ Load error: {e}")
    st.stop()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📡 Telecom Customer Churn Predictor")
st.markdown("Customer ki information dal kar predict karein ke wo company chorenge ya nahi.")
st.divider()

# ─────────────────────────────────────────────
# INPUT FORM  (XGBoost important features)
# ─────────────────────────────────────────────
st.subheader("👤 Customer Details")

customer_name = st.text_input("Customer Name (optional)", placeholder="e.g. Ali Hassan")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("📅 Tenure (months)", min_value=0, max_value=72, value=12,
                       help="Customer kitne mahine se company ke saath hai")
    monthly_charges = st.slider("💵 Monthly Charges ($)", min_value=0.0, max_value=120.0,
                                 value=65.0, step=0.5)
    total_charges = st.number_input("💰 Total Charges ($)", min_value=0.0, max_value=9000.0,
                                     value=float(tenure * monthly_charges), step=10.0)
    contract = st.selectbox("📋 Contract Type",
                             ["Month-to-month", "One year", "Two year"],
                             help="Month-to-month = highest churn risk")

with col2:
    internet_service = st.selectbox("🌐 Internet Service",
                                    ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("💳 Payment Method",
                                  ["Bank transfer (automatic)",
                                   "Credit card (automatic)",
                                   "Electronic check",
                                   "Mailed check"])
    paperless_billing = st.selectbox("🧾 Paperless Billing", ["Yes", "No"])
    senior_citizen = st.selectbox("👴 Senior Citizen", ["No", "Yes"])

col3, col4 = st.columns(2)
with col3:
    online_security = st.selectbox("🔒 Online Security",
                                   ["Yes", "No", "No internet service"])
with col4:
    tech_support = st.selectbox("🛠️ Tech Support",
                                ["Yes", "No", "No internet service"])

gender = st.selectbox("⚧ Gender", ["Male", "Female"])

st.divider()

# ─────────────────────────────────────────────
# PREPROCESSING  (same as training notebook)
# ─────────────────────────────────────────────
def preprocess(tenure, monthly_charges, total_charges, senior_citizen,
               contract, internet_service, payment_method,
               paperless_billing, online_security, tech_support, gender):

    def tenure_group(x):
        if x <= 12:    return "New Customer"
        elif x <= 48:  return "Regular Customer"
        else:          return "Loyal Customer"

    def charges_group(x):
        if x <= 30:    return "Low Spend"
        elif x <= 60:  return "Medium Spend"
        else:          return "High Spend"

    def revenue_group(x):
        if x <= 1000:   return "Low Revenue"
        elif x <= 5000: return "Medium Revenue"
        else:           return "High Revenue"

    def contract_risk(c):
        if c == "Month-to-month": return "High Risk"
        elif c == "One year":     return "Medium Risk"
        else:                     return "Low Risk"

    yes_no = lambda v: 1 if v == "Yes" else 0

    row = {
        "tenure":              tenure,
        "MonthlyCharges":      monthly_charges,
        "TotalCharges":        total_charges,
        "SeniorCitizen":       yes_no(senior_citizen),
        "Partner":             0,
        "Dependents":          0,
        "PhoneService":        1,
        "MultipleLines":       0,
        "OnlineSecurity":      yes_no(online_security),
        "OnlineBackup":        0,
        "DeviceProtection":    0,
        "TechSupport":         yes_no(tech_support),
        "StreamingTV":         0,
        "StreamingMovies":     0,
        "PaperlessBilling":    yes_no(paperless_billing),
        "InternetService":     internet_service,
        "PaymentMethod":       payment_method,
        "Contract":            contract,
        "gender":              gender,
        "Tenure_Group":        tenure_group(tenure),
        "MonthlyCharge_Group": charges_group(monthly_charges),
        "Revenue_Group":       revenue_group(total_charges),
        "Contract_Risk":       contract_risk(contract),
    }

    df = pd.DataFrame([row])

    # Scale numerical columns (same as training)
    df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    # One-hot encode (same as training)
    df = pd.get_dummies(df, columns=["InternetService", "PaymentMethod",
                                      "Contract", "gender"], drop_first=True)
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with saved model
    df = df.reindex(columns=saved_columns, fill_value=0)

    return df

# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
if st.button("🔍 Predict Churn"):
    with st.spinner("Analyzing customer data..."):
        try:
            input_df = preprocess(
                tenure, monthly_charges, total_charges, senior_citizen,
                contract, internet_service, payment_method,
                paperless_billing, online_security, tech_support, gender
            )
            prediction = xgb_model.predict(input_df)[0]
            churn_prob = xgb_model.predict_proba(input_df)[0][1]
            name_display = customer_name.strip() if customer_name.strip() else "This Customer"

            # ── CHURN = 1 ──────────────────────────────────────────
            if prediction == 1:
                st.markdown(f"""
                <div class="churn-box">
                    <h3 style="color:#fca5a5;">⚠️ {name_display} – HIGH CHURN RISK</h3>
                    <p style="color:#fecaca; font-size:1.1rem;">
                        Churn Probability: <strong>{churn_prob:.1%}</strong>
                    </p>
                    <p style="color:#fca5a5;">Is customer ka company chhorne ka imkaan zyada hai.
                    Fauran action lein!</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="action-box">
                    <h4 style="color:#f97316;">📞 Recommended Actions – Contact Customer Now</h4>
                    <ul style="color:#e2e8f0; line-height:2;">
                        <li>📱 <strong>24 ghante ke andar call ya email karein</strong></li>
                        <li>🎁 <strong>Loyalty Bonus offer karein</strong> —
                            unke current product pe <strong>15–20% discount</strong></li>
                        <li>🤝 <strong>One-to-One Contract propose karein</strong> —
                            customer ki usage ke hisaab se personalized plan</li>
                        <li>🔒 <strong>Free add-ons dein</strong> —
                            3 mahine ke liye Online Security ya Tech Support free</li>
                        <li>📈 <strong>Contract upgrade offer karein</strong> —
                            Month-to-month se One Year ya Two Year contract mein special price pe</li>
                    </ul>
                    <p class="tip">💡 Ek existing customer ko retain karna, naye customer lane se
                    5 guna sasta hota hai.</p>
                </div>
                """, unsafe_allow_html=True)

            # ── CHURN = 0 ──────────────────────────────────────────
            else:
                st.markdown(f"""
                <div class="safe-box">
                    <h3 style="color:#86efac;">✅ {name_display} – LOW CHURN RISK</h3>
                    <p style="color:#bbf7d0; font-size:1.1rem;">
                        Churn Probability: <strong>{churn_prob:.1%}</strong>
                    </p>
                    <p style="color:#86efac;">Yeh customer loyal hai.
                    Relationship grow karein aur revenue maximize karein!</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="action-box">
                    <h4 style="color:#34d399;">💼 Recommended Actions – Grow Revenue</h4>
                    <ul style="color:#e2e8f0; line-height:2;">
                        <li>📈 <strong>Premium product upgrade offer karein</strong> —
                            Fiber Optic ya higher-tier plan ke saath profit margin lagayein</li>
                        <li>🎯 <strong>Cross-sell services karein</strong> —
                            Streaming TV, Online Backup, Device Protection bundle offer karein</li>
                        <li>🏆 <strong>Loyalty reward dein</strong> —
                            contract renewal pe exclusive discount ya free upgrade</li>
                        <li>💡 <strong>Upsell karein</strong> —
                            higher monthly plan recommend karein increased revenue ke liye</li>
                        <li>📊 <strong>Product usage monitor karein</strong> —
                            customer kitna product use kar raha hai track karo pehle</li>
                    </ul>
                    <p class="tip">⚠️ Note: Is customer ko One-to-One Contract mat dein —
                    product usage kam hai jo company ke liye risky ho sakta hai.</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction mein error aaya: {e}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "Telecom Customer Churn Prediction System · Built with XGBoost + Streamlit"
    "</p>",
    unsafe_allow_html=True
)