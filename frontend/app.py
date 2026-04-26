"""
frontend/app.py
────────────────
Streamlit frontend for Insurance Fraud Detection.
All form fields and payload keys match the existing backend ClaimRequest schema exactly:
  months_as_customer, age, policy_state, policy_csl, policy_deductable,
  policy_annual_premium, umbrella_limit, insured_sex, insured_education_level,
  insured_occupation, insured_hobbies, insured_relationship, capital-gains,
  capital-loss, incident_type, collision_type, incident_severity,
  authorities_contacted, incident_state, incident_city, incident_hour_of_the_day,
  number_of_vehicles_involved, bodily_injuries, witnesses, police_report_available,
  total_claim_amount, injury_claim, property_claim, vehicle_claim, auto_make, auto_year

HOW API_BASE IS RESOLVED:
  Local  (streamlit run frontend/app.py) → http://localhost:8000  (default fallback)
  Docker (docker compose up)             → http://backend:8000    (set via env var)
"""

import os
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="InsureGuard — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
[data-testid="metric-container"] {
    background:#f8faff; border:1px solid #dce8ff;
    border-radius:10px; padding:14px 18px;
}
.section-title {
    font-size:0.78rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.5px; color:#1a56db; margin-bottom:6px;
}
.result-box  { border-radius:12px; padding:24px 28px; margin-top:16px; }
.result-fraud{ background:#fff1f2; border-left:5px solid #ef4444; }
.result-legit{ background:#f0fdf4; border-left:5px solid #22c55e; }
.badge-LOW   { background:#d1fae5; color:#065f46; padding:4px 14px;
               border-radius:20px; font-weight:700; font-size:0.88rem; }
.badge-MEDIUM{ background:#fef3c7; color:#92400e; padding:4px 14px;
               border-radius:20px; font-weight:700; font-size:0.88rem; }
.badge-HIGH  { background:#fee2e2; color:#991b1b; padding:4px 14px;
               border-radius:20px; font-weight:700; font-size:0.88rem; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/80/shield.png", width=60)
    st.title("InsureGuard")
    st.caption("AI-powered Insurance Fraud Detection")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🔍 Predict Fraud", "📂 Batch Predict", "⚙️ ML Pipeline", "📖 User Guide"],
        label_visibility="collapsed",
    )

    st.divider()
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        d = r.json()
        if d.get("model_loaded"):
            st.success("✅ API: Online | Model: Loaded")
        else:
            st.warning(
                "⚠️ API online but model not loaded.\n\n"
                "Run `dvc repro` then promote to **Production** in MLflow."
            )
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Cannot reach `{API_BASE}`\n\n"
            "**Locally:** start the backend:\n"
            "```\nuvicorn backend.app.main:app --port 8000\n```\n\n"
            "**Docker:** `docker compose up`"
        )
    except Exception as e:
        st.error(f"❌ {e}")
    st.caption(f"API: `{API_BASE}`")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def post_predict(payload: dict, threshold: Optional[float] = None) -> dict:
    url = f"{API_BASE}/api/v1/predict"
    if threshold is not None:
        url += f"?threshold={threshold}"
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def render_result(result: dict) -> None:
    tier     = result["risk_tier"]
    is_fraud = result["is_fraud"]
    risk     = result["risk_score"]
    prob_pct = result["fraud_probability"] * 100
    verdict  = "⚠️ FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE CLAIM"
    box_cls  = "result-fraud" if is_fraud else "result-legit"

    st.markdown(
        f'<div class="result-box {box_cls}">'
        f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">'
        f'<span style="font-size:1.4rem;font-weight:800;">{verdict}</span>'
        f'<span class="badge-{tier}">{tier} RISK</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Score",        f"{risk} / 100")
    c2.metric("Fraud Probability", f"{prob_pct:.1f}%")
    c3.metric("Threshold Used",    result["threshold_used"])
    st.progress(int(risk), text=f"Risk Level: {tier}")
    st.info(result["message"])
    st.caption(f"Claim ID: `{result.get('claim_id', 'N/A')}`")


def handle_error(e: Exception) -> None:
    if isinstance(e, requests.exceptions.ConnectionError):
        st.error(
            f"❌ **Connection failed** — cannot reach `{API_BASE}`\n\n"
            "**Locally:** `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`\n\n"
            "**Docker:** `docker compose up`"
        )
    elif isinstance(e, requests.exceptions.Timeout):
        st.error("❌ Request timed out. The backend may still be loading the model.")
    elif isinstance(e, requests.exceptions.HTTPError):
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        st.error(f"❌ API error ({e.response.status_code}): {detail}")
    else:
        st.error(f"❌ Unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Claim Prediction
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Predict Fraud":
    st.title("🔍 Insurance Fraud Prediction")
    st.markdown("Fill in the claim details to get an AI-powered fraud risk assessment.")
    st.divider()

    with st.expander("⚙️ Advanced Options"):
        use_custom = st.checkbox("Override decision threshold", value=False)
        threshold_val = None
        if use_custom:
            threshold_val = st.slider("Threshold (lower = more sensitive to fraud)", 0.10, 0.90, 0.40, 0.05)
            st.caption("Default: 0.40 — tuned to maximise fraud recall.")

    with st.form("claim_form"):

        # ── Policy ────────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">📋 Policy Information</p>', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        months_as_customer    = r1c1.number_input("Months as Customer *", 0, 600, 36)
        age                   = r1c2.number_input("Age *", 16, 100, 35)
        policy_state          = r1c3.selectbox("Policy State *", [
            "OH", "IL", "IN", "NY", "CA", "TX", "FL", "PA", "NC", "WA", "GA", "AZ"
        ])

        r2c1, r2c2, r2c3 = st.columns(3)
        policy_csl            = r2c1.selectbox("Coverage Limit (CSL) *", ["100/300", "250/500", "500/1000"])
        policy_deductable     = r2c2.selectbox("Deductible ($) *", [500, 1000, 2000])
        policy_annual_premium = r2c3.number_input("Annual Premium ($) *", 0.0, value=1200.0, step=50.0)
        umbrella_limit        = st.number_input("Umbrella Limit ($)", 0, value=0, step=100000)

        st.divider()

        # ── Insured Person ────────────────────────────────────────────────────
        st.markdown('<p class="section-title">👤 Insured Person</p>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3 = st.columns(3)
        insured_sex              = r3c1.selectbox("Sex *", ["MALE", "FEMALE"])
        insured_education_level  = r3c2.selectbox("Education Level *", [
            "High School", "College", "Associate", "MD", "Masters", "PhD", "JD"
        ])
        insured_occupation       = r3c3.text_input("Occupation *", "craft-repair")

        r4c1, r4c2, r4c3 = st.columns(3)
        insured_hobbies          = r4c1.text_input("Hobbies", "chess")
        insured_relationship     = r4c2.selectbox("Relationship to Policy *", [
            "husband", "wife", "own-child", "unmarried", "other-relative"
        ])
        r4c3.write("")

        r5c1, r5c2 = st.columns(2)
        capital_gains = r5c1.number_input("Capital Gains ($)", 0, value=0)
        capital_loss  = r5c2.number_input("Capital Loss ($)",  0, value=0)

        st.divider()

        # ── Incident ──────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">🚗 Incident Details</p>', unsafe_allow_html=True)
        r6c1, r6c2, r6c3 = st.columns(3)
        incident_type        = r6c1.selectbox("Incident Type *", [
            "Single Vehicle Collision", "Multi-vehicle Collision",
            "Vehicle Theft", "Parked Car"
        ])
        incident_severity    = r6c2.selectbox("Severity *", [
            "Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"
        ])
        collision_type       = r6c3.selectbox("Collision Type", [
            "Front Collision", "Rear Collision", "Side Collision", "NA"
        ])

        r7c1, r7c2, r7c3 = st.columns(3)
        authorities_contacted = r7c1.selectbox("Authorities Contacted", [
            "Police", "Fire", "Ambulance", "Other", "None"
        ])
        incident_state        = r7c2.text_input("Incident State *", "OH", max_chars=2)
        incident_city         = r7c3.text_input("Incident City *", "Columbus")

        r8c1, r8c2, r8c3 = st.columns(3)
        incident_hour_of_the_day    = r8c1.number_input("Hour of Incident (0–23) *", 0, 23, 14)
        number_of_vehicles_involved = r8c2.number_input("Vehicles Involved *", 1, 10, 1)
        bodily_injuries             = r8c3.number_input("Bodily Injuries *", 0, 10, 0)

        r9c1, r9c2 = st.columns(2)
        witnesses               = r9c1.number_input("Witnesses *", 0, 10, 0)
        police_report_available = r9c2.selectbox("Police Report Available *", ["YES", "NO"])

        st.divider()

        # ── Claim Amounts ─────────────────────────────────────────────────────
        st.markdown('<p class="section-title">💰 Claim Amounts</p>', unsafe_allow_html=True)
        r10c1, r10c2 = st.columns(2)
        total_claim_amount = r10c1.number_input("Total Claim Amount ($) *", 0.0, value=65000.0, step=500.0)
        injury_claim       = r10c2.number_input("Injury Claim ($)",         0.0, value=10000.0, step=500.0)
        r11c1, r11c2 = st.columns(2)
        property_claim     = r11c1.number_input("Property Claim ($)",       0.0, value=5000.0,  step=500.0)
        vehicle_claim      = r11c2.number_input("Vehicle Claim ($)",        0.0, value=50000.0, step=500.0)

        st.divider()

        # ── Vehicle ───────────────────────────────────────────────────────────
        st.markdown('<p class="section-title">🚘 Vehicle Details</p>', unsafe_allow_html=True)
        r12c1, r12c2 = st.columns(2)
        auto_make = r12c1.text_input("Auto Make *", "Saab")
        auto_year = r12c2.number_input("Auto Year *", 1980, 2025, 2012)

        submitted = st.form_submit_button("🔍 Analyse Claim", use_container_width=True, type="primary")

    # ── On submit ─────────────────────────────────────────────────────────────
    if submitted:
        payload = {
            "months_as_customer":           int(months_as_customer),
            "age":                          int(age),
            "policy_state":                 policy_state,
            "policy_csl":                   policy_csl,
            "policy_deductable":            int(policy_deductable),
            "policy_annual_premium":        float(policy_annual_premium),
            "umbrella_limit":               int(umbrella_limit),
            "insured_sex":                  insured_sex,
            "insured_education_level":      insured_education_level,
            "insured_occupation":           insured_occupation,
            "insured_hobbies":              insured_hobbies or None,
            "insured_relationship":         insured_relationship,
            "capital-gains":                int(capital_gains),
            "capital-loss":                 int(capital_loss),
            "incident_type":                incident_type,
            "collision_type":               None if collision_type == "NA" else collision_type,
            "incident_severity":            incident_severity,
            "authorities_contacted":        None if authorities_contacted == "None" else authorities_contacted,
            "incident_state":               incident_state.upper(),
            "incident_city":                incident_city,
            "incident_hour_of_the_day":     int(incident_hour_of_the_day),
            "number_of_vehicles_involved":  int(number_of_vehicles_involved),
            "bodily_injuries":              int(bodily_injuries),
            "witnesses":                    int(witnesses),
            "police_report_available":      police_report_available,
            "total_claim_amount":           float(total_claim_amount),
            "injury_claim":                 float(injury_claim),
            "property_claim":               float(property_claim),
            "vehicle_claim":                float(vehicle_claim),
            "auto_make":                    auto_make,
            "auto_year":                    int(auto_year),
        }

        with st.spinner("Analysing claim..."):
            try:
                result = post_predict(payload, threshold_val)
                st.success("Analysis complete!")
                render_result(result)
            except Exception as e:
                handle_error(e)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch Predict":
    st.title("📂 Batch Fraud Prediction")
    st.markdown("Upload a CSV file with multiple claims (up to 100 rows).")
    st.divider()

    sample = pd.DataFrame([{
        "months_as_customer": 36, "age": 35, "policy_state": "OH",
        "policy_csl": "250/500", "policy_deductable": 500,
        "policy_annual_premium": 1200.0, "umbrella_limit": 0,
        "insured_sex": "MALE", "insured_education_level": "MD",
        "insured_occupation": "craft-repair", "insured_hobbies": "chess",
        "insured_relationship": "husband", "capital-gains": 0, "capital-loss": 0,
        "incident_type": "Single Vehicle Collision",
        "collision_type": "Front Collision", "incident_severity": "Major Damage",
        "authorities_contacted": "Police", "incident_state": "OH",
        "incident_city": "Columbus", "incident_hour_of_the_day": 14,
        "number_of_vehicles_involved": 1, "bodily_injuries": 1,
        "witnesses": 0, "police_report_available": "YES",
        "total_claim_amount": 65000.0, "injury_claim": 10000.0,
        "property_claim": 5000.0, "vehicle_claim": 50000.0,
        "auto_make": "Saab", "auto_year": 2012,
    }])
    st.download_button("⬇️  Download CSV Template", data=sample.to_csv(index=False),
                       file_name="claims_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload claims CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**{len(df)} claims loaded.** Preview:")
        st.dataframe(df.head(5), use_container_width=True)

        required = [
            "months_as_customer", "age", "policy_state", "policy_csl",
            "policy_deductable", "policy_annual_premium", "insured_sex",
            "insured_education_level", "insured_occupation", "insured_relationship",
            "incident_type", "incident_severity", "incident_state", "incident_city",
            "incident_hour_of_the_day", "number_of_vehicles_involved",
            "bodily_injuries", "witnesses", "police_report_available",
            "total_claim_amount", "auto_make", "auto_year",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"CSV missing required columns: `{missing}`")
        elif st.button("🚀 Run Batch Prediction", type="primary"):
            int_cols   = ["months_as_customer", "age", "policy_deductable", "umbrella_limit",
                          "capital-gains", "capital-loss", "incident_hour_of_the_day",
                          "number_of_vehicles_involved", "bodily_injuries", "witnesses", "auto_year"]
            float_cols = ["policy_annual_premium", "total_claim_amount",
                          "injury_claim", "property_claim", "vehicle_claim"]
            opt_cols   = ["collision_type", "authorities_contacted", "insured_hobbies"]

            claims = df.fillna("").to_dict(orient="records")
            for c in claims:
                for k in int_cols:
                    if k in c:
                        try: c[k] = int(float(str(c[k]))) if str(c[k]).strip() else 0
                        except: c[k] = 0
                for k in float_cols:
                    if k in c:
                        try: c[k] = float(str(c[k])) if str(c[k]).strip() else 0.0
                        except: c[k] = 0.0
                for k in opt_cols:
                    if not str(c.get(k, "")).strip():
                        c[k] = None

            with st.spinner(f"Processing {min(len(claims), 100)} claims..."):
                try:
                    r = requests.post(f"{API_BASE}/api/v1/predict/batch",
                                      json={"claims": claims[:100]}, timeout=60)
                    r.raise_for_status()
                    data = r.json()

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total",        data["total"])
                    col2.metric("Fraud",        data["fraud_count"])
                    col3.metric("Legitimate",   data["legitimate_count"])
                    col4.metric("Fraud Rate",   f"{data['fraud_count']/data['total']*100:.1f}%")

                    rows = [{
                        "Row": i+1, "Risk Score": p["risk_score"], "Tier": p["risk_tier"],
                        "Fraud Prob": f"{p['fraud_probability']*100:.1f}%",
                        "Verdict": "⚠️ FRAUD" if p["is_fraud"] else "✅ Legit",
                    } for i, p in enumerate(data["predictions"])]

                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    st.download_button("⬇️  Download Results",
                                       data=pd.DataFrame(rows).to_csv(index=False),
                                       file_name="fraud_predictions.csv", mime="text/csv")
                except Exception as e:
                    handle_error(e)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ ML Pipeline":
    st.title("⚙️ ML Pipeline & MLOps Console")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🌀 Apache Airflow")
        try: requests.get("http://localhost:8080/health", timeout=3); st.success("✅ Running")
        except: st.warning("⚠️ Not reachable")
        st.markdown("- **DAG**: `insurance_fraud_ml_pipeline`\n- **Schedule**: Daily 02:00 UTC")
        st.link_button("Open Airflow →", "http://localhost:8080")

    with col2:
        st.subheader("📊 MLflow")
        try: requests.get("http://localhost:5000/health", timeout=3); st.success("✅ Running")
        except: st.warning("⚠️ Not reachable")
        st.markdown("- **Experiment**: `insurance_fraud_detection`\n- **Registry**: `insurance_fraud_model`")
        st.link_button("Open MLflow →", "http://localhost:5000")

    with col3:
        st.subheader("📈 Grafana")
        try: requests.get("http://localhost:3001/api/health", timeout=3); st.success("✅ Running")
        except: st.warning("⚠️ Not reachable")
        st.markdown("- **Source**: Prometheus (9090)\n- **Alert**: error_rate > 5%")
        st.link_button("Open Grafana →", "http://localhost:3001")

    st.divider()
    st.subheader("🔀 DVC Pipeline DAG")
    st.code("ingest\n  └── validate\n        └── preprocess\n              └── train\n                    └── evaluate", language="text")
    st.markdown("```bash\ndvc repro\ndvc dag\ndvc metrics show\n```")

    st.divider()
    st.subheader("🤖 Current Model")
    try:
        r = requests.get(f"{API_BASE}/info", timeout=3)
        if r.status_code == 200:
            info = r.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Name", info["model_name"]); c2.metric("Stage", info["model_stage"])
            c3.metric("Algorithm", info["algorithm"]); c4.metric("Threshold", info["threshold"])
        if st.button("🔄 Reload Model"):
            rr = requests.post(f"{API_BASE}/api/v1/model/reload", timeout=5)
            st.success("Reloaded!") if rr.ok else st.error("Failed.")
    except Exception:
        st.info("Start the backend to see model info.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — User Guide
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📖 User Guide":
    st.title("📖 User Guide")
    st.divider()
    st.markdown("""
    ## What Does InsureGuard Do?

    InsureGuard analyses auto insurance claims and determines whether a claim shows
    signs of **fraud** using a trained machine learning model.

    ---

    ## Risk Levels

    | Badge | Score | Meaning | Action |
    |-------|-------|---------|--------|
    | 🟢 LOW | 0–30 | Claim appears legitimate | Process normally |
    | 🟡 MEDIUM | 30–60 | Moderate indicators | Manual review recommended |
    | 🔴 HIGH | 60–100 | Strong fraud signals | Escalate for investigation |

    ---

    ## How to Submit a Claim

    1. Click **🔍 Predict Fraud** in the sidebar
    2. Fill all fields marked `*`
    3. Click **Analyse Claim**
    4. Follow the recommendation shown

    ---

    ## Key Fraud Signals the Model Uses

    - High claim-to-premium ratio
    - Night-time incidents (22:00–06:00)
    - No police report filed
    - Claim components don't sum to total
    - Multiple vehicles involved

    ---

    ## Disclaimer

    The model is an **aid for investigators**, not a final decision-maker.
    HIGH risk = warrants review, not automatic rejection.
    """)