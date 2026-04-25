"""
frontend/app.py
────────────────
Streamlit UI for Insurance Fraud Detection.
Updated to include property_damage field.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://backend:8000")

st.set_page_config(
    page_title="Insurance Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Detector")
st.sidebar.caption("DA5402 MLOps Project")
page = st.sidebar.radio(
    "Navigate",
    ["Single Prediction", "Batch Prediction", "Drift Monitoring", "Model Info"],
)


# ── API helpers ───────────────────────────────────────────────────────────────
def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def api_predict(payload: dict, threshold: float | None = None) -> dict:
    url = f"{API_BASE}/api/v1/predict"
    if threshold is not None:
        url += f"?threshold={threshold}"
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def api_predict_batch(claims: list) -> dict:
    r = requests.post(
        f"{API_BASE}/api/v1/predict/batch",
        json={"claims": claims},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_model_info() -> dict:
    try:
        r = requests.get(f"{API_BASE}/info", timeout=5)
        return r.json()
    except Exception:
        return {}


def api_metrics_raw() -> str:
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=5)
        return r.text
    except Exception:
        return ""


def parse_drift_scores(raw: str) -> dict:
    scores = {}
    for line in raw.splitlines():
        if line.startswith("fraud_feature_drift_score{"):
            try:
                feature = line.split('feature_name="')[1].split('"')[0]
                value = float(line.split("} ")[1])
                scores[feature] = value
            except (IndexError, ValueError):
                continue
    return scores


# ── Health badge ──────────────────────────────────────────────────────────────
health = api_health()
status = health.get("status", "unreachable")
badge = "🟢" if status == "healthy" else "🟡" if status == "degraded" else "🔴"
st.sidebar.markdown(f"**API:** {badge} {status.upper()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Prediction
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Single Prediction":
    st.title("🕵️ Single Claim Prediction")

    with st.expander("⚙️ Threshold override", expanded=False):
        use_custom = st.checkbox("Override default threshold")
        custom_threshold = st.slider("Threshold", 0.0, 1.0, 0.4, 0.01) if use_custom else None

    with st.form("claim_form"):
        st.subheader("Policy Information")
        c1, c2, c3 = st.columns(3)
        months_as_customer = c1.number_input("Months as Customer", 0, 600, 36)
        age = c2.number_input("Age", 16, 100, 35)
        policy_state = c3.selectbox("Policy State", ["OH", "IL", "IN", "PA", "NC", "SC", "WV", "VA", "NY"])
        c4, c5, c6 = st.columns(3)
        policy_csl = c4.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
        policy_deductable = c5.selectbox("Deductible ($)", [500, 1000, 2000])
        policy_annual_premium = c6.number_input("Annual Premium ($)", 0.0, 50000.0, 1200.0)
        umbrella_limit = st.number_input("Umbrella Limit ($)", 0, 10_000_000, 0, step=100_000)

        st.subheader("Insured Person")
        c7, c8 = st.columns(2)
        insured_sex = c7.selectbox("Sex", ["MALE", "FEMALE"])
        insured_education_level = c8.selectbox(
            "Education", ["High School", "College", "Associate", "MD", "Masters", "PhD", "JD"]
        )
        c9, c10 = st.columns(2)
        insured_occupation = c9.selectbox("Occupation", [
            "craft-repair", "machine-op-inspct", "sales", "armed-forces",
            "tech-support", "prof-specialty", "other-service", "exec-managerial",
            "transport-moving", "handlers-cleaners", "farming-fishing",
            "protective-serv", "priv-house-serv", "adm-clerical",
        ])
        insured_hobbies = c10.selectbox("Hobbies", [
            "sleeping", "reading", "board-games", "bungie-jumping", "base-jumping",
            "golf", "camping", "movies", "hiking", "yachting", "paintball",
            "chess", "cross-fit", "polo", "skydiving", "kayaking",
        ])
        c11, c12 = st.columns(2)
        insured_relationship = c11.selectbox(
            "Relationship",
            ["husband", "wife", "own-child", "not-in-family", "other-relative", "unmarried"],
        )
        capital_gains = c12.number_input("Capital Gains ($)", 0, 200_000, 0)
        capital_loss = st.number_input("Capital Loss ($)", 0, 200_000, 0)

        st.subheader("Incident Details")
        c13, c14 = st.columns(2)
        incident_type = c13.selectbox("Incident Type", [
            "Single Vehicle Collision", "Multi-vehicle Collision",
            "Vehicle Theft", "Parked Car",
        ])
        collision_type = c14.selectbox(
            "Collision Type", ["Front Collision", "Rear Collision", "Side Collision", "NA"]
        )
        c15, c16 = st.columns(2)
        incident_severity = c15.selectbox(
            "Severity", ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"]
        )
        authorities_contacted = c16.selectbox(
            "Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"]
        )
        c17, c18 = st.columns(2)
        incident_state = c17.selectbox(
            "Incident State", ["OH", "IL", "IN", "PA", "NC", "SC", "WV", "VA", "NY"]
        )
        incident_city = c18.text_input("Incident City", "Columbus")

        c19, c20, c21 = st.columns(3)
        incident_hour = c19.number_input("Hour of Day (0–23)", 0, 23, 14)
        num_vehicles = c20.number_input("Vehicles Involved", 1, 10, 1)
        property_damage = c21.selectbox("Property Damage", ["YES", "NO", "Unknown"])

        c22, c23, c24 = st.columns(3)
        bodily_injuries = c22.number_input("Bodily Injuries", 0, 10, 1)
        witnesses = c23.number_input("Witnesses", 0, 10, 0)
        police_report = c24.selectbox("Police Report Available", ["YES", "NO"])

        st.subheader("Claim Amounts")
        c25, c26, c27, c28 = st.columns(4)
        total_claim = c25.number_input("Total Claim ($)", 0, 1_000_000, 65000)
        injury_claim = c26.number_input("Injury Claim ($)", 0, 500_000, 10000)
        property_claim = c27.number_input("Property Claim ($)", 0, 500_000, 5000)
        vehicle_claim = c28.number_input("Vehicle Claim ($)", 0, 500_000, 50000)

        st.subheader("Vehicle")
        c29, c30 = st.columns(2)
        auto_make = c29.selectbox("Auto Make", [
            "Saab", "Mercedes", "Dodge", "Chevrolet", "Accura", "Ford",
            "Jeep", "Suburu", "Toyota", "Honda", "BMW", "Nissan",
            "Audi", "Volkswagen", "Pontiac",
        ])
        auto_year = c30.number_input("Auto Year", 1980, 2025, 2012)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        payload = {
            "months_as_customer": months_as_customer,
            "age": age,
            "policy_state": policy_state,
            "policy_csl": policy_csl,
            "policy_deductable": policy_deductable,
            "policy_annual_premium": policy_annual_premium,
            "umbrella_limit": umbrella_limit,
            "insured_sex": insured_sex,
            "insured_education_level": insured_education_level,
            "insured_occupation": insured_occupation,
            "insured_hobbies": insured_hobbies,
            "insured_relationship": insured_relationship,
            "capital-gains": capital_gains,
            "capital-loss": capital_loss,
            "incident_type": incident_type,
            "collision_type": None if collision_type == "NA" else collision_type,
            "incident_severity": incident_severity,
            "authorities_contacted": None if authorities_contacted == "None" else authorities_contacted,
            "incident_state": incident_state,
            "incident_city": incident_city,
            "incident_hour_of_the_day": incident_hour,
            "number_of_vehicles_involved": num_vehicles,
            "property_damage": None if property_damage == "Unknown" else property_damage,
            "bodily_injuries": bodily_injuries,
            "witnesses": witnesses,
            "police_report_available": police_report,
            "total_claim_amount": total_claim,
            "injury_claim": injury_claim,
            "property_claim": property_claim,
            "vehicle_claim": vehicle_claim,
            "auto_make": auto_make,
            "auto_year": auto_year,
        }
        with st.spinner("Running prediction ..."):
            try:
                result = api_predict(payload, custom_threshold)
                tier = result["risk_tier"]
                color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[tier]
                verdict = "⚠️ FRAUD DETECTED" if result["is_fraud"] else "✅ LEGITIMATE"

                st.markdown("---")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Verdict", verdict)
                col_b.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
                col_c.metric("Risk Score", f"{result['risk_score']} / 100")
                st.markdown(f"### {color} Risk Tier: **{tier}**")
                st.info(result["message"])
                with st.expander("Full response"):
                    st.json(result)

            except requests.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Batch Prediction":
    st.title("📋 Batch Claim Prediction")
    st.caption("Upload a CSV (max 100 rows). Column names must match the claim schema.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)} rows**, {len(df.columns)} columns.")
        st.dataframe(df.head(5))

        if len(df) > 100:
            st.warning("Only the first 100 rows will be scored (API limit).")
            df = df.head(100)

        if st.button("🔍 Run Batch Prediction", use_container_width=True):
            claims = df.to_dict(orient="records")
            with st.spinner(f"Scoring {len(claims)} claims ..."):
                try:
                    result = api_predict_batch(claims)
                    preds = result["predictions"]

                    cols = st.columns(3)
                    cols[0].metric("Total Claims", result["total"])
                    cols[1].metric("Fraud Detected", result["fraud_count"])
                    cols[2].metric("Legitimate", result["legitimate_count"])

                    rows = []
                    for i, p in enumerate(preds):
                        rows.append({
                            "claim_id": p.get("claim_id", i),
                            "is_fraud": "⚠️ Fraud" if p["is_fraud"] else "✅ Legit",
                            "fraud_probability": f"{p['fraud_probability']:.1%}",
                            "risk_score": p["risk_score"],
                            "risk_tier": p["risk_tier"],
                            "message": p["message"],
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

                    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_bytes,
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                    )
                except requests.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Drift Monitoring
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Drift Monitoring":
    st.title("📡 Data Drift Monitoring")
    st.caption("PSI scores per feature. PSI > 0.1 = moderate drift, PSI > 0.2 = significant.")

    if st.button("🔄 Refresh"):
        st.rerun()

    raw = api_metrics_raw()
    if not raw:
        st.error("Could not reach /metrics. Is the backend running?")
    else:
        scores = parse_drift_scores(raw)
        if not scores:
            st.info("No drift scores yet. The backend needs prediction traffic first.")
        else:
            rows = []
            for feature, psi in sorted(scores.items(), key=lambda x: -x[1]):
                if psi > 0.2:
                    status = "🔴 Significant"
                elif psi > 0.1:
                    status = "🟡 Moderate"
                else:
                    status = "🟢 Stable"
                rows.append({"Feature": feature, "PSI Score": round(psi, 4), "Status": status})

            drift_df = pd.DataFrame(rows)
            st.dataframe(drift_df, use_container_width=True)
            st.bar_chart(drift_df.set_index("Feature")["PSI Score"])

            alert_lines = [l for l in raw.splitlines() if l.startswith("fraud_drift_alert ")]
            if alert_lines:
                alert_val = float(alert_lines[0].split(" ")[1])
                if alert_val == 1:
                    st.error("⚠️ DRIFT ALERT ACTIVE — at least one feature exceeds PSI threshold.")
                else:
                    st.success("✅ No significant drift detected.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Info
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Info":
    st.title("ℹ️ Model Information")

    info = api_model_info()
    if not info:
        st.error("Could not fetch model info. Is the backend running?")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Model Name", info.get("model_name", "—"))
        c2.metric("Stage", info.get("model_stage", "—"))
        c3, c4 = st.columns(2)
        c3.metric("Algorithm", info.get("algorithm", "—"))
        c4.metric("Decision Threshold", info.get("threshold", "—"))
        st.caption(f"MLflow Tracking URI: `{info.get('mlflow_tracking_uri', '—')}`")

    st.markdown("---")
    st.subheader("Backend Health")
    st.json(api_health())

    st.markdown("---")
    st.subheader("Force Model Reload")
    st.caption("Use after promoting a new model version to Production in MLflow.")
    if st.button("🔁 Reload Model from Registry"):
        try:
            r = requests.post(f"{API_BASE}/api/v1/model/reload", timeout=15)
            r.raise_for_status()
            st.success(r.json().get("message", "Reloaded."))
        except Exception as e:
            st.error(f"Reload failed: {e}")
