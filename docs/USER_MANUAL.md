# User Manual — InsureGuard Fraud Detection

## For Non-Technical Users

---

## What Does This System Do?

InsureGuard analyses insurance claims and tells you whether a claim is **likely fraudulent** or **legitimate**. It gives you a **risk score from 0 to 100** and one of three risk levels:

- 🟢 **LOW** (0–30): Claim appears legitimate
- 🟡 **MEDIUM** (30–60): Moderate indicators — recommend manual review
- 🔴 **HIGH** (60–100): Strong fraud indicators — escalate for investigation

---

## Step-by-Step: How to Submit a Claim

**1. Open your browser** and go to: `http://localhost:3000`

**2. Fill in the claim form.** The form has five sections:
   - **Policy Information** — deductible, premium, state, customer age
   - **Insured Person** — sex, education, occupation, relationship
   - **Incident Details** — type of accident, severity, hour, vehicles, witnesses
   - **Claim Amounts** — total claim and individual components
   - **Vehicle Details** — make and year

   > 💡 All fields marked with `*` are required. If you are not sure, use the **"Load Example"** button to see a sample claim.

**3. Click "Analyse Claim"** and wait 1–2 seconds.

**4. Read your results:**

   | What you see | What it means |
   |--------------|---------------|
   | ✓ LEGITIMATE | Low fraud risk. Process normally. |
   | ⚠ FRAUD DETECTED | Fraud indicators found. See message below for action. |
   | Risk Score: 72 | 72 out of 100 fraud risk |
   | Fraud Probability: 72% | Model is 72% confident this is fraud |

**5. Follow the recommendation** shown in the coloured box below the score.

---

## Tips

- You can submit one claim at a time through the form, or ask your IT team to use the batch API for bulk processing.
- If you see an error message in red at the bottom of the screen, the backend server may not be running. Contact your administrator.
- The system learns from new data — models are retrained automatically every night.

---

## ML Pipeline Screen

Click **"ML Pipeline"** in the top navigation to see:
- **Airflow**: Status of the daily data pipeline
- **MLflow**: Experiment history and model versions
- **Grafana**: Live monitoring dashboards

These are for administrators and data scientists.

---

## Frequently Asked Questions

**Q: Can the system be wrong?**  
A: Yes. The model has ~85% accuracy but is not perfect. HIGH risk claims should be reviewed by a human investigator, not automatically rejected.

**Q: Is my data stored?**  
A: Claim data entered in the form is sent to the backend API for prediction only. It is not permanently stored unless your IT team has configured logging.

**Q: What if the form says "Model not available"?**  
A: The ML model has not been trained yet. Ask your administrator to run `dvc repro` to train the model.
