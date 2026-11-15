"""Modern Streamlit interface for the cardiovascular dataset MLP research demo."""

import json
from pathlib import Path

import streamlit as st
import torch

from model import CardioMLP
from preprocessing import make_feature_row, transform_frame


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2rem 2.1rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #111827 0%, #26364d 62%, #7f1d1d 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 750;
        letter-spacing: -0.04em;
        margin: 0.2rem 0 0.6rem 0;
    }
    .hero p {
        color: #e5e7eb;
        font-size: 1rem;
        margin: 0;
    }
    .eyebrow {
        color: #fecaca;
        text-transform: uppercase;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.13em;
    }
    .section-label {
        font-size: 1.12rem;
        font-weight: 700;
        margin: 0.3rem 0 0.2rem 0;
    }
    .muted {
        color: #64748b;
        font-size: 0.9rem;
    }
    .result-card {
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 18px;
        background: rgba(148, 163, 184, 0.07);
    }
    .result-title {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 15px;
        padding: 0.9rem 1rem;
        background: rgba(148, 163, 184, 0.06);
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button {
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 700;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    config = json.loads((ARTIFACTS / "preprocessing.json").read_text())
    report = json.loads((ARTIFACTS / "metrics.json").read_text())
    state = torch.load(
        ARTIFACTS / "mlp_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = CardioMLP(config["input_dim"])
    model.load_state_dict(state)
    model.eval()
    return config, report, model


try:
    config, report, model = load_assets()
except Exception as exc:
    st.error(
        "The trained model files are unavailable. Check the latest training workflow "
        "and confirm that the artifacts folder is committed."
    )
    st.exception(exc)
    st.stop()


operating_point = report["operating_points"]["high_recall"]
metrics = operating_point["test"]
threshold = float(operating_point["threshold"])


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Machine learning research application</div>
        <h1>CardioRisk AI</h1>
        <p>Explore a cardiovascular dataset classification result using a PyTorch neural network.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research and educational demonstration only. This model is not clinically validated "
    "and must not be used for diagnosis, treatment, screening, or healthcare decisions."
)

st.markdown('<div class="section-label">Model performance</div>', unsafe_allow_html=True)
st.caption(
    "Held-out test metrics for the selected high-recall operating point. "
    "These are dataset evaluation metrics, not individual-level clinical confidence."
)

metric_columns = st.columns(4)
metric_columns[0].metric("Recall", f'{metrics["recall"]:.1%}')
metric_columns[1].metric("Precision", f'{metrics["precision"]:.1%}')
metric_columns[2].metric("ROC-AUC", f'{metrics["roc_auc"]:.3f}')
metric_columns[3].metric("PR-AUC", f'{metrics["pr_auc"]:.3f}')

st.divider()
st.markdown('<div class="section-label">Analyse an observation</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="muted">Enter values in the same format as the source dataset. '
    'All inputs are processed locally in this app session.</p>',
    unsafe_allow_html=True,
)

with st.form("cardio_observation_form"):
    demographic_column, body_column, clinical_column = st.columns(3)

    with demographic_column:
        st.markdown("**Demographics**")
        age = st.slider("Age (years)", 30, 80, 55)
        gender = st.selectbox("Gender", ["Female", "Male"])

    with body_column:
        st.markdown("**Body measurements**")
        height = st.number_input("Height (cm)", 120, 220, 165)
        weight = st.number_input("Weight (kg)", 35.0, 200.0, 75.0, 0.5)

    with clinical_column:
        st.markdown("**Blood pressure**")
        systolic = st.number_input("Systolic pressure", 70, 250, 120)
        diastolic = st.number_input("Diastolic pressure", 40, 180, 80)

    st.divider()
    lifestyle_column, habit_column = st.columns(2)

    with lifestyle_column:
        st.markdown("**Cholesterol and glucose**")
        cholesterol = st.selectbox(
            "Cholesterol",
            ["Normal", "Above normal", "Well above normal"],
        )
        glucose = st.selectbox(
            "Glucose",
            ["Normal", "Above normal", "Well above normal"],
        )

    with habit_column:
        st.markdown("**Lifestyle**")
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol intake", ["No", "Yes"])
        active = st.selectbox("Physical activity", ["Yes", "No"])

    submitted = st.form_submit_button(
        "Analyse observation",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if systolic <= diastolic:
        st.error("Systolic pressure must be greater than diastolic pressure.")
    else:
        feature_row = make_feature_row(
            age,
            1 if gender == "Female" else 2,
            height,
            weight,
            systolic,
            diastolic,
            ["Normal", "Above normal", "Well above normal"].index(cholesterol) + 1,
            ["Normal", "Above normal", "Well above normal"].index(glucose) + 1,
            int(smoking == "Yes"),
            int(alcohol == "Yes"),
            int(active == "Yes"),
        )
        transformed = transform_frame(feature_row, config)
        tensor = torch.tensor(transformed, dtype=torch.float32)

        with torch.no_grad():
            score = float(torch.sigmoid(model(tensor)).item())

        predicted_positive = score >= threshold
        st.divider()
        st.markdown('<div class="section-label">Analysis result</div>', unsafe_allow_html=True)

        result_column, score_column = st.columns([1.15, 0.85])

        with result_column:
            if predicted_positive:
                st.error("Dataset class prediction: Positive (class 1)")
            else:
                st.success("Dataset class prediction: Negative (class 0)")

            st.markdown(
                '<p class="muted">This is the model output for the supplied dataset-style '
                'inputs. It is not a medical assessment.</p>',
                unsafe_allow_html=True,
            )

        with score_column:
            st.metric("Uncalibrated model score", f"{score:.1%}")
            st.progress(min(max(score, 0.0), 1.0))
            st.caption(f"Fixed validation-selected threshold: {threshold:.4f}")

        st.info(
            "The score is uncalibrated and is not a disease probability. "
            "A negative class prediction cannot rule out disease or replace medical advice."
        )


with st.expander("Evaluation design and limitations"):
    st.write(
        "The model is a PyTorch MLP with 128, 64, and 32 hidden units. "
        "The threshold was selected using out-of-fold training predictions, "
        "subject to a minimum precision constraint of 0.55."
    )
    st.write(
        "The final test partition was held out from preprocessing, model selection, "
        "epoch selection, and threshold selection."
    )
    st.write(
        "The dataset is cross-sectional. There is no external or temporal validation, "
        "calibration study, confidence interval analysis, subgroup fairness audit, "
        "or clinical validation."
    )

st.caption("CardioRisk AI · PyTorch MLP · Educational research use only")
