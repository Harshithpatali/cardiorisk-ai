"""Premium Streamlit interface for the cardiovascular dataset MLP research demo."""

import json
from pathlib import Path

import pandas as pd

import streamlit as st
import torch

from model import CardioMLP
from preprocessing import make_feature_row, transform_frame
from explainability import (
    build_background, explain_observation, explanation_diagnostics, feature_names,
)


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="CardioRisk AI | Research Studio",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #10233f;
        --muted: #718096;
        --teal: #12b8a6;
        --coral: #ff806e;
        --line: #e5edf4;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 4% 0%, rgba(18,184,166,.10), transparent 26rem),
            radial-gradient(circle at 96% 8%, rgba(255,128,110,.10), transparent 25rem),
            #f4f7fb;
        color: var(--ink);
    }
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    .block-container {
        max-width: 1240px;
        padding-top: 1.4rem;
        padding-bottom: 3.5rem;
    }
    .topline {
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:1.15rem;
        color:#66758a;
        font-size:.82rem;
        font-weight:700;
        letter-spacing:.08em;
        text-transform:uppercase;
    }
    .brandmark {
        display:flex;
        align-items:center;
        gap:.6rem;
        color:#142b49;
        letter-spacing:.02em;
    }
    .brandicon {
        display:inline-flex;
        width:34px;
        height:34px;
        align-items:center;
        justify-content:center;
        border-radius:11px;
        color:white;
        background:linear-gradient(135deg,#12b8a6,#087f86);
        font-size:1.1rem;
    }
    .livepill {
        border:1px solid #bdebe4;
        background:#e8faf6;
        color:#087f75;
        border-radius:30px;
        padding:.38rem .75rem;
        letter-spacing:.02em;
        text-transform:none;
        font-size:.76rem;
    }
    .hero {
        position:relative;
        overflow:hidden;
        padding:2.45rem 2.6rem;
        border-radius:27px;
        background:linear-gradient(112deg,#10233f 0%,#173b56 58%,#087f86 100%);
        color:#fff;
        box-shadow:0 18px 45px rgba(16,35,63,.16);
        margin-bottom:1.4rem;
    }
    .hero:after {
        content:'';
        position:absolute;
        width:310px;
        height:310px;
        border:1px solid rgba(255,255,255,.14);
        border-radius:50%;
        right:7%;
        top:-155px;
        box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025);
    }
    .hero-kicker {
        color:#8de9dc;
        font-size:.75rem;
        font-weight:800;
        letter-spacing:.16em;
        text-transform:uppercase;
    }
    .hero h1 {
        color:#fff;
        font-family:'Manrope',sans-serif;
        font-size:clamp(2.4rem,5vw,3.65rem);
        line-height:1.08;
        font-weight:800;
        letter-spacing:-.055em;
        margin:.75rem 0 .8rem 0;
    }
    .hero p {
        color:#d7e6ee;
        max-width:650px;
        font-size:1.02rem;
        line-height:1.65;
        margin:0;
    }
    .hero-foot {
        display:flex;
        flex-wrap:wrap;
        gap:.55rem;
        margin-top:1.45rem;
    }
    .hero-tag {
        border:1px solid rgba(255,255,255,.22);
        border-radius:30px;
        padding:.38rem .75rem;
        color:#effafa;
        font-size:.78rem;
        background:rgba(255,255,255,.08);
    }
    .section-head {
        display:flex;
        align-items:flex-end;
        justify-content:space-between;
        gap:1rem;
        margin:1.9rem 0 .9rem 0;
    }
    .section-head h2 {
        color:#142b49;
        font-family:'Manrope',sans-serif;
        font-size:1.35rem;
        font-weight:800;
        letter-spacing:-.035em;
        margin:0;
    }
    .section-head p {
        color:#8492a6;
        font-size:.83rem;
        margin:0;
    }
    .metric-card {
        background:rgba(255,255,255,.9);
        border:1px solid #e6edf4;
        border-radius:18px;
        padding:1.05rem 1.15rem;
        min-height:116px;
        box-shadow:0 8px 24px rgba(23,49,78,.045);
    }
    .metric-top {
        display:flex;
        align-items:center;
        justify-content:space-between;
        color:#718096;
        font-size:.78rem;
        font-weight:700;
    }
    .metric-dot {
        width:8px;
        height:8px;
        border-radius:50%;
        display:inline-block;
        background:#12b8a6;
        box-shadow:0 0 0 4px #e2f8f4;
    }
    .metric-number {
        font-family:'Manrope',sans-serif;
        color:#142b49;
        font-size:1.9rem;
        line-height:1.25;
        font-weight:800;
        margin-top:.55rem;
        letter-spacing:-.04em;
    }
    .metric-foot {
        color:#9aa7b7;
        font-size:.72rem;
        margin-top:.1rem;
    }
    .notice {
        border:1px solid #f5d9b8;
        background:#fff8ed;
        color:#855b2e;
        border-radius:13px;
        padding:.85rem 1rem;
        font-size:.82rem;
        line-height:1.55;
        margin:1rem 0;
    }
    .form-shell {
        background:#fff;
        border:1px solid #e5edf4;
        border-radius:22px;
        padding:1.3rem 1.4rem;
        box-shadow:0 12px 35px rgba(23,49,78,.055);
    }
    .input-section {
        color:#173b56;
        font-family:'Manrope',sans-serif;
        font-weight:800;
        font-size:.98rem;
        padding:.15rem 0 .55rem 0;
        border-bottom:1px solid #edf1f5;
        margin-bottom:.7rem;
    }
    .input-hint {
        color:#94a3b8;
        font-size:.76rem;
        margin-top:-.35rem;
        margin-bottom:.75rem;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[role="combobox"] {
        border-radius:11px;
    }
    div[data-testid="stSlider"] {
        padding-top:.2rem;
    }
    div[data-testid="stForm"] {
        background:transparent;
        border:0;
        padding:0;
    }
    div[data-testid="stFormSubmitButton"] button {
        border:0;
        border-radius:13px;
        min-height:3.25rem;
        color:white;
        background:linear-gradient(100deg,#0c938e,#12b8a6);
        font-size:.98rem;
        font-weight:800;
        box-shadow:0 8px 20px rgba(18,184,166,.22);
        transition:transform .18s ease,box-shadow .18s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform:translateY(-2px);
        box-shadow:0 12px 24px rgba(18,184,166,.28);
        color:white;
    }
    .result-panel {
        border-radius:21px;
        padding:1.5rem;
        background:#fff;
        border:1px solid #e5edf4;
        box-shadow:0 12px 35px rgba(23,49,78,.06);
    }
    .result-positive {
        border-left:5px solid #ff806e;
        background:linear-gradient(110deg,#fff4f1,#fff);
    }
    .result-negative {
        border-left:5px solid #12b8a6;
        background:linear-gradient(110deg,#edfbf8,#fff);
    }
    .result-label {
        color:#718096;
        font-size:.75rem;
        font-weight:800;
        text-transform:uppercase;
        letter-spacing:.12em;
    }
    .result-heading {
        color:#142b49;
        font-family:'Manrope',sans-serif;
        font-size:1.65rem;
        font-weight:800;
        letter-spacing:-.04em;
        margin:.45rem 0;
    }
    .result-copy {
        color:#718096;
        font-size:.88rem;
        line-height:1.55;
    }
    .score-number {
        color:#142b49;
        font-family:'Manrope',sans-serif;
        font-size:2.35rem;
        font-weight:800;
        letter-spacing:-.05em;
    }
    .stProgress > div > div > div > div {
        background:linear-gradient(90deg,#12b8a6,#ff806e);
    }
    [data-testid="stExpander"] {
        background:#fff;
        border:1px solid #e5edf4;
        border-radius:14px;
    }
    .explain-intro {
        display:flex;
        align-items:center;
        gap:1rem;
        padding:1.15rem 1.25rem;
        margin:.35rem 0 1rem 0;
        border:1px solid #dcece9;
        border-radius:18px;
        background:linear-gradient(105deg,#effaf8,#fff 75%);
    }
    .explain-icon {
        display:flex;
        align-items:center;
        justify-content:center;
        width:44px;
        height:44px;
        flex:0 0 44px;
        border-radius:14px;
        color:#087f75;
        background:#d8f5ef;
        font-size:1.35rem;
    }
    .explain-title {
        color:#142b49;
        font-family:'Manrope',sans-serif;
        font-size:1.02rem;
        font-weight:800;
    }
    .explain-subtitle {
        color:#718096;
        font-size:.79rem;
        line-height:1.5;
        margin-top:.2rem;
    }
    .impact-card {
        padding:.9rem 1.05rem;
        margin:.62rem 0;
        border:1px solid #e5edf4;
        border-radius:15px;
        background:#fff;
        box-shadow:0 5px 16px rgba(23,49,78,.035);
    }
    .impact-top {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:.7rem;
        margin-bottom:.62rem;
    }
    .impact-name {
        display:flex;
        align-items:center;
        gap:.65rem;
        color:#1b3552;
        font-size:.9rem;
        font-weight:800;
    }
    .impact-rank {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width:26px;
        height:26px;
        border-radius:8px;
        color:#6d8197;
        background:#f0f4f8;
        font-size:.68rem;
        font-weight:800;
    }
    .impact-direction {
        font-size:.72rem;
        font-weight:800;
        text-align:right;
    }
    .impact-track {
        width:100%;
        height:8px;
        overflow:hidden;
        border-radius:20px;
        background:#edf2f6;
    }
    .impact-fill {
        height:100%;
        border-radius:20px;
    }
    .impact-value {
        color:#8a98a9;
        font-size:.7rem;
        margin-top:.48rem;
    }
    .impact-value strong {
        color:#42566d;
        font-variant-numeric:tabular-nums;
    }
    .explain-note {
        padding:1rem 1.1rem;
        margin-top:1rem;
        border-radius:14px;
        color:#64758a;
        background:#f0f4f8;
        font-size:.78rem;
        line-height:1.6;
    }
    .explain-note strong {color:#30465e;}
    @media(max-width:700px) {
        .hero {padding:1.7rem 1.4rem;}
        .hero:after {right:-180px;}
        .block-container {padding-top:1rem;}
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


@st.cache_resource
def load_explanation_background(config_signature):
    config, _, _ = load_assets()
    return build_background(ROOT / "data" / "cardio_train.csv", config)


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
    <div class="topline">
        <div class="brandmark"><span class="brandicon">✚</span> CARDIORISK AI</div>
        <div class="livepill">● &nbsp; Research environment</div>
    </div>
    <div class="hero">
        <div class="hero-kicker">Intelligent health data · Research studio</div>
        <h1>Understand the<br>signals in your data.</h1>
        <p>A high-recall neural-network interface for exploring cardiovascular dataset patterns, with a separate two-tier verification research study.
        Enter an observation, run the model, and inspect its output in seconds.</p>
        <div class="hero-foot">
            <span class="hero-tag">PyTorch MLP · 128 / 64 / 32</span>
            <span class="hero-tag">5-fold cross-validation</span>
            <span class="hero-tag">Leakage-aware evaluation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        <strong>Research use only.</strong> This application is not clinically validated and
        must not be used for diagnosis, treatment, screening, or healthcare decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-head">
        <div><h2>Model snapshot</h2><p>Held-out test-set performance</p></div>
        <p>High-recall operating point</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_values = [
    ("Recall", f'{metrics["recall"]:.1%}', "Sensitivity on held-out test data"),
    ("False negatives", f'{metrics["confusion_matrix"][1][0]:,}', "Actual positive rows missed"),
    ("Precision", f'{metrics["precision"]:.1%}', "Positive prediction quality"),
    ("Specificity", f'{metrics["specificity"]:.1%}', "Actual negative rows correctly rejected"),
]

metric_columns = st.columns(4)

for column, (label, value, foot) in zip(metric_columns, metric_values):
    column.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top">{label}<span class="metric-dot"></span></div>
            <div class="metric-number">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



@st.cache_data
def load_research_report():
    path = ARTIFACTS / "research_validation.json"
    return json.loads(path.read_text()) if path.exists() else {}

@st.cache_data
def load_operating_frontier():
    path = ARTIFACTS / "mlp_operating_point_frontier.json"
    return json.loads(path.read_text()) if path.exists() else {"rows": []}

@st.cache_data
def load_tree_cascade_report():
    path = ARTIFACTS / "tree_stage2_cascade_comparison.json"
    return json.loads(path.read_text()) if path.exists() else {}

research_report = load_research_report()
frontier_report = load_operating_frontier()
tree_cascade_report = load_tree_cascade_report()
st.markdown(
    """
    <div class="section-head">
        <div><h2>Research analytics workspace</h2>
        <p>Cross-validation, operating-point analysis and model diagnostics</p></div>
        <p>MODEL INTELLIGENCE / 360°</p>
    </div>
    """, unsafe_allow_html=True,
)
cascade_tab, overview_tab, threshold_tab, calibration_tab, subgroup_tab, benchmark_tab, specificity_tab, boosting_tab = st.tabs(
    ["Two-tier study", "Overview", "Threshold lab", "Calibration", "Subgroups", "Benchmark", "Specificity study", "Boosting & calibration"]
)

with cascade_tab:
    st.subheader("High-recall MLP + selective tree verification")
    st.caption("Research comparison only. The live single-observation form below still uses the deployed PyTorch MLP alone.")
    if tree_cascade_report and tree_cascade_report.get("models"):
        st.warning(
            "False negatives are treated as the more costly error in this research design. "
            "A Stage-2 candidate is not preferred merely because precision rises: check that "
            "its untouched-test false-negative count does not increase."
        )
        gate = float(tree_cascade_report.get("stage1_gate_threshold", float("nan")))
        gate_recall = float(tree_cascade_report.get("stage1_gate_oof_recall", float("nan")))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stage-1 gate", f"{gate:.4f}")
        m2.metric("Stage-1 OOF recall", f"{gate_recall:.2%}")
        m3.metric("Stage-1 test recall", f'{tree_cascade_report.get("stage1_gate_test_recall", float("nan")):.2%}')
        m4.metric("Stage-1 test misses below gate", f'{tree_cascade_report.get("stage1_gate_test_false_negatives", 0):,}')
        st.caption(f'Borderline development rows sent to Stage 2: {tree_cascade_report["models"][0]["stage2_development_rows"]:,}. Rows below the gate cannot be recovered by Stage 2.')
        st.markdown("#### How the two tiers route an observation")
        route_a, route_b, route_c = st.columns(3)
        route_a.info(f"**1 · MLP screen**\n\nBelow {gate:.4f}: Stage-1 negative. The second model cannot recover this case.")
        route_b.warning(f"**2 · Borderline band**\n\nFrom {gate:.4f} to below 0.50: send the row to XGBoost or LightGBM.")
        route_c.success("**3 · Direct positive**\n\nAt or above 0.50: bypass Stage 2 and retain the positive classification.")
        st.markdown(
            "For routed rows, Stage 2 returns a calibrated model score. The final positive rule is "
            "**direct Stage-1 positive OR (borderline AND Stage-2 score above its development-selected threshold)**. "
            "The score is not a literal individual precision or a clinically validated probability."
        )
        target_values = sorted({
            float(row["recall_target"])
            for model_result in tree_cascade_report["models"]
            for row in model_result.get("recall_target_results", [])
        })
        preferred = 0.995 if 0.995 in target_values else max(target_values)
        target = st.select_slider(
            "Development OOF end-to-end recall constraint",
            options=target_values,
            value=preferred,
            format_func=lambda value: f"{value:.1%}",
            key="tree_cascade_recall_target",
        )
        by_model = {item["model"]: item for item in tree_cascade_report["models"]}
        def get_result(model_name):
            return next(row for row in by_model[model_name]["recall_target_results"]
                        if float(row["recall_target"]) == float(target))
        light = get_result("lightgbm")
        xgb = get_result("xgboost")
        baseline_xgb = {
            "model": "MLP · XGB-matched",
            "precision": xgb["baseline_test_precision"],
            "recall": xgb["baseline_test_recall"],
            "specificity": xgb["baseline_test_specificity"],
            "f1": xgb["baseline_test_f1"],
            "fp": xgb["baseline_test_false_positives"],
            "fn": xgb["baseline_test_false_negatives"],
            "threshold": None,
        }
        baseline_light = {
            "model": "MLP · LGBM-matched",
            "precision": light["baseline_test_precision"],
            "recall": light["baseline_test_recall"],
            "specificity": light["baseline_test_specificity"],
            "f1": light["baseline_test_f1"],
            "fp": light["baseline_test_false_positives"],
            "fn": light["baseline_test_false_negatives"],
            "threshold": None,
        }
        comparison_rows = [
            baseline_xgb,
            {
                "model": "XGBoost cascade",
                "precision": xgb["cascade_test_precision"],
                "recall": xgb["cascade_test_recall"],
                "specificity": xgb["cascade_test_specificity"],
                "f1": xgb["cascade_test_f1"],
                "fp": xgb["cascade_test_false_positives"],
                "fn": xgb["cascade_test_false_negatives"],
                "threshold": xgb["stage2_threshold"],
            },
            baseline_light,
            {
                "model": "LightGBM cascade",
                "precision": light["cascade_test_precision"],
                "recall": light["cascade_test_recall"],
                "specificity": light["cascade_test_specificity"],
                "f1": light["cascade_test_f1"],
                "fp": light["cascade_test_false_positives"],
                "fn": light["cascade_test_false_negatives"],
                "threshold": light["stage2_threshold"],
            },
        ]
        compare = pd.DataFrame(comparison_rows)
        st.markdown("#### Untouched-test comparison")
        st.caption("All rows use the same held-out test observations. The baseline is matched to each cascade's development OOF recall.")
        st.dataframe(
            compare.style.format({
                "precision": "{:.2%}", "recall": "{:.2%}",
                "specificity": "{:.2%}", "f1": "{:.3f}",
                "fp": "{:,}", "fn": "{:,}", "threshold": lambda value: "—" if pd.isna(value) else f"{value:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("#### Performance profile")
        st.bar_chart(compare.set_index("model")[["recall", "precision", "specificity"]])
        st.markdown("#### Error counts — false negatives deserve priority")
        error_fp, error_fn = st.columns(2)
        with error_fp:
            st.caption("False positives")
            st.bar_chart(compare.set_index("model")[["fp"]])
        with error_fn:
            st.caption("False negatives")
            st.bar_chart(compare.set_index("model")[["fn"]])
        for label, row, matched_baseline in [
            ("XGBoost", xgb, baseline_xgb),
            ("LightGBM", light, baseline_light),
        ]:
            delta_fn = int(row["cascade_test_false_negatives"]) - int(matched_baseline["fn"])
            delta_fp = int(row["cascade_test_false_positives"]) - int(matched_baseline["fp"])
            if delta_fn > 0:
                st.error(f"{label}: {delta_fn:,} additional false negatives and {abs(delta_fp):,} fewer false positives than the matched MLP. This is not an acceptable trade-off under a false-negative-first policy.")
            elif delta_fn == 0 and delta_fp < 0:
                st.success(f"{label}: same false-negative count as the matched MLP and {abs(delta_fp):,} fewer false positives on this test set.")
            elif delta_fn < 0:
                st.success(f"{label}: {abs(delta_fn):,} fewer false negatives than the matched MLP.")
            else:
                st.info(f"{label}: no false-negative improvement; false-positive change = {delta_fp:+,}.")
        st.markdown("#### Interpretation")
        st.write(
            f"At the {target:.1%} development constraint, the LightGBM cascade has "
            f"{int(light['cascade_test_false_negatives']):,} false negatives versus "
            f"{int(light['baseline_test_false_negatives']):,} for the matched MLP, and "
            f"{int(light['cascade_test_false_positives']):,} false positives versus "
            f"{int(light['baseline_test_false_positives']):,}. The untouched test result is "
            "descriptive; it must not be used to retune thresholds."
        )
        st.info(
            "Conservative reading: the 99.5% operating point is the most appropriate comparison "
            "for this project's stated error preference. In the current test results, LightGBM "
            "matches the MLP's 12 false negatives and removes only 5 false positives. That is "
            "a very small gain, not evidence to replace the deployed MLP."
        )
        st.caption(
            "Development OOF threshold selection; one untouched test evaluation; 2,000 paired "
            "bootstrap resamples. Bootstrap intervals are conditional on fitted models and "
            "selected thresholds and do not include training/selection uncertainty."
        )
    else:
        st.info("The two-tier comparison report is not available. Run the GitHub Actions training workflow.")

with overview_tab:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Held-out model performance")
        rows = [
            {"Metric": "ROC-AUC", "Value": metrics["roc_auc"]},
            {"Metric": "PR-AUC", "Value": metrics["pr_auc"]},
            {"Metric": "Recall", "Value": metrics["recall"]},
            {"Metric": "Specificity", "Value": metrics["specificity"]},
            {"Metric": "Precision", "Value": metrics["precision"]},
            {"Metric": "F1", "Value": metrics["f1"]},
            {"Metric": "Balanced accuracy", "Value": (metrics["recall"] + metrics["specificity"]) / 2},
            {"Metric": "Brier score", "Value": metrics.get("brier_score", float("nan"))},
            {"Metric": "ECE · 10 bins", "Value": metrics.get("ece_10_bins", float("nan"))},
        ]
        st.dataframe(pd.DataFrame(rows).style.format({"Value": "{:.4f}"}), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
            index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]),
            use_container_width=True)
        st.caption("Rows = actual labels · columns = predicted labels")
        st.metric("Selected threshold", f"{threshold:.4f}")
    folds = report.get("cross_validation", {}).get("folds", [])
    if folds:
        st.subheader("Five-fold MLP validation")
        frame = pd.DataFrame([{"Fold": x["fold"], "Rows": x["rows"],
            "ROC-AUC": x["metrics_at_0_5"]["roc_auc"],
            "PR-AUC": x["metrics_at_0_5"]["pr_auc"],
            "Selected epochs": x["selected_epochs"]} for x in folds])
        st.dataframe(frame.style.format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}"}),
                     use_container_width=True, hide_index=True)
with threshold_tab:
    st.subheader("Recall-constrained operating-point explorer")
    st.caption("Explore the highest development OOF threshold meeting a recall target. This does not change deployed inference.")
    rows = frontier_report.get("rows", [])
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame[["recall_floor", "threshold", "recall", "specificity",
            "precision", "f1", "balanced_accuracy", "fp", "fn"]].style.format({
            c: "{:.3f}" for c in ["recall_floor", "threshold", "recall", "specificity",
                                  "precision", "f1", "balanced_accuracy"]}),
            use_container_width=True, hide_index=True)
        floor = st.select_slider("Minimum recall target", options=[0.80, 0.85, 0.90, 0.93, 0.95],
                                 value=0.90, format_func=lambda x: f"{x:.0%}")
        selected = min(rows, key=lambda x: abs(x["recall_floor"] - floor))
        a,b,c,d = st.columns(4)
        a.metric("OOF threshold", f'{selected["threshold"]:.4f}')
        b.metric("OOF recall", f'{selected["recall"]:.1%}')
        c.metric("OOF specificity", f'{selected["specificity"]:.1%}')
        d.metric("OOF F1", f'{selected["f1"]:.3f}')
        st.info(f'OOF confusion counts · TN {selected["tn"]:,} · FP {selected["fp"]:,} · FN {selected["fn"]:,} · TP {selected["tp"]:,}')
        st.line_chart(frame.set_index("recall_floor")[["recall", "specificity", "precision", "f1"]])
    else:
        st.warning("Operating-point frontier is not available yet.")
with calibration_tab:
    st.subheader("Probability calibration diagnostics")
    bins = research_report.get("calibration", {}).get("reliability_bins", [])
    if bins:
        frame = pd.DataFrame(bins)
        st.line_chart(frame.set_index("mean_score")[["observed_rate"]])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.caption("Development benchmark reliability bins; not a clinical calibration study.")
    else:
        st.info("Calibration bins are not available.")
with subgroup_tab:
    st.subheader("Descriptive subgroup diagnostics")
    groups = research_report.get("deployed_mlp_oof_analysis", {}).get("subgroup_analysis", {})
    if groups:
        rows = []
        for dimension, members in groups.items():
            for name, values in members.items():
                rows.append({"Dimension": dimension.replace("_", " ").title(),
                    "Group": name.replace("_", " ").title(), "N": values.get("n"),
                    "Prevalence": values.get("prevalence"), "Recall": values.get("recall"),
                    "Specificity": values.get("specificity"), "Precision": values.get("precision"),
                    "Balanced accuracy": values.get("balanced_accuracy")})
        frame = pd.DataFrame(rows)
        st.dataframe(frame.style.format({k: "{:.1%}" for k in
            ["Prevalence", "Recall", "Specificity", "Precision", "Balanced accuracy"]}),
            use_container_width=True, hide_index=True)
        st.warning("Descriptive comparisons only; these do not establish fairness or clinical equivalence.")
    else:
        st.info("MLP subgroup diagnostics are not available.")
with benchmark_tab:
    st.subheader("Cross-validated model benchmark")
    variant_report_path = ARTIFACTS / "mlp_variant_benchmark.json"
    if variant_report_path.exists():
        variant_report = json.loads(variant_report_path.read_text())
        st.markdown("#### Experimental MLP architecture comparison")
        variant_rows = [{
            "Variant": item["variant"], "Architecture": "–".join(map(str, item["hidden_dims"])),
            "Dropout": item["dropout"], "Positive weight multiplier": item["positive_weight_multiplier"],
            "OOF ROC-AUC": item["oof_roc_auc"], "OOF PR-AUC": item["oof_pr_auc"],
        } for item in variant_report.get("variants", [])]
        if variant_rows:
            st.dataframe(pd.DataFrame(variant_rows).style.format({
                "Dropout": "{:.2f}", "Positive weight multiplier": "{:.2f}",
                "OOF ROC-AUC": "{:.3f}", "OOF PR-AUC": "{:.3f}",
            }), use_container_width=True, hide_index=True)
            st.caption("Experimental three-fold development OOF results. No candidate is promoted automatically.")
    models = research_report.get("models", {})
    if models:
        rows = []
        for name, values in models.items():
            stats = values.get("fold_mean_95ci", {})
            rows.append({"Model": name.replace("_", " ").title(),
                "ROC-AUC": stats.get("roc_auc", {}).get("mean"),
                "ROC-AUC CI low": stats.get("roc_auc", {}).get("ci95_low"),
                "ROC-AUC CI high": stats.get("roc_auc", {}).get("ci95_high"),
                "PR-AUC": stats.get("pr_auc", {}).get("mean"),
                "Brier": stats.get("brier_score", {}).get("mean"),
                "ECE": stats.get("ece_10_bins", {}).get("mean")})
        st.dataframe(pd.DataFrame(rows).style.format({
            k: "{:.3f}" for k in ["ROC-AUC", "ROC-AUC CI low", "ROC-AUC CI high", "PR-AUC", "Brier", "ECE"]}),
            use_container_width=True, hide_index=True)
        st.caption("Development-only reference models. The deployed PyTorch MLP remains the inference model.")
    else:
        st.info("Benchmark report is not available.")

with specificity_tab:
    st.subheader("Specificity at matched recall")
    st.caption("Development-only five-fold OOF comparison. Each candidate gets its own threshold to meet the same recall floor; this is not a deployment setting.")
    specificity_path = ARTIFACTS / "specificity_investigation.json"
    if specificity_path.exists():
        specificity_report = json.loads(specificity_path.read_text())
        rows = specificity_report.get("candidates", [])
        if rows:
            chosen_floor = st.select_slider(
                "Recall floor for comparison",
                options=[0.85, 0.90, 0.95],
                value=0.90,
                format_func=lambda value: f"{value:.0%}",
                key="specificity_recall_floor",
            )
            selected_rows = [row for row in rows if row["recall_floor"] == chosen_floor]
            comparison = pd.DataFrame([{
                "Candidate": row["candidate"].replace("_", " ").title(),
                "Threshold": row["threshold"],
                "Recall": row["recall"],
                "Specificity": row["specificity"],
                "Precision": row["precision"],
                "PR-AUC": row.get("pr_auc"),
                "ROC-AUC": row.get("roc_auc"),
                "Specificity CI low": row["specificity_fold_ci95"]["ci95_low"],
                "Specificity CI high": row["specificity_fold_ci95"]["ci95_high"],
            } for row in selected_rows]).sort_values("Specificity", ascending=False)
            st.dataframe(comparison.style.format({
                column: "{:.3f}" for column in comparison.columns if column != "Candidate"
            }), use_container_width=True, hide_index=True)
            st.bar_chart(comparison.set_index("Candidate")[["Specificity", "Recall", "Precision"]])
            best = comparison.iloc[0]
            st.metric("Highest observed specificity at this recall floor",
                      f'{best["Specificity"]:.1%}', best["Candidate"])
            st.caption("Specificity intervals are descriptive t-intervals across five held-out folds at the pooled-OOF-selected threshold. They do not include threshold-selection uncertainty.")
        else:
            st.info("Specificity study has no candidate results.")
    else:
        st.info("Run the latest training workflow to generate the specificity investigation.")


with boosting_tab:
    st.subheader("CatBoost and XGBoost · calibration study")
    st.caption("Five-fold development-only OOF comparison. Sigmoid calibrators are fitted inside each outer training fold; these candidates are not deployed.")
    study_path = ARTIFACTS / "boosting_calibration_study.json"
    if study_path.exists():
        study = json.loads(study_path.read_text())
        model_rows = [{
            "Model": f'{item["model"].replace("_", " ").title()} · {item["probability_type"].replace("_", " ").title()}',
            "ROC-AUC": item["roc_auc"], "PR-AUC": item["pr_auc"],
            "Brier": item["brier_score"], "ECE": item["ece_10_bins"],
        } for item in study.get("models", [])]
        if model_rows:
            frame = pd.DataFrame(model_rows)
            st.dataframe(frame.style.format({
                "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
                "Brier": "{:.3f}", "ECE": "{:.3f}",
            }), use_container_width=True, hide_index=True)
            st.markdown("#### Specificity at matched recall")
            floors = sorted({row["recall_floor"] for row in study.get("operating_points", [])}, reverse=True)
            if floors:
                floor = st.select_slider(
                    "Minimum recall target", options=floors, value=0.90,
                    format_func=lambda value: f"{value:.0%}",
                    key="boosting_recall_floor",
                )
                selected = [row for row in study["operating_points"] if row["recall_floor"] == floor]
                comparison = pd.DataFrame([{
                    "Model": f'{row["model"].replace("_", " ").title()} · {row["probability_type"].replace("_", " ").title()}',
                    "Recall": row["recall"], "Specificity": row["specificity"],
                    "Precision": row["precision"], "Threshold": row["threshold"],
                    "Brier": next((m["brier_score"] for m in study["models"]
                        if m["model"] == row["model"] and m["probability_type"] == row["probability_type"]), None),
                    "ECE": next((m["ece_10_bins"] for m in study["models"]
                        if m["model"] == row["model"] and m["probability_type"] == row["probability_type"]), None),
                } for row in selected]).sort_values("Specificity", ascending=False)
                st.dataframe(comparison.style.format({
                    c: "{:.3f}" for c in comparison.columns if c != "Model"
                }), use_container_width=True, hide_index=True)
                st.bar_chart(comparison.set_index("Model")[["Specificity", "Recall", "Precision"]])
            st.caption("Brier score and ECE summarize probability reliability. Lower is better, but neither establishes clinical calibration.")
        else:
            st.info("The boosting study contains no model summaries.")
    else:
        st.info("The CatBoost/XGBoost study will appear after the research workflow completes.")
    st.warning("All results are exploratory development OOF estimates. Pooled-OOF threshold selection is optimistic; independent confirmation is required. The deployed PyTorch MLP remains unchanged.")

st.markdown(
    """
    <div class="section-head">
        <div><h2>Explore an observation</h2><p>Provide dataset-style attributes below</p></div>
        <p>01 / INPUT</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "To test a new observation, edit the values in the form below and press "
    "**Run cardiovascular data analysis**. Each submission runs the deployed "
    "single-stage MLP; the experimental XGBoost/LightGBM cascade is currently "
    "shown as an evaluation study and is not used for individual predictions."
)
with st.form("cardio_observation_form"):
    st.caption("Example starting point: age 55 · height 165 cm · weight 75 kg · blood pressure 120/80.")
    first_column, second_column, third_column = st.columns(3)

    with first_column:
        st.markdown('<div class="input-section">01 &nbsp; Demographics</div>', unsafe_allow_html=True)
        st.markdown('<div class="input-hint">Basic profile information</div>', unsafe_allow_html=True)
        age = st.slider("Age · years", 30, 80, 55)
        gender = st.selectbox("Gender", ["Female", "Male"])

    with second_column:
        st.markdown('<div class="input-section">02 &nbsp; Body profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="input-hint">Height and weight measurements</div>', unsafe_allow_html=True)
        height = st.number_input("Height · cm", 120, 220, 165)
        weight = st.number_input("Weight · kg", 35.0, 200.0, 75.0, 0.5)

    with third_column:
        st.markdown('<div class="input-section">03 &nbsp; Blood pressure</div>', unsafe_allow_html=True)
        st.markdown('<div class="input-hint">Systolic and diastolic readings</div>', unsafe_allow_html=True)
        systolic = st.number_input("Systolic · mmHg", 70, 250, 120)
        diastolic = st.number_input("Diastolic · mmHg", 40, 180, 80)

    st.markdown("---")
    lifestyle_column, habits_column = st.columns(2)

    with lifestyle_column:
        st.markdown('<div class="input-section">04 &nbsp; Clinical indicators</div>', unsafe_allow_html=True)
        cholesterol = st.selectbox(
            "Cholesterol level",
            ["Normal", "Above normal", "Well above normal"],
        )
        glucose = st.selectbox(
            "Glucose level",
            ["Normal", "Above normal", "Well above normal"],
        )

    with habits_column:
        st.markdown('<div class="input-section">05 &nbsp; Lifestyle</div>', unsafe_allow_html=True)
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol intake", ["No", "Yes"])
        active = st.selectbox("Physical activity", ["Yes", "No"])

    st.markdown("")
    submitted = st.form_submit_button(
        "✦  Run cardiovascular data analysis",
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

        st.markdown(
            """
            <div class="section-head">
                <div><h2>Analysis output</h2><p>02 / MODEL RESULT</p></div>
                <p>Threshold locked from OOF validation</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result_column, score_column = st.columns([1.2, 0.8])

        with result_column:
            result_class = "result-positive" if predicted_positive else "result-negative"
            result_text = "Positive class" if predicted_positive else "Negative class"
            result_detail = (
                "The model output is above the selected classification threshold."
                if predicted_positive
                else "The model output is below the selected classification threshold."
            )
            result_icon = "↗" if predicted_positive else "✓"
            st.markdown(
                f"""
                <div class="result-panel {result_class}">
                    <div class="result-label">Dataset classification</div>
                    <div class="result-heading">{result_icon} &nbsp; {result_text}</div>
                    <div class="result-copy">{result_detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with score_column:
            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-label">Uncalibrated model score</div>
                    <div class="score-number">{score:.1%}</div>
                    <div class="result-copy">Model output score · not a clinical probability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(score, 0.0), 1.0))
            st.caption(f"Selected threshold: {threshold:.4f}")

        st.info(
            "This score is uncalibrated and is not a disease probability. "
            "A negative class prediction cannot rule out disease or replace medical advice. False negatives are especially important in this research context; never use this output to delay professional evaluation."
        )

        st.markdown(
            """
            <div class="section-head">
                <div><h2>What influenced this result?</h2>
                <p>03 / LOCAL MODEL EXPLANATION</p></div>
                <p>Integrated Gradients · this observation</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            with st.spinner("Calculating feature contributions..."):
                background = load_explanation_background("cardio-background-v1")
                contributions = explain_observation(model, background, transformed)
                labels = feature_names(config)
            diagnostics = explanation_diagnostics(
                model, background, transformed, contributions
            )

            if len(contributions) != len(labels):
                raise ValueError("Integrated Gradients attributions do not match the saved feature schema.")

            ranked = sorted(
                zip(labels, contributions),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:8]
            max_impact = max((abs(float(value)) for _, value in ranked), default=0.0)

            st.markdown(
                """
                <div class="explain-intro">
                    <div class="explain-icon">✦</div>
                    <div>
                        <div class="explain-title">The strongest signals behind this model output</div>
                        <div class="explain-subtitle">Features are ranked by their contribution magnitude.
                        Coral pushes the model score upward; teal pushes it downward.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for rank, (label, value) in enumerate(ranked, start=1):
                value = float(value)
                direction = "Raises model output" if value > 0 else "Lowers model output"
                color = "#f07868" if value > 0 else "#0c9b91"
                width = 0.0 if max_impact == 0 else abs(value) / max_impact * 100.0
                st.markdown(
                    f"""
                    <div class="impact-card">
                        <div class="impact-top">
                            <div class="impact-name"><span class="impact-rank">{rank:02d}</span>{label}</div>
                            <div class="impact-direction" style="color:{color}">{direction}</div>
                        </div>
                        <div class="impact-track">
                            <div class="impact-fill" style="width:{width:.1f}%;background:{color}"></div>
                        </div>
                        <div class="impact-value">Attribution&nbsp; <strong>{value:+.4f}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                f"Integrated Gradients completeness check · absolute residual: {diagnostics['absolute_residual']:.6f}. "
                "This is a numerical consistency diagnostic, not clinical validation."
            )

            st.markdown(
                """
                <div class="explain-note">
                    <strong>How to read this:</strong> these are model attributions, not causes of
                    cardiovascular disease. Correlated inputs and engineered features can share
                    attribution. Integrated Gradients explains this model's output; it does not establish causation
                    or clinical importance.
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.warning("The prediction is available, but its feature explanation could not be generated.")
            st.caption(str(exc))


st.caption(
    "Live prediction path: deployed PyTorch MLP only. The XGBoost/LightGBM second-stage "
    "models are research comparisons and are not called by this form."
)

with st.expander("How this model was evaluated"):
    st.write(
        "The model is a PyTorch MLP with 128, 64, and 32 hidden units. "
        f"The threshold was selected using out-of-fold development predictions, "
        f"maximizing recall subject to a minimum precision guardrail of "
        f"{report['threshold_selection']['minimum_precision']:.0%}."
    )
    st.write(
        "The final test partition was held out from preprocessing, model selection, "
        "epoch selection, and threshold selection."
    )
    st.write(
        f"High-recall threshold: {threshold:.4f}. Test specificity: {metrics['specificity']:.1%}; "
        f"balanced accuracy: {(metrics['recall'] + metrics['specificity']) / 2:.1%}. "
        f"Confusion matrix (actual rows, predicted columns): {metrics['confusion_matrix']}."
    )
    st.write(
        "The dataset is cross-sectional. There is no external or temporal validation, "
        "calibration study, confidence interval analysis, subgroup fairness audit, "
        "or clinical validation."
    )

st.markdown(
    """
    <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #e2eaf1;
    color:#94a3b8;font-size:.75rem;display:flex;justify-content:space-between;flex-wrap:wrap;">
        <span>CardioRisk AI · Research Studio</span>
        <span>Built with PyTorch & Streamlit · Educational use only</span>
    </div>
    """,
    unsafe_allow_html=True,
)
