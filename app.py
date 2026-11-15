"""Streamlit research interface for the single high-recall PyTorch MLP."""
import json
from pathlib import Path
import streamlit as st, torch
from model import CardioMLP
from preprocessing import make_feature_row, transform_frame
ROOT=Path(__file__).resolve().parent; ART=ROOT/"artifacts"
st.set_page_config(page_title="Cardio MLP Research Demo",page_icon="🫀",layout="wide")
@st.cache_resource
def load_assets():
    cfg=json.loads((ART/"preprocessing.json").read_text()); report=json.loads((ART/"metrics.json").read_text())
    state=torch.load(ART/"mlp_state_dict.pt",map_location="cpu",weights_only=True)
    net=CardioMLP(cfg["input_dim"]); net.load_state_dict(state); net.eval()
    return cfg,report,net
try: config,report,model=load_assets()
except Exception as exc:
    st.error("Trained MLP artifacts are not available. Run the GitHub Actions workflow 'Train and publish high-recall MLP'.")
    st.exception(exc); st.stop()
point=report["operating_points"]["high_recall"]; metrics=point["test"]; threshold=float(point["threshold"])
st.title("🫀 Cardiovascular Dataset Classification")
st.caption("Single PyTorch MLP • Educational / research demonstration")
st.warning("Not a medical device. Not clinically validated. Do not use for diagnosis, treatment, screening, or care decisions.")
st.subheader("High-recall research operating point")
st.write("The threshold prioritizes recall subject to validation precision of at least 0.55. This accepts more false positives and is a research choice, not a validated clinical policy.")
a,b,c,d=st.columns(4)
a.metric("Test recall",f'{metrics["recall"]:.2%}'); b.metric("Test precision",f'{metrics["precision"]:.2%}')
c.metric("False negatives",f'{metrics["confusion_matrix"][1][0]:,}'); d.metric("False positives",f'{metrics["confusion_matrix"][0][1]:,}')
st.caption(f"Validation-selected threshold: {threshold:.4f}. Applied unchanged to the held-out test split.")
with st.sidebar:
    st.subheader("Model"); st.write("PyTorch MLP (128–64–32)"); st.write(f"Threshold: **{threshold:.4f}**")
    for label,key,fmt in [("Recall","recall",".2%"),("Precision","precision",".2%"),("F1","f1",".3f"),("Specificity","specificity",".2%"),("ROC-AUC","roc_auc",".3f"),("PR-AUC","pr_auc",".3f")]:
        st.metric(f"Test {label}",format(metrics[key],fmt))
st.subheader("Enter dataset-style values")
x,y,z=st.columns(3)
with x:
    age=st.slider("Age (years)",30,80,55); gender=st.selectbox("Gender",["Female","Male"])
    height=st.number_input("Height (cm)",120,220,165); weight=st.number_input("Weight (kg)",35.0,200.0,75.0,.5)
with y:
    sys=st.number_input("Systolic blood pressure",70,250,120); dia=st.number_input("Diastolic blood pressure",40,180,80)
    chol=st.selectbox("Cholesterol",["Normal","Above normal","Well above normal"])
    gluc=st.selectbox("Glucose",["Normal","Above normal","Well above normal"])
with z:
    smoke=st.selectbox("Smoking",["No","Yes"]); alco=st.selectbox("Alcohol intake",["No","Yes"])
    active=st.selectbox("Physical activity",["Yes","No"])
if st.button("Run MLP classification",type="primary",use_container_width=True):
    if sys<=dia: st.error("Systolic pressure must be greater than diastolic pressure.")
    else:
        row=make_feature_row(age,1 if gender=="Female" else 2,height,weight,sys,dia,
          ["Normal","Above normal","Well above normal"].index(chol)+1,
          ["Normal","Above normal","Well above normal"].index(gluc)+1,
          int(smoke=="Yes"),int(alco=="Yes"),int(active=="Yes"))
        values=transform_frame(row,config)
        with torch.no_grad(): score=float(torch.sigmoid(model(torch.tensor(values,dtype=torch.float32))).item())
        st.metric("Uncalibrated model score",f"{score:.1%}")
        st.write(f"Fixed high-recall threshold: **{threshold:.4f}**")
        if score>=threshold: st.error("Dataset class prediction: positive (class 1)")
        else: st.success("Dataset class prediction: negative (class 0)")
        st.info("This score is not a calibrated disease probability, diagnosis, or prediction of future disease. A negative result must not be used to rule out disease or delay medical care.")
with st.expander("Evaluation and limitations"):
    st.write("Confusion matrix (actual rows × predicted columns):",metrics["confusion_matrix"])
    st.write("Split sizes:",report["split_sizes"]); st.write("Threshold selection:",point["selection"])
    st.write("One public cross-sectional dataset; no external or temporal validation, calibration study, confidence intervals, subgroup fairness audit, or clinical validation.")
