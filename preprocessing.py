"""Dataset cleaning and JSON-configured feature transformations."""
import numpy as np
import pandas as pd
NUMERIC_COLUMNS=["age_years","height","weight","ap_hi","ap_lo","bmi","pulse_pressure","map"]
CATEGORICAL_COLUMNS=["gender","cholesterol","gluc"]
BINARY_COLUMNS=["smoke","alco","active"]
FEATURE_COLUMNS=NUMERIC_COLUMNS+CATEGORICAL_COLUMNS+BINARY_COLUMNS
def clean_dataset(path):
    df=pd.read_csv(path,sep=";")
    required={"id","age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active","cardio"}
    missing=required.difference(df.columns)
    if missing: raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df["age_years"]=(df["age"]/365.25).round(1); df["bmi"]=df["weight"]/((df["height"]/100)**2)
    df["pulse_pressure"]=df["ap_hi"]-df["ap_lo"]; df["map"]=df["ap_lo"]+df["pulse_pressure"]/3
    bp=df["ap_hi"].between(70,250)&df["ap_lo"].between(40,180)&(df["ap_hi"]>df["ap_lo"])
    body=df["height"].between(120,220)&df["weight"].between(35,200)&df["bmi"].between(12,60)
    out=df.loc[bp&body].drop(columns=["id","age"]).copy()
    if out.empty: raise ValueError("No rows remain after data-quality filters.")
    return out
def transform_frame(frame,config):
    x=frame[config["feature_columns"]].copy()
    num=x[config["numeric_columns"]].to_numpy(dtype=np.float64)
    mean=np.asarray(config["scaler_mean"],dtype=np.float64); scale=np.asarray(config["scaler_scale"],dtype=np.float64)
    num=(num-mean)/np.where(scale==0,1,scale)
    blocks=[]
    for col,cats in zip(config["categorical_columns"],config["categories"]):
        vals=x[col].to_numpy(); blocks.append(np.column_stack([(vals==v).astype(float) for v in cats]))
    binary=x[config["binary_columns"]].to_numpy(dtype=np.float64)
    result=np.concatenate([num,*blocks,binary],axis=1).astype(np.float32)
    if result.shape[1]!=int(config["input_dim"]): raise ValueError("Feature dimension mismatch")
    if not np.isfinite(result).all(): raise ValueError("Non-finite model input")
    return result
def make_feature_row(age_years,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active):
    pp=float(ap_hi)-float(ap_lo)
    return pd.DataFrame([{"age_years":float(age_years),"height":float(height),"weight":float(weight),
      "ap_hi":float(ap_hi),"ap_lo":float(ap_lo),"bmi":float(weight)/(float(height)/100)**2,
      "pulse_pressure":pp,"map":float(ap_lo)+pp/3,"gender":int(gender),"cholesterol":int(cholesterol),
      "gluc":int(gluc),"smoke":int(smoke),"alco":int(alco),"active":int(active)}],columns=FEATURE_COLUMNS)
