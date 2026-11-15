"""Deterministic MLP training; validation selects the high-recall threshold."""
import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_curve, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from model import CardioMLP
from preprocessing import clean_dataset, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, BINARY_COLUMNS, FEATURE_COLUMNS
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data"/"cardio_train.csv"; OUT=ROOT/"artifacts"; SEED=42
def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
def predict(model,x):
    model.eval()
    with torch.no_grad(): return torch.sigmoid(model(torch.tensor(x,dtype=torch.float32))).numpy()
def evaluate(y,p,t):
    pred=(p>=t).astype(int); cm=confusion_matrix(y,pred,labels=[0,1])
    return {"recall":float(recall_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),
      "f1":float(f1_score(y,pred)),"roc_auc":float(roc_auc_score(y,p)),
      "pr_auc":float(average_precision_score(y,p)),"accuracy":float(accuracy_score(y,pred)),
      "specificity":float(cm[0,0]/max(cm[0].sum(),1)),"threshold":float(t),"confusion_matrix":cm.tolist()}
def main():
    seed_all(); OUT.mkdir(exist_ok=True)
    df=clean_dataset(DATA); X=df[FEATURE_COLUMNS]; y=df.cardio.astype(int)
    xt,xe,yt,ye=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    xa,xv,ya,yv=train_test_split(xt,yt,test_size=.25,random_state=SEED,stratify=yt)
    scaler=StandardScaler(); a=scaler.fit_transform(xa[NUMERIC_COLUMNS]); v=scaler.transform(xv[NUMERIC_COLUMNS]); e=scaler.transform(xe[NUMERIC_COLUMNS])
    try: enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    except TypeError: enc=OneHotEncoder(handle_unknown="ignore",sparse=False)
    ac=enc.fit_transform(xa[CATEGORICAL_COLUMNS]); vc=enc.transform(xv[CATEGORICAL_COLUMNS]); ec=enc.transform(xe[CATEGORICAL_COLUMNS])
    def combine(n,c,f): return np.concatenate([n,c,f[BINARY_COLUMNS].to_numpy(float)],axis=1).astype("float32")
    a=combine(a,ac,xa); v=combine(v,vc,xv); e=combine(e,ec,xe)
    model=CardioMLP(a.shape[1]); pw=torch.tensor([(ya.eq(0).sum()/max(ya.eq(1).sum(),1))*1.3])
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=pw); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=5,factor=.5)
    ds=TensorDataset(torch.tensor(a),torch.tensor(ya.to_numpy(dtype="float32")))
    vd=TensorDataset(torch.tensor(v),torch.tensor(yv.to_numpy(dtype="float32")))
    loader=DataLoader(ds,batch_size=512,shuffle=True); vloader=DataLoader(vd,batch_size=1024)
    best=float("inf"); best_state=None; stale=0; history=[]
    for epoch in range(40):
        model.train(); total=0.
        for xb,yb in loader:
            opt.zero_grad(); l=loss_fn(model(xb),yb); l.backward(); opt.step(); total+=l.item()*len(xb)
        model.eval(); vl=0.
        with torch.no_grad():
            for xb,yb in vloader: vl+=loss_fn(model(xb),yb).item()*len(xb)
        vl/=len(vd); sched.step(vl); history.append({"epoch":epoch+1,"train_loss":total/len(ds),"val_loss":vl})
        print(f"epoch={epoch+1} train_loss={total/len(ds):.4f} val_loss={vl:.4f}")
        if vl<best-1e-4:
            best=vl; best_state={k:z.detach().cpu().clone() for k,z in model.state_dict().items()}; stale=0
        else:
            stale+=1
            if stale>=8: break
    if best_state is None: raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state); vp=predict(model,v); ep=predict(model,e)
    prec,rec,thr=precision_recall_curve(yv,vp)
    eligible=[(r,t) for p,r,t in zip(prec[:-1],rec[:-1],thr) if p>=.55]
    if not eligible: raise RuntimeError("No validation threshold meets precision >= 0.55")
    ht=float(max(eligible,key=lambda z:z[0])[1])
    f1s=2*prec[:-1]*rec[:-1]/np.maximum(prec[:-1]+rec[:-1],1e-12); bt=float(thr[int(np.argmax(f1s))])
    high_v=evaluate(yv,vp,ht); high_t=evaluate(ye,ep,ht)
    bal_v=evaluate(yv,vp,bt); bal_t=evaluate(ye,ep,bt)
    config={"numeric_columns":NUMERIC_COLUMNS,"categorical_columns":CATEGORICAL_COLUMNS,"binary_columns":BINARY_COLUMNS,
      "feature_columns":FEATURE_COLUMNS,"scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist(),
      "categories":[[int(z) for z in cats] for cats in enc.categories_],"input_dim":int(a.shape[1])}
    torch.save(model.state_dict(),OUT/"mlp_state_dict.pt")
    (OUT/"preprocessing.json").write_text(json.dumps(config,indent=2))
    report={"model":"PyTorch MLP (128-64-32)","default_operating_point":"high_recall",
      "threshold":ht,"threshold_selection":"Maximum validation recall subject to validation precision >= 0.55",
      "validation":high_v,"test":high_t,"operating_points":{
        "high_recall":{"threshold":ht,"selection":"Maximum validation recall subject to validation precision >= 0.55","validation":high_v,"test":high_t},
        "balanced":{"threshold":bt,"selection":"Maximum validation F1","validation":bal_v,"test":bal_t}},
      "rows_after_cleaning":len(df),"split_sizes":{"train":len(xa),"validation":len(xv),"test":len(xe)},
      "epochs_run":len(history),"best_validation_loss":best,
      "warning":"Research/educational dataset metrics only; not clinical validation or individual medical risk."}
    (OUT/"metrics.json").write_text(json.dumps(report,indent=2)); pd.DataFrame(history).to_csv(OUT/"training_history.csv",index=False)
    print("High-recall held-out test:",json.dumps(high_t,indent=2))
if __name__=="__main__": main()
