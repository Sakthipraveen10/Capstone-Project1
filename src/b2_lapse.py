"""Part B tasks 4-8: lapse label/features, leak-free preprocessing, tuning, balance experiment, interpretation, persistence."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
o = o.merge(p[["product_id", "category"]], on="product_id")

# ---------------------------------------------------------------- label (September window) and features (before cutoff only)
before = o[o.order_ts < CUTOFF]                       # the ONLY orders features may use
done_b = before[before.status == "completed"]
sept = o[(o.order_ts >= CUTOFF) & (o.order_ts < END) & (o.status == "completed")]

f = done_b.groupby("customer_id").agg(last=("order_ts", "max"), frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                      avg_delivery=("delivery_minutes", "mean"))
f["recency"] = (CUTOFF - f["last"]).dt.days
f["cancel_rate"] = before.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean()).reindex(f.index)
f["fav_category"] = done_b.groupby("customer_id")["category"].agg(lambda s: s.mode().iloc[0])
f = f.drop(columns="last").reset_index().merge(c, on="customer_id")
f["tenure_days"] = (CUTOFF - f["signup_date"]).dt.days
f["lapsed"] = (~f["customer_id"].isin(sept["customer_id"])).astype(int)
NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["fav_category", "city", "gender"]
X, y = f[NUM + CAT], f["lapsed"]
print(f"cohort {len(f)} (ref 1156) | lapsed {int(y.sum())} = {y.mean()*100:.2f}% (ref 443, 38.32%)")
assert not X.isna().any().any(), "missing values in features"
print("no feature column contains a missing value: True")
assert done_b.order_ts.max() < CUTOFF

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train", len(Xtr), "test", len(Xte), "(ref 924 / 232)")
base = np.zeros(len(yte), dtype=int)
print(f"baseline always 'not lapsed': accuracy {accuracy_score(yte, base)*100:.1f}% lapsed-F1 {f1_score(yte, base, zero_division=0):.2f} (ref 61.6% / 0)")

def prep():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

# ---------------------------------------------------------------- tuning (preprocessing lives inside the pipeline => fitted on training folds only)
lr_gs = GridSearchCV(Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000))]),
                     {"m__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_gs = GridSearchCV(Pipeline([("prep", prep()), ("m", RandomForestClassifier(n_estimators=100, random_state=42))]),
                     {"m__max_depth": [3, 5, 8], "m__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("\nLogReg best:", lr_gs.best_params_, "CV F1", round(lr_gs.best_score_, 3))
print("RandForest best:", rf_gs.best_params_, "CV F1", round(rf_gs.best_score_, 3))

# learned scaler mean (training rows only) vs full cohort mean
fi = NUM.index("frequency")
print("scaler mean of frequency (train only):", round(lr_gs.best_estimator_.named_steps["prep"].named_transformers_["num"].mean_[fi], 3),
      "| full-cohort mean:", round(f["frequency"].mean(), 3))

def metrics(name, model, Xe=Xte):
    pred = model.predict(Xe)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    pr, rc, f1 = (precision_score(yte, pred, zero_division=0), recall_score(yte, pred, zero_division=0), f1_score(yte, pred, zero_division=0))
    print(f"{name:28s} TN {tn} FP {fp} FN {fn} TP {tp} | precision {pr:.3f} recall {rc:.3f} F1 {f1:.3f}")
    return pr, rc, f1
print("\nTest set (lapsed = positive class):")
m_lr = metrics("LogReg (tuned)", lr_gs.best_estimator_)
m_rf = metrics("RandomForest (tuned)", rf_gs.best_estimator_)
# selection uses cross-validated F1 (training data only); the test set is only used to report the chosen model
best_name, best = ("RandomForest", rf_gs.best_estimator_) if rf_gs.best_score_ >= lr_gs.best_score_ else ("LogReg", lr_gs.best_estimator_)
print("selected:", best_name)

# ---------------------------------------------------------------- over-sampling of the TRAINING set only
rng = np.random.RandomState(42)
lap_idx = np.where(ytr.values == 1)[0]
n_add = int((ytr == 0).sum() - (ytr == 1).sum())
extra = rng.choice(lap_idx, size=n_add, replace=True)
Xb, yb = pd.concat([Xtr, Xtr.iloc[extra]]), pd.concat([ytr, ytr.iloc[extra]])
print("\nbalanced training counts:", yb.value_counts().to_dict(), "(ref 570 each)")
lr_bal = Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000, C=lr_gs.best_params_["m__C"]))]).fit(Xb, yb)
m_bal = metrics("LogReg + oversampled train", lr_bal)
print("change vs un-balanced LogReg: precision %+.3f recall %+.3f F1 %+.3f" % tuple(np.array(m_bal) - np.array(m_lr)))

# ---------------------------------------------------------------- interpretation
rf_names = rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_gs.best_estimator_.named_steps["m"].feature_importances_, index=rf_names).sort_values(ascending=False)
print("\nTop-5 random-forest importances:\n", imp.head(5).round(3).to_string())
lr_names = lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr_gs.best_estimator_.named_steps["m"].coef_[0], index=lr_names)
print("\nLogReg coefficients (standardised numeric features):\n", coef[["num__recency", "num__frequency", "num__monetary", "num__cancel_rate"]].round(3).to_string())

# ---------------------------------------------------------------- persist and reload
joblib.dump(best, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
same = bool((re.predict(Xte) == best.predict(Xte)).all())
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

# ---------------------------------------------------------------- retention list (illustrative, see README)
seg = pd.read_csv("outputs/customer_segments.csv")
prem = set(seg.loc[seg.is_premium, "customer_id"])
te = f.loc[Xte.index, ["customer_id"]].copy()
te["lapse_probability"] = best.predict_proba(Xte)[:, 1]
te["actually_lapsed"] = yte.values
te = te[te.customer_id.isin(prem)].sort_values("lapse_probability", ascending=False).head(10)
te["lapse_probability"] = te["lapse_probability"].round(4)
te.to_csv("outputs/retention_list.csv", index=False)
print("premium customers in test set:", len(prem & set(f.loc[Xte.index, 'customer_id'])), "| retention list rows:", len(te),
      "| truly lapsed:", int(te.actually_lapsed.sum()))
