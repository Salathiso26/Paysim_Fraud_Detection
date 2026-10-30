"""
Step 4 - RQ2: class-imbalance severity, mitigation strategies and threshold tuning.
Writes: results/rq2_*.csv, powerbi/rq2_*.csv, figures 11-13, and best_strat / t_cost / t_f1 / dummy_acc in meta.json.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import (OUT, RANDOM_STATE, FP_COST, save_fig, evaluate, rf, make_business_cost, save_meta,
                   load_modelling_data, split_data)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMB = True
except Exception:
    HAS_IMB = False
    print("[warn] imbalanced-learn not installed -> pip install imbalanced-learn (RQ2 will be limited)")

data, FEATURES = load_modelling_data()
X_train, X_test, y_train, y_test = split_data(data, FEATURES)

print("\n" + "=" * 70 + "\n4. RQ2\n" + "=" * 70)


# 4a. How does imbalance severity hurt? Progressively dilute fraud in TRAINING data (test fixed).
sev_rows = []
fraud_tr = X_train[y_train == 1]; legit_tr = X_train[y_train == 0]
for target_ratio in [0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001]:
    n_f = min(len(fraud_tr), int(len(legit_tr) * target_ratio / (1 - target_ratio)))
    n_f = max(n_f, 10)
    Xs = pd.concat([legit_tr, fraud_tr.sample(n_f, random_state=1)])
    ys = np.r_[np.zeros(len(legit_tr)), np.ones(n_f)]
    m = rf().fit(Xs, ys)
    sc = m.predict_proba(X_test)[:, 1]
    sev_rows.append({"train_fraud_ratio": n_f / len(ys), "n_fraud_train": n_f,
                     **evaluate(y_test, (sc >= 0.5).astype(int), sc)})
sev = pd.DataFrame(sev_rows)
print(sev[["train_fraud_ratio", "n_fraud_train", "precision", "recall", "f1", "pr_auc"]].round(4).to_string(index=False))
sev.to_csv(f"{OUT}/results/rq2_imbalance_severity.csv", index=False)
sev.to_csv(f"{OUT}/powerbi/rq2_imbalance_severity.csv", index=False)
plt.figure(figsize=(7, 4.5))
plt.semilogx(sev.train_fraud_ratio, sev.recall, "o-", label="Recall")
plt.semilogx(sev.train_fraud_ratio, sev.precision, "s-", label="Precision")
plt.semilogx(sev.train_fraud_ratio, sev.pr_auc, "^-", label="PR-AUC")
plt.xlabel("Fraud share in training data (log)"); plt.legend(); plt.title("RQ2 - Effect of imbalance severity (RF)")
save_fig("11_rq2_severity")

# 4b. Accuracy paradox demonstration
dummy_acc = 1 - y_test.mean()
print(f"\nAccuracy paradox: predicting 'legit' for everything gives accuracy {dummy_acc:.4%} but recall 0.")
save_meta(dummy_acc=float(dummy_acc))

# 4c. Strategy comparison
# NOTE: resampling is applied ONLY to the training data (inside an imblearn Pipeline) - never to test data.
strategies = {
    "Baseline (no handling)": rf(),
    "Class weight (balanced)": rf(class_weight="balanced_subsample"),
    "Class weight (custom 1:50)": rf(class_weight={0: 1, 1: 50}),
}
if HAS_IMB:
    strategies.update({
        "Random Undersampling": ImbPipeline([("s", RandomUnderSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)), ("m", rf())]),
        "Random Oversampling": ImbPipeline([("s", RandomOverSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)), ("m", rf())]),
        "SMOTE": ImbPipeline([("s", SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=RANDOM_STATE)), ("m", rf())]),
        "ADASYN": ImbPipeline([("s", ADASYN(sampling_strategy=0.1, random_state=RANDOM_STATE)), ("m", rf())]),
        "SMOTE + Tomek": ImbPipeline([("s", SMOTETomek(sampling_strategy=0.1, random_state=RANDOM_STATE)), ("m", rf())]),
        "SMOTE + class weight": ImbPipeline([("s", SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)),
                                             ("m", rf(class_weight="balanced_subsample"))]),
    })
if HAS_XGB:
    strategies["XGBoost scale_pos_weight"] = XGBClassifier(
        n_estimators=200, max_depth=6, n_jobs=-1, eval_metric="aucpr", random_state=RANDOM_STATE,
        scale_pos_weight=(y_train == 0).sum() / y_train.sum())

# Cost-sensitive evaluation: a missed fraud costs far more than a false alarm (FP_COST set in utils.py).
business_cost = make_business_cost(X_test["amount"].values)


rq2_rows, rq2_scores = [], {}
for name, model in strategies.items():
    print(f"  fitting: {name}")
    model.fit(X_train, y_train)
    sc = model.predict_proba(X_test)[:, 1]
    pred = (sc >= 0.5).astype(int)
    rq2_rows.append({"strategy": name, **evaluate(y_test, pred, sc), "business_cost": business_cost(y_test, pred)})
    rq2_scores[name] = (sc, model)
rq2 = pd.DataFrame(rq2_rows).sort_values("pr_auc", ascending=False)
print(rq2.round(4).to_string(index=False))
rq2.to_csv(f"{OUT}/results/rq2_strategy_comparison.csv", index=False)
rq2.to_csv(f"{OUT}/powerbi/rq2_strategy_comparison.csv", index=False)

rq2.set_index("strategy")[["precision", "recall", "f1"]].plot(kind="bar", figsize=(12, 5))
plt.title("RQ2 - Imbalance strategies (threshold 0.5)"); plt.xticks(rotation=35, ha="right")
save_fig("12_rq2_strategies")

# 4d. Threshold optimisation (cost-sensitive decision rule) on the best-PR-AUC strategy
best_strat = rq2.iloc[0]["strategy"]
sc_best = rq2_scores[best_strat][0]
thr_grid = np.linspace(0.02, 0.98, 49)
thr_rows = []
for t in thr_grid:
    p = (sc_best >= t).astype(int)
    thr_rows.append({"threshold": t, "precision": precision_score(y_test, p, zero_division=0),
                     "recall": recall_score(y_test, p), "f1": f1_score(y_test, p),
                     "business_cost": business_cost(y_test, p)})
thr_df = pd.DataFrame(thr_rows)
t_cost = thr_df.loc[thr_df.business_cost.idxmin(), "threshold"]
t_f1 = thr_df.loc[thr_df.f1.idxmax(), "threshold"]
print(f"\nBest strategy: {best_strat} | cost-optimal threshold={t_cost:.2f} | F1-optimal threshold={t_f1:.2f}")
thr_df.to_csv(f"{OUT}/powerbi/rq2_threshold_tuning.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(thr_df.threshold, thr_df.precision, label="Precision"); ax[0].plot(thr_df.threshold, thr_df.recall, label="Recall")
ax[0].plot(thr_df.threshold, thr_df.f1, label="F1"); ax[0].axvline(t_cost, ls="--", c="grey"); ax[0].legend()
ax[0].set_title(f"Threshold tuning - {best_strat}")
ax[1].plot(thr_df.threshold, thr_df.business_cost); ax[1].axvline(t_cost, ls="--", c="grey")
ax[1].set_title("Expected business cost vs threshold")
save_fig("13_rq2_threshold_cost")
save_meta(best_strat=best_strat, t_cost=float(t_cost), t_f1=float(t_f1))
