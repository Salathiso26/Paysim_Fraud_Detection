"""
Step 5 - RQ3: anomaly detection for emerging / unseen fraud patterns.
Writes: results/rq3_anomaly_results.csv, powerbi/rq3_*.csv, figures 14-15, best_anom in meta.json.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from utils import (OUT, RANDOM_STATE, save_fig, evaluate, flag_top, recall_at_k, rf, save_meta,
                   load_modelling_data)

data, FEATURES = load_modelling_data()

print("\n" + "=" * 70 + "\n5. RQ3\n" + "=" * 70)

# 5a. Temporal validation: train on the past, test on the future (simulates deployment)
cut = data["step"].quantile(0.6)
past, future = data[data.step <= cut], data[data.step > cut]
Xp, yp = past[FEATURES], past["isFraud"]
Xf, yf = future[FEATURES], future["isFraud"]
print(f"Past: {len(past):,} rows / {int(yp.sum())} fraud | Future: {len(future):,} rows / {int(yf.sum())} fraud")

# Anomaly detectors are fitted on the past *legitimate* behaviour only (no fraud labels needed)
scal = StandardScaler().fit(Xp[yp == 0])
Zp0 = scal.transform(Xp[yp == 0]); Zf = scal.transform(Xf)
fr = float(yf.mean())


def run_detectors(Ztrain, Ztest):
    sc = {}
    m = IsolationForest(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1).fit(Ztrain)
    sc["Isolation Forest"] = -m.score_samples(Ztest)
    idx = np.random.choice(len(Ztrain), min(40_000, len(Ztrain)), replace=False)
    sc["Local Outlier Factor"] = -LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Ztrain[idx]).score_samples(Ztest)
    p = PCA(n_components=0.90, random_state=RANDOM_STATE).fit(Ztrain)
    sc["PCA Reconstruction"] = ((Ztest - p.inverse_transform(p.transform(Ztest))) ** 2).mean(axis=1)
    ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation="relu", max_iter=60, early_stopping=True,
                      random_state=RANDOM_STATE)
    sub = Ztrain[np.random.choice(len(Ztrain), min(100_000, len(Ztrain)), replace=False)]
    ae.fit(sub, sub)
    sc["Autoencoder (MLP)"] = ((Ztest - ae.predict(Ztest)) ** 2).mean(axis=1)
    return sc


anom_scores = run_detectors(Zp0, Zf)
rq3_rows = []
for n, s in anom_scores.items():
    rq3_rows.append({"model": n, "scenario": "Temporal (past->future)", "pr_auc": average_precision_score(yf, s),
                     "roc_auc": roc_auc_score(yf, s), "precision_at_fraud_rate": evaluate(yf, flag_top(s, fr))["precision"],
                     "recall_at_fraud_rate": evaluate(yf, flag_top(s, fr))["recall"],
                     "recall_top1pct": recall_at_k(yf, s, 0.01), "recall_top5pct": recall_at_k(yf, s, 0.05)})

# Supervised reference trained on past fraud labels
sup_ref = rf(class_weight="balanced_subsample").fit(Xp, yp)
s_sup = sup_ref.predict_proba(Xf)[:, 1]
rq3_rows.append({"model": "Supervised RF (reference)", "scenario": "Temporal (past->future)",
                 "pr_auc": average_precision_score(yf, s_sup), "roc_auc": roc_auc_score(yf, s_sup),
                 "precision_at_fraud_rate": evaluate(yf, flag_top(s_sup, fr))["precision"],
                 "recall_at_fraud_rate": evaluate(yf, flag_top(s_sup, fr))["recall"],
                 "recall_top1pct": recall_at_k(yf, s_sup, 0.01), "recall_top5pct": recall_at_k(yf, s_sup, 0.05)})

# 5b. NOVEL-PATTERN simulation: cluster fraud into behavioural patterns, hide one pattern from training.
#     Supervised model has never seen the hidden pattern; anomaly detectors never use fraud labels at all.
fraud_all = data[data.isFraud == 1]
fz = StandardScaler().fit_transform(np.log1p(fraud_all[FEATURES].clip(lower=0)))
k_pat = 3
fraud_all = fraud_all.assign(pattern=KMeans(n_clusters=k_pat, n_init=10, random_state=RANDOM_STATE).fit_predict(fz))
print("Fraud pattern sizes:\n", fraud_all.pattern.value_counts().sort_index())
print(fraud_all.groupby("pattern")[["amount", "oldbalanceOrg", "errorBalanceDest", "origEmptied"]].mean().round(1))
fraud_all.groupby("pattern")[["amount", "oldbalanceOrg", "newbalanceDest", "errorBalanceDest", "origEmptied"]] \
    .mean().reset_index().to_csv(f"{OUT}/powerbi/rq3_fraud_patterns.csv", index=False)

legit_all = data[data.isFraud == 0]
Xtr_l, Xte_l = train_test_split(legit_all, test_size=0.3, random_state=RANDOM_STATE)
novel_rows = []
for hold in range(k_pat):
    f_seen = fraud_all[fraud_all.pattern != hold]
    f_new = fraud_all[fraud_all.pattern == hold]
    if len(f_new) < 5 or len(f_seen) < 5:
        continue
    f_seen_tr, _ = train_test_split(f_seen, test_size=0.3, random_state=RANDOM_STATE)
    train_df = pd.concat([Xtr_l, f_seen_tr])
    test_df = pd.concat([Xte_l, f_new])
    ytest = test_df["isFraud"]
    # supervised
    ms = rf(class_weight="balanced_subsample").fit(train_df[FEATURES], train_df["isFraud"])
    ss = ms.predict_proba(test_df[FEATURES])[:, 1]
    novel_rows.append({"held_out_pattern": hold, "n_novel_fraud": len(f_new), "model": "Supervised RF",
                       "pr_auc": average_precision_score(ytest, ss),
                       "recall_top1pct": recall_at_k(ytest, ss, 0.01), "recall_top5pct": recall_at_k(ytest, ss, 0.05)})
    # anomaly detectors (trained on legit only)
    sc2 = StandardScaler().fit(Xtr_l[FEATURES])
    for n, s in run_detectors(sc2.transform(Xtr_l[FEATURES]), sc2.transform(test_df[FEATURES])).items():
        novel_rows.append({"held_out_pattern": hold, "n_novel_fraud": len(f_new), "model": n,
                           "pr_auc": average_precision_score(ytest, s),
                           "recall_top1pct": recall_at_k(ytest, s, 0.01), "recall_top5pct": recall_at_k(ytest, s, 0.05)})
novel = pd.DataFrame(novel_rows)
novel_avg = novel.groupby("model")[["pr_auc", "recall_top1pct", "recall_top5pct"]].mean().reset_index()
novel_avg["scenario"] = "Novel pattern (hold-one-pattern-out, mean)"
rq3 = pd.concat([pd.DataFrame(rq3_rows), novel_avg], ignore_index=True)
print(rq3.round(4).to_string(index=False))
rq3.to_csv(f"{OUT}/results/rq3_anomaly_results.csv", index=False)
rq3.to_csv(f"{OUT}/powerbi/rq3_anomaly_results.csv", index=False)
novel.to_csv(f"{OUT}/powerbi/rq3_novel_pattern_detail.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
rq3[rq3.scenario.str.startswith("Temporal")].set_index("model")[["pr_auc", "recall_top1pct", "recall_top5pct"]].plot(kind="bar", ax=ax[0])
ax[0].set_title("RQ3 - Temporal validation (past -> future)"); ax[0].tick_params(axis="x", rotation=35)
rq3[rq3.scenario.str.startswith("Novel")].set_index("model")[["pr_auc", "recall_top1pct", "recall_top5pct"]].plot(kind="bar", ax=ax[1])
ax[1].set_title("RQ3 - Unseen fraud patterns"); ax[1].tick_params(axis="x", rotation=35)
save_fig("14_rq3_anomaly")

# Score distribution of best detector on the future window
best_anom = rq3[(rq3.scenario.str.startswith("Temporal")) & (~rq3.model.str.contains("Supervised"))].sort_values("pr_auc").iloc[-1]["model"]
plt.figure(figsize=(7, 4))
sns.kdeplot(x=np.log1p(anom_scores[best_anom] - anom_scores[best_anom].min()), hue=yf.values, common_norm=False)
plt.title(f"Anomaly score by class - {best_anom}"); plt.xlabel("log anomaly score")
save_fig("15_rq3_score_dist")
save_meta(best_anom=best_anom)
