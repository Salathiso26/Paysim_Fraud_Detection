"""
Step 3 - RQ1: supervised vs unsupervised models.
Writes: results/rq1_model_comparison.csv, figures 07-10, powerbi/feature_importance.csv,
        interim/best_model.joblib and best_name in meta.json.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from utils import (OUT, INTERIM, RANDOM_STATE, save_fig, evaluate, flag_top, save_meta,
                   load_modelling_data, split_data)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

data, FEATURES = load_modelling_data()
X_train, X_test, y_train, y_test = split_data(data, FEATURES)
print("Train:", X_train.shape, "fraud:", int(y_train.sum()), "| Test:", X_test.shape, "fraud:", int(y_test.sum()))

print("\n" + "=" * 70 + "\n3. RQ1\n" + "=" * 70)
rq1_rows, rq1_scores, rq1_models = [], {}, {}

supervised = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=2,
                                            n_jobs=-1, random_state=RANDOM_STATE),
    "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
}
if HAS_XGB:
    supervised["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1,
                                          scale_pos_weight=(y_train == 0).sum() / y_train.sum(),
                                          eval_metric="aucpr", random_state=RANDOM_STATE)

for name, model in supervised.items():
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    rq1_rows.append({"model": name, "type": "Supervised", **evaluate(y_test, pred, score)})
    rq1_scores[name] = score
    rq1_models[name] = model
    print(f"{name:22s} P={rq1_rows[-1]['precision']:.3f}  R={rq1_rows[-1]['recall']:.3f}  PR-AUC={rq1_rows[-1]['pr_auc']:.3f}")

# --- Unsupervised: trained WITHOUT labels. contamination = known fraud rate in train.
scaler_u = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler_u.transform(X_train), scaler_u.transform(X_test)
contam = float(y_train.mean())


# Isolation Forest
iso = IsolationForest(n_estimators=300, contamination=contam, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr_s)
s_iso = -iso.score_samples(Xte_s)
rq1_rows.append({"model": "Isolation Forest", "type": "Unsupervised", **evaluate(y_test, flag_top(s_iso, contam), s_iso)})
rq1_scores["Isolation Forest"] = s_iso

# Local Outlier Factor (novelty mode, fit on a subsample for speed)
lof_idx = np.random.choice(len(Xtr_s), min(50_000, len(Xtr_s)), replace=False)
lof = LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=contam).fit(Xtr_s[lof_idx])
s_lof = -lof.score_samples(Xte_s)
rq1_rows.append({"model": "Local Outlier Factor", "type": "Unsupervised", **evaluate(y_test, flag_top(s_lof, contam), s_lof)})
rq1_scores["Local Outlier Factor"] = s_lof

# PCA reconstruction error
pca = PCA(n_components=0.90, random_state=RANDOM_STATE).fit(Xtr_s)
recon = lambda Z: ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).mean(axis=1)
s_pca = recon(Xte_s)
rq1_rows.append({"model": "PCA Reconstruction", "type": "Unsupervised", **evaluate(y_test, flag_top(s_pca, contam), s_pca)})
rq1_scores["PCA Reconstruction"] = s_pca

# K-Means distance to nearest centroid
km = KMeans(n_clusters=8, n_init=5, random_state=RANDOM_STATE).fit(Xtr_s)
s_km = km.transform(Xte_s).min(axis=1)
rq1_rows.append({"model": "K-Means Distance", "type": "Unsupervised", **evaluate(y_test, flag_top(s_km, contam), s_km)})
rq1_scores["K-Means Distance"] = s_km

rq1 = pd.DataFrame(rq1_rows).sort_values("pr_auc", ascending=False)
print(rq1.round(4).to_string(index=False))
rq1.to_csv(f"{OUT}/results/rq1_model_comparison.csv", index=False)
rq1.to_csv(f"{OUT}/powerbi/rq1_model_comparison.csv", index=False)

# Figures: PR curves + metric bars
plt.figure(figsize=(8, 6))
for n, s in rq1_scores.items():
    p, r, _ = precision_recall_curve(y_test, s)
    plt.plot(r, p, label=f"{n} (AP={average_precision_score(y_test, s):.2f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("RQ1 - Precision-Recall curves"); plt.legend(fontsize=8)
save_fig("07_rq1_pr_curves")

rq1.set_index("model")[["precision", "recall", "f1"]].plot(kind="bar", figsize=(11, 5))
plt.title("RQ1 - Precision / Recall / F1"); plt.xticks(rotation=35, ha="right")
save_fig("08_rq1_metric_bars")

best_name = rq1[rq1.type == "Supervised"].iloc[0]["model"]
best_model = rq1_models[best_name]
cm = confusion_matrix(y_test, (rq1_scores[best_name] >= 0.5).astype(int))
plt.figure(figsize=(4.5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Confusion matrix - {best_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
save_fig("09_rq1_confusion_best")

# Feature importance
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(15)
elif best_name in rq1_models:
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(best_model, X_test.sample(min(20000, len(X_test)), random_state=1),
                                y_test.loc[X_test.sample(min(20000, len(X_test)), random_state=1).index],
                                scoring="average_precision", n_repeats=3, random_state=1, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False).head(15)
imp.sort_values().plot(kind="barh", figsize=(7, 6), title=f"Top features - {best_name}")
save_fig("10_rq1_feature_importance")
imp.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
    f"{OUT}/powerbi/feature_importance.csv", index=False)

joblib.dump(best_model, f"{INTERIM}/best_model.joblib")
save_meta(best_name=best_name)
