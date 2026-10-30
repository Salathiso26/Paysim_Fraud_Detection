"""
Step 6 - Power BI exports (row-level scored dataset, KPI table and build guide).
Needs the outputs of steps 2-5 (data, best model, threshold, etc.).
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from utils import OUT, INTERIM, RANDOM_STATE, load_meta, load_modelling_data

data, FEATURES = load_modelling_data()
best_model = joblib.load(f"{INTERIM}/best_model.joblib")
results_log = load_meta()            # key numbers written by steps 1, 3, 4 and 5
fraud_rate = results_log["fraud_rate"]
best_name = results_log["best_name"]
best_strat = results_log["best_strat"]
t_cost = results_log["t_cost"]

# Same 60/40 temporal split as RQ3 (anomaly detector is fitted on past legitimate rows)
cut = data["step"].quantile(0.6)
past = data[data.step <= cut]
Xp, yp = past[FEATURES], past["isFraud"]

# 6. POWER BI EXPORTS
# =============================================================================
print("\n" + "=" * 70 + "\n6. POWER BI\n" + "=" * 70)
# Row-level scored dataset (kept small: all fraud + 100k legit) with model score, anomaly score and flags
pbi = data.copy()
pbi["fraud_probability"] = best_model.predict_proba(pbi[FEATURES])[:, 1]
sc_all = StandardScaler().fit(Xp[yp == 0])
iso_all = IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1).fit(sc_all.transform(Xp[yp == 0]))
pbi["anomaly_score"] = -iso_all.score_samples(sc_all.transform(pbi[FEATURES]))
pbi["predicted_fraud"] = (pbi["fraud_probability"] >= t_cost).astype(int)
pbi["risk_band"] = pd.cut(pbi["fraud_probability"], [-0.01, 0.2, 0.5, 0.8, 1.01], labels=["Low", "Medium", "High", "Critical"])
pbi["outcome"] = np.select([(pbi.isFraud == 1) & (pbi.predicted_fraud == 1), (pbi.isFraud == 0) & (pbi.predicted_fraud == 1),
                            (pbi.isFraud == 1) & (pbi.predicted_fraud == 0)], ["TP", "FP", "FN"], "TN")
pbi_keep = pd.concat([pbi[pbi.isFraud == 1], pbi[pbi.isFraud == 0].sample(min(100_000, (pbi.isFraud == 0).sum()), random_state=1)])
pbi_cols = ["step", "day", "hour", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
            "errorBalanceOrig", "errorBalanceDest", "origEmptied", "isFraud", "fraud_probability", "anomaly_score",
            "predicted_fraud", "risk_band", "outcome"]
pbi_keep[pbi_cols].to_csv(f"{OUT}/powerbi/transactions_scored.csv", index=False)
pd.DataFrame({"metric": ["total_transactions", "total_frauds", "fraud_rate", "best_model", "best_strategy", "cost_optimal_threshold"], "value": [results_log["n_rows"], results_log["n_fraud"], f"{fraud_rate:.5f}", best_name, best_strat, f"{t_cost:.2f}"]}).to_csv(f"{OUT}/powerbi/kpi_summary.csv", index=False)

POWERBI_GUIDE = f"""POWER BI DASHBOARD BUILD GUIDE
================================
1. Home > Get Data > Text/CSV. Load every file in outputs/powerbi/:
   transactions_scored, fraud_by_type, fraud_by_hour, fraud_by_day, rq1_model_comparison,
   rq2_strategy_comparison, rq2_imbalance_severity, rq2_threshold_tuning, rq3_anomaly_results,
   rq3_novel_pattern_detail, rq3_fraud_patterns, feature_importance, kpi_summary.
2. Model view: transactions_scored is the fact table (no relationships needed to the summary tables).
   In Power Query set risk_band to Text and sort it Low < Medium < High < Critical (add a sort-order column).

DAX MEASURES (create on transactions_scored)
--------------------------------------------
Total Transactions = COUNTROWS(transactions_scored)
Fraud Count        = CALCULATE(COUNTROWS(transactions_scored), transactions_scored[isFraud] = 1)
Fraud Rate         = DIVIDE([Fraud Count], [Total Transactions])
Fraud Amount       = CALCULATE(SUM(transactions_scored[amount]), transactions_scored[isFraud] = 1)
TP = CALCULATE(COUNTROWS(transactions_scored), transactions_scored[outcome] = "TP")
FP = CALCULATE(COUNTROWS(transactions_scored), transactions_scored[outcome] = "FP")
FN = CALCULATE(COUNTROWS(transactions_scored), transactions_scored[outcome] = "FN")
Precision = DIVIDE([TP], [TP] + [FP])
Recall    = DIVIDE([TP], [TP] + [FN])
F1        = DIVIDE(2 * [Precision] * [Recall], [Precision] + [Recall])
Missed Fraud Amount = CALCULATE(SUM(transactions_scored[amount]), transactions_scored[outcome] = "FN")
Prevented Fraud Amount = CALCULATE(SUM(transactions_scored[amount]), transactions_scored[outcome] = "TP")

NOTE: transactions_scored over-represents fraud (all fraud + 100k legit rows), so use it for
patterns/precision-recall; quote the real-world fraud rate ({fraud_rate:.4%}) from kpi_summary / fraud_by_type.

PAGES
-----
Page 1  Executive Overview : KPI cards (Total Transactions, Fraud Count, Fraud Rate, Fraud Amount),
        donut of fraud by type, line chart of fraud_by_day, slicers (type, day, risk_band).
Page 2  EDA / Patterns     : column chart fraud_rate by hour, scatter amount vs oldbalanceOrg (legend isFraud),
        histogram-style column of amount bins (Data > New group > Bin), heatmap matrix type x risk_band.
Page 3  RQ1 Model Compare  : clustered bar precision/recall/F1 by model (rq1_model_comparison), table with PR-AUC, 
        legend by 'type' (Supervised vs Unsupervised), feature_importance bar chart.
Page 4  RQ2 Imbalance      : line chart train_fraud_ratio vs recall/precision (rq2_imbalance_severity),
        clustered bars by strategy (rq2_strategy_comparison), line chart threshold vs business_cost (rq2_threshold_tuning).
Page 5  RQ3 Anomaly        : bars of pr_auc / recall_top1pct by model and scenario (rq3_anomaly_results),
        table of rq3_fraud_patterns, anomaly_score distribution by isFraud.
Page 6  Operations         : confusion-matrix matrix visual (rows outcome), table of top 50 transactions by
        fraud_probability, drill-through on type.
Format: consistent palette (blue = legit, red = fraud), titles stating the insight, tooltips on, sync slicers across pages.
"""
with open(f"{OUT}/powerbi/POWER_BI_GUIDE.txt", "w") as fh:
    fh.write(POWERBI_GUIDE)
print("Power BI files written to", f"{OUT}/powerbi/")
