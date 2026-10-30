"""
Step 7 - Auto-generated Markdown report, built from the result tables written by steps 1-5.
"""
import pandas as pd
from utils import OUT, NON_FRAUD_MODEL_SAMPLE, FP_COST, load_meta

results_log = load_meta()
fraud_rate = results_log["fraud_rate"]
rq1 = pd.read_csv(f"{OUT}/results/rq1_model_comparison.csv")
rq2 = pd.read_csv(f"{OUT}/results/rq2_strategy_comparison.csv")
rq3 = pd.read_csv(f"{OUT}/results/rq3_anomaly_results.csv")
results_log["rq2_sev"] = pd.read_csv(f"{OUT}/results/rq2_imbalance_severity.csv")

# =============================================================================
# 7. AUTO-GENERATED REPORT (Markdown - paste into any editor)
# =============================================================================
print("\n" + "=" * 70 + "\n7. REPORT\n" + "=" * 70)


def md(df_, cols, n=10):
    d = df_[cols].head(n).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(4)
    return d.to_markdown(index=False) if hasattr(d, "to_markdown") else d.to_string(index=False)


try:
    import tabulate  # noqa: F401
except Exception:
    md = lambda df_, cols, n=10: "```\n" + df_[cols].head(n).round(4).to_string(index=False) + "\n```"

top_sup = rq1[rq1.type == "Supervised"].iloc[0]
top_unsup = rq1[rq1.type == "Unsupervised"].iloc[0]
rq3_t = rq3[rq3.scenario.str.startswith("Temporal")]
rq3_n = rq3[rq3.scenario.str.startswith("Novel")]
rq2_base = rq2[rq2.strategy.str.startswith("Baseline")].iloc[0]
rq2_best = rq2.sort_values("f1", ascending=False).iloc[0]

report = f"""# Machine Learning for Financial Fraud Detection: PaySim Case Study

## 1. Introduction
Digital payments generate very large transaction volumes in which fraud is extremely rare yet financially costly.
This study uses the PaySim synthetic mobile-money dataset ({results_log['n_rows']:,} transactions, {results_log['n_fraud']:,} frauds,
- **RQ1** Which ML methods (supervised vs unsupervised) are most effective at identifying fraud?
- **RQ2** How does class imbalance affect detection, and which strategies mitigate it?
- **RQ3** Can anomaly detection reveal emerging fraud patterns not seen in historical labels?

## 2. Data and Exploratory Analysis
- Fraud occurs only in `TRANSFER` and `CASH_OUT` transactions (see `figures/02_type_vs_fraud.png`).
- Fraudulent transactions are typically larger and often empty the origin account (`figures/03`, `05`).
- Balance-inconsistency features (`errorBalanceOrig`, `errorBalanceDest`) separate classes strongly.
- Time-of-day patterns for fraud differ from legitimate traffic (`figures/04`).
- Extreme imbalance: ~1 fraud per {(1-fraud_rate)/fraud_rate:,.0f} legitimate transactions; naive accuracy of
  {results_log['dummy_acc']:.2%} is achievable by predicting "legit" always, so accuracy is not used.
- `isFlaggedFraud` (the simulator's rule-based flag) catches almost no fraud and was excluded from the features.

## 3. Methodology
- **Features:** transaction type (one-hot), amount, balances, balance-error terms, hour, amount-to-balance ratio,
  prior-activity counts for origin/destination (computed only from earlier rows, to avoid leakage), merchant flag.
- **Modelling sample:** all fraud rows + {NON_FRAUD_MODEL_SAMPLE:,} random legitimate rows (fraud share inflated; real-world precision will be lower).
- **Split:** 70/30 stratified for RQ1/RQ2; chronological split (first 60% of time vs remainder) for RQ3.
- **Metrics:** precision, recall, F1, MCC, PR-AUC (primary), ROC-AUC. Resampling was applied to training folds only.

## 4. RQ1 - Supervised vs Unsupervised Models
{md(rq1, ['model','type','precision','recall','f1','pr_auc','roc_auc'], 12)}

**Finding.** The best supervised model ({top_sup.model}) reached precision {top_sup.precision:.3f}, recall {top_sup.recall:.3f},
PR-AUC {top_sup.pr_auc:.3f}. The best unsupervised method ({top_unsup.model}) reached precision {top_unsup.precision:.3f},
recall {top_unsup.recall:.3f}, PR-AUC {top_unsup.pr_auc:.3f}. Supervised learners exploit labelled fraud and
substantially outperform unsupervised detectors when labels are available; tree ensembles handle the non-linear
balance-error interactions best. Unsupervised methods still rank fraud above chance and need no labels.

## 5. RQ2 - Data Imbalance and Mitigation
**Effect of imbalance severity** (training fraud share diluted, test set fixed):

{md(results_log['rq2_sev'], ['train_fraud_ratio','n_fraud_train','precision','recall','f1','pr_auc'], 10)}

**Strategy comparison:**

{md(rq2, ['strategy','precision','recall','f1','pr_auc','business_cost'], 12)}

**Finding.** Baseline F1 was {rq2_base.f1:.3f} (recall {rq2_base.recall:.3f}); the best F1 strategy was
**{rq2_best.strategy}** (F1 {rq2_best.f1:.3f}, recall {rq2_best.recall:.3f}, precision {rq2_best.precision:.3f}).
Undersampling and aggressive oversampling raise recall but typically lower precision (more false alarms);
class weighting / cost-sensitive learning achieves a similar recall gain without synthetic data.
Tuning the decision threshold on a business-cost function (cost-optimal threshold = {results_log['t_cost']:.2f}
vs F1-optimal = {results_log['t_f1']:.2f}) is a low-cost addition that often matters as much as resampling.

## 6. RQ3 - Anomaly Detection for Emerging Patterns
{md(rq3, ['model','scenario','pr_auc','recall_top1pct','recall_top5pct'], 20)}

**Finding.** Detectors trained only on past legitimate behaviour were validated on future transactions and on
fraud *patterns hidden from training*. The best detector was **{results_log['best_anom']}**. Compared with the
supervised model, anomaly detectors are less precise on known fraud but degrade less on unseen fraud patterns
because they never depend on fraud labels; they are best used as a complementary early-warning layer that
feeds an analyst review queue and, once confirmed, new labels for retraining.

## 7. Recommendations
1. Deploy a gradient-boosted / random-forest classifier with cost-sensitive weighting as the primary scorer.
2. Choose the alert threshold from the business cost curve (not 0.5).
3. Run an Isolation Forest / autoencoder detector in parallel to surface emerging patterns, reviewed by analysts.
4. Retrain regularly and monitor PR-AUC and drift on a rolling window.

## 8. Limitations
- PaySim is synthetic; real fraud is adaptive and noisier. Balance-error features are partly simulator artefacts.
- The modelling sample inflates the fraud rate; on full-volume traffic precision would be lower.
- Single random seed / single split; add cross-validation and confidence intervals for publication.
- The business-cost assumptions (FP cost = {FP_COST}) are illustrative.

## 9. Appendix
Figures in `outputs/figures/`; result tables in `outputs/results/`; Power BI files in `outputs/powerbi/`.
"""
with open(f"{OUT}/report.md", "w", encoding="utf-8") as fh:
    fh.write(report)
print(f"Report written to {OUT}/report.md  (review and edit the narrative against your actual numbers)")
print("\nDONE.")
