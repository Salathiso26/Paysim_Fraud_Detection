# Machine Learning for Financial Fraud Detection (PaySim)

An end-to-end Python pipeline that explores the PaySim mobile-money dataset, trains and compares fraud detection models, and exports tables for a Power BI dashboard.

## Research questions
- **RQ1:** Which methods, supervised or unsupervised, detect fraud most effectively?
- **RQ2:** How does class imbalance affect detection, and which strategies help?
- **RQ3:** Can anomaly detection find fraud patterns not seen in the labels?

## Dataset
PaySim is a synthetic mobile-money dataset (Lopez-Rojas, Elmir and Axelsson, 2016) with 6,362,620 transactions. Only 8,213 (0.129%) are fraud, and all of them are `TRANSFER` or `CASH_OUT`. The CSV is not included because of its size. Download it from Kaggle and save it as `Paysim.csv` in the project root.

## Project structure
- `utils.py`: shared settings and helper functions
- `01_eda.py`: exploratory data analysis
- `02_features.py`: feature engineering and modelling sample
- `03_rq1_models.py`: supervised vs unsupervised models
- `04_rq2_imbalance.py`: imbalance strategies and threshold tuning
- `05_rq3_anomaly.py`: anomaly detection on future and unseen patterns
- `06_powerbi_export.py`: scored data and tables for Power BI
- `07_report.py`: auto-generated report draft
- `run_all.py`: runs all steps in order
- `outputs/`: figures, results, powerbi and interim files

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib imbalanced-learn xgboost tabulate
python run_all.py        # all steps
```
`xgboost` and `imbalanced-learn` are optional. Settings such as `RANDOM_STATE = 42` are in `utils.py`.

## Method
- The models use all fraud rows plus 400,000 random legitimate rows, with a stratified 70/30 split.
- Features include balance-error terms, an emptied-account flag, amount-to-balance ratio and prior transaction counts, which use earlier rows only to avoid leakage.
- PR-AUC is the main metric, because accuracy is misleading when fraud is this rare.

## Key results
| Model | Type | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Random Forest | Supervised | 1.000 | 0.997 | 0.999 |
| XGBoost | Supervised | 0.999 | 0.997 | 0.999 |
| Logistic Regression | Supervised | 0.380 | 0.977 | 0.909 |
| Isolation Forest | Unsupervised | 0.171 | 0.171 | 0.104 |
| K-Means Distance | Unsupervised | 0.131 | 0.131 | 0.110 |

- Tree ensembles perform best. Unsupervised methods are much weaker when labels exist.
- On unseen fraud patterns, the Autoencoder is the best anomaly detector (PR-AUC 0.389). The supervised Random Forest still scores higher (0.666), so anomaly detection works best as a complementary early-warning layer.
- RQ2 tables are in `outputs/results/`.

## Power BI
Step 6 writes the dashboard data to `outputs/powerbi/`. Load the files with Get data > Text/CSV. On a computer set to English (South Africa), set the import locale to English (United States) so decimals load correctly. `transactions_scored.csv` holds all fraud plus 100,000 legitimate rows scored on training data, so use it for patterns, not overall rates.

## Limitations
- PaySim is synthetic, so real fraud will be noisier.
- Concept Drift
- The sample inflates the fraud share, so precision would be lower on real traffic.
- Results use one random seed and one split.

## Author
Yonda Salathiso Ntsele | [GitHub](https://github.com/Salathiso26) | [LinkedIn](https://linkedin.com/in/salathiso-ntsele)
