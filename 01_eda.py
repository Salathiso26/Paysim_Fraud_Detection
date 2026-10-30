"""
Step 1 - Exploratory data analysis (uses the FULL dataset).
Writes: outputs/figures/01-06, outputs/powerbi/fraud_by_{type,hour,day}.csv, meta.json (row/fraud counts).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import OUT, load_raw, add_derived_columns, save_fig, save_meta

df = load_raw()

print("\n" + "=" * 70 + "\n1. EDA\n" + "=" * 70)
df.columns = [c.strip() for c in df.columns]
print(df.shape)
df.info()
print(df.describe().T)
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

fraud_rate = df["isFraud"].mean()
print(f"\nFraud cases: {df['isFraud'].sum():,}  ({fraud_rate:.4%});  imbalance ratio 1:{(1-fraud_rate)/fraud_rate:,.0f}")
save_meta(n_rows=len(df), n_fraud=int(df["isFraud"].sum()), fraud_rate=float(fraud_rate))

# 1a. Class balance
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df["isFraud"].value_counts().plot(kind="bar", ax=ax[0], color=["#4C78A8", "#E45756"])
ax[0].set_title("Class counts"); ax[0].set_xticklabels(["Legit", "Fraud"], rotation=0)
df["isFraud"].value_counts().plot(kind="bar", ax=ax[1], logy=True, color=["#4C78A8", "#E45756"])
ax[1].set_title("Class counts (log scale)"); ax[1].set_xticklabels(["Legit", "Fraud"], rotation=0)
save_fig("01_class_balance")

# 1b. Transaction type vs fraud
type_summary = (df.groupby("type")
                  .agg(transactions=("isFraud", "size"), frauds=("isFraud", "sum"),
                       avg_amount=("amount", "mean"))
                  .assign(fraud_rate=lambda x: x.frauds / x.transactions)
                  .reset_index())
print(type_summary)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=df, x="type", hue="isFraud", ax=ax[0]); ax[0].set_yscale("log")
ax[0].set_title("Transactions by type (log)")
sns.barplot(data=type_summary, x="type", y="fraud_rate", ax=ax[1], color="#E45756")
ax[1].set_title("Fraud rate by type")
save_fig("02_type_vs_fraud")

# 1c. Amount distributions
samp = df.sample(min(len(df), 500_000), random_state=1)[["amount", "type", "isFraud"]].copy()
samp["log_amount"] = np.log1p(samp["amount"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(data=samp, x="log_amount", hue="isFraud", bins=60, stat="density", common_norm=False, ax=ax[0])
ax[0].set_xlabel("log(1+amount)"); ax[0].set_title("Amount distribution by class")
sns.boxplot(data=samp[samp.type.isin(["TRANSFER", "CASH_OUT"])], x="type", y="log_amount", hue="isFraud", ax=ax[1])
ax[1].set_title("log(amount): TRANSFER / CASH_OUT")
save_fig("03_amount_distribution")

df = add_derived_columns(df)   # hour, day, errorBalanceOrig/Dest, origEmptied
# 1d. Time patterns
hourly = df.groupby("hour").agg(transactions=("isFraud", "size"), frauds=("isFraud", "sum")).reset_index()
hourly["fraud_rate"] = hourly.frauds / hourly.transactions
daily = df.groupby("day").agg(transactions=("isFraud", "size"), frauds=("isFraud", "sum")).reset_index()
daily["fraud_rate"] = daily.frauds / daily.transactions
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hourly.hour, hourly.fraud_rate, marker="o", color="#E45756"); ax[0].set_title("Fraud rate by hour of day")
ax[1].plot(daily.day, daily.frauds, marker="o", color="#E45756"); ax[1].set_title("Fraud count per day")
save_fig("04_time_patterns")

# 1e. Balance-inconsistency (the strongest signal in PaySim)
tc = df[df.type.isin(["TRANSFER", "CASH_OUT"])]
tc = tc.sample(min(len(tc), 300_000), random_state=1).copy()
tc["signed_log_err_dest"] = np.sign(tc["errorBalanceDest"]) * np.log1p(tc["errorBalanceDest"].abs())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(data=tc, x="isFraud", y="signed_log_err_dest", ax=ax[0])
ax[0].set_title("Destination balance error (signed log)")
df.groupby("origEmptied")["isFraud"].mean().plot(kind="bar", ax=ax[1], color="#E45756")
ax[1].set_title("Fraud rate when origin account emptied")
save_fig("05_balance_errors")

# 1f. Correlation heatmap (numeric)
num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
            "errorBalanceOrig", "errorBalanceDest", "origEmptied", "hour", "isFraud"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix")
save_fig("06_correlation")

# 1g. Other data-quality checks
print("isFlaggedFraud vs isFraud:\n", pd.crosstab(df["isFlaggedFraud"], df["isFraud"]))
print("Fraud only in types:", df.loc[df.isFraud == 1, "type"].unique())
print("Merchant (M*) destinations in fraud:", df.loc[df.isFraud == 1, "nameDest"].str.startswith("M").sum())

# Save EDA tables for Power BI
type_summary.to_csv(f"{OUT}/powerbi/fraud_by_type.csv", index=False)
hourly.to_csv(f"{OUT}/powerbi/fraud_by_hour.csv", index=False)
daily.to_csv(f"{OUT}/powerbi/fraud_by_day.csv", index=False)
