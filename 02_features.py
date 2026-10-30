"""
Step 2 - Feature engineering + modelling dataset.
Reads the raw CSV, builds features, samples the legitimate rows and writes
outputs/interim/data.pkl and features.json for the modelling scripts.
"""
import gc
import json
import numpy as np
import pandas as pd
from utils import (INTERIM, RANDOM_STATE, NON_FRAUD_MODEL_SAMPLE, load_raw,
                   add_derived_columns, split_data)

df = add_derived_columns(load_raw())
fraud_rate = df["isFraud"].mean()

print("\n" + "=" * 70 + "\n2. FEATURES\n" + "=" * 70)

# Customer-level behavioural features (leakage-safe: only the sender's *prior* activity)
df = df.sort_values("step").reset_index(drop=True)
df["orig_txn_count_prior"] = df.groupby("nameOrig").cumcount()
df["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["dest_is_merchant"] = df["nameDest"].str.startswith("M").astype(int)
df["dest_txn_count_prior"] = df.groupby("nameDest").cumcount()

type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=int)
FEATURES = ["step", "hour", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
            "errorBalanceOrig", "errorBalanceDest", "origEmptied", "amount_to_balance",
            "dest_is_merchant", "orig_txn_count_prior", "dest_txn_count_prior"]
data_full = pd.concat([df[FEATURES + ["isFraud", "isFlaggedFraud", "type", "day"]],
                       type_dummies], axis=1)
FEATURES = FEATURES + list(type_dummies.columns)

del df, type_dummies
gc.collect()

# Modelling sample: all fraud + sampled legit
if NON_FRAUD_MODEL_SAMPLE and (data_full.isFraud == 0).sum() > NON_FRAUD_MODEL_SAMPLE:
    legit = data_full[data_full.isFraud == 0].sample(NON_FRAUD_MODEL_SAMPLE, random_state=RANDOM_STATE)
    data = pd.concat([legit, data_full[data_full.isFraud == 1]]).sort_values("step").reset_index(drop=True)
    print(f"Modelling sample: {len(data):,} rows (fraud rate {data.isFraud.mean():.3%}) "
          f"- fraud rate is inflated vs. the real {fraud_rate:.3%}; precision on real traffic would be lower.")
else:
    data = data_full.copy()


# Hand-over to the modelling scripts
data.to_pickle(f"{INTERIM}/data.pkl")
with open(f"{INTERIM}/features.json", "w") as fh:
    json.dump(FEATURES, fh)

X_train, X_test, y_train, y_test = split_data(data, FEATURES)
print("Train:", X_train.shape, "fraud:", int(y_train.sum()), "| Test:", X_test.shape, "fraud:", int(y_test.sum()))
print(f"Saved modelling dataset to {INTERIM}/data.pkl")
