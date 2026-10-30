"""
Shared configuration and helper functions for the PaySim fraud-detection pipeline.
Imported by every numbered script, so settings only need editing here.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef, confusion_matrix)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# CONFIG

OUT = "outputs"
INTERIM = f"{OUT}/interim"      # hand-over files between scripts (data, model, key numbers)
DATA_CSV = "Paysim.csv"
for d in ["figures", "powerbi", "results", "interim"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

# Full PaySim = ~6.3M rows. Modelling on all rows is slow, so we keep ALL fraud rows
# and a random share of legitimate rows for the modelling stages (EDA uses full data).
# Set to None to use everything.
NON_FRAUD_MODEL_SAMPLE = 400_000

# Business assumption (edit freely): FN = transaction amount lost; FP = fixed review/customer-friction cost.
FP_COST = 5.0


def load_raw():
    # ---- required loading snippet ----
    df = pd.read_csv(
        DATA_CSV,
        dtype={"step": "int16", "type": "category", "isFraud": "int8", "isFlaggedFraud": "int8"},
    )
    # ----------------------------------
    df.columns = [c.strip() for c in df.columns]
    return df


def add_derived_columns(df):
    """Time and balance-inconsistency columns used by both the EDA and the feature stage."""
    df["hour"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["origEmptied"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(int)
    return df


def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/{name}.png", dpi=150)
    plt.close()


def evaluate(y_true, y_pred, y_score=None):
    """Precision/recall focused metrics for rare-event problems."""
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None:
        out["pr_auc"] = average_precision_score(y_true, y_score)
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"TP": tp, "FP": fp, "FN": fn, "TN": tn})
    return out


def flag_top(score, rate):
    """Flag the top `rate` share of anomaly scores as fraud."""
    thr = np.quantile(score, 1 - rate)
    return (score >= thr).astype(int)


def recall_at_k(y_true, score, k_frac):
    k = max(1, int(len(score) * k_frac))
    top = np.argsort(-score)[:k]
    return y_true.values[top].sum() / y_true.sum()


def rf(**kw):
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE, **kw)


def make_business_cost(amounts):
    """Returns business_cost(y_true, y_pred) for a test set whose transaction amounts are `amounts`."""
    def business_cost(y_true, y_pred):
        fn = (y_true.values == 1) & (y_pred == 0)
        fp = (y_true.values == 0) & (y_pred == 1)
        return amounts[fn].sum() + FP_COST * fp.sum()
    return business_cost



# HAND-OVER BETWEEN SCRIPTS

def save_meta(**kw):
    """Merge key numbers into outputs/interim/meta.json (replaces the in-memory results_log)."""
    path = f"{INTERIM}/meta.json"
    meta = json.load(open(path)) if os.path.exists(path) else {}
    meta.update(kw)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, default=float)


def load_meta():
    with open(f"{INTERIM}/meta.json") as fh:
        return json.load(fh)


def split_data(data, features):
    """The stratified 70/30 split used for RQ1/RQ2 (deterministic, so every script gets the same split)."""
    X, y = data[features], data["isFraud"]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)


def load_modelling_data():
    """Reads the modelling dataset written by 02_features.py."""
    data = pd.read_pickle(f"{INTERIM}/data.pkl")
    with open(f"{INTERIM}/features.json") as fh:
        features = json.load(fh)
    return data, features
