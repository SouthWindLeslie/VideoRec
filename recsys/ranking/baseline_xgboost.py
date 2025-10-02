import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from data_pipeline import load_movielens_100k

# 1. Load dataset
df = load_movielens_100k()

# 2. Encode categorical features
df["user_id_enc"] = df["user_id"].astype("category").cat.codes
df["item_id_enc"] = df["item_id"].astype("category").cat.codes
X = df[["user_id_enc", "item_id_enc"]]
y = df["label"]

# 3. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. XGBoost training
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {"objective": "binary:logistic", "eval_metric": "auc"}
bst = xgb.train(params, dtrain, num_boost_round=200,
                evals=[(dval, "validation")], early_stopping_rounds=20)

# 5. Predictions
val_pred = bst.predict(dval)
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)

# 6. Ranking metrics
val_df = X_val.copy()
val_df["user_id"] = df.loc[X_val.index, "user_id"].values
val_df["label"]   = y_val.values
val_df["score"]   = val_pred

def precision_at_k(g, k=10):
    topk = g.sort_values("score", ascending=False).head(k)
    return topk["label"].mean()

def dcg_at_k(labels):
    labels = np.asarray(labels)
    return ((2**labels - 1) / np.log2(np.arange(2, len(labels)+2))).sum()

def ndcg_at_k(g, k=10):
    g_sorted = g.sort_values("score", ascending=False).head(k)
    dcg = dcg_at_k(g_sorted["label"].values)
    ideal = dcg_at_k(sorted(g["label"].values, reverse=True)[:k])
    return dcg / ideal if ideal > 0 else 0.0

p_at_10 = val_df.groupby("user_id").apply(precision_at_k, k=10).mean()
ndcg_10 = val_df.groupby("user_id").apply(ndcg_at_k, k=10).mean()
print(f"Precision@10: {p_at_10:.4f} | NDCG@10: {ndcg_10:.4f}")
