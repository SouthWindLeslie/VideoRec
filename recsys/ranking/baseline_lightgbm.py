import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from recsys.data_pipeline.ingest import load_movielens_100k


# 1. Load dataset
df = load_movielens_100k()

# 2. Encode user_id and item_id
df["user_id_enc"] = df["user_id"].astype("category").cat.codes
df["item_id_enc"] = df["item_id"].astype("category").cat.codes

# Features and label
X = df[["user_id_enc", "item_id_enc"]]
y = df["label"]

# 3. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Prepare LightGBM datasets
train_ds = lgb.Dataset(X_train, label=y_train)
val_ds = lgb.Dataset(X_val, label=y_val)

# 5. Model parameters
params = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=31)

# 6. Train model
model = lgb.train(
    params,
    train_ds,
    valid_sets=[val_ds],
    num_boost_round=200,
    callbacks=[
        lgb.early_stopping(20),           # stop after 20 rounds no improvement
        lgb.log_evaluation(50)            # print every 50 iterations
    ]
)



# 7. Prediction
val_pred = model.predict(X_val, num_iteration=model.best_iteration)

# 8. AUC score
auc = roc_auc_score(y_val, val_pred)
print("Validation AUC:", auc)

# ---------------------------
# Extra: Ranking Metrics
# ---------------------------

val_df = X_val.copy()
val_df["user_id"] = df.loc[X_val.index, "user_id"].values
val_df["label"]   = y_val.values
val_df["score"]   = val_pred

def precision_at_k(g: pd.DataFrame, k=10):
    """Compute Precision@K for a single user group."""
    topk = g.sort_values("score", ascending=False).head(k)
    return topk["label"].mean()

def dcg_at_k(labels):
    """Compute Discounted Cumulative Gain (DCG)."""
    labels = np.asarray(labels)
    return ((2**labels - 1) / np.log2(np.arange(2, len(labels)+2))).sum()

def ndcg_at_k(g: pd.DataFrame, k=10):
    """Compute Normalized DCG (NDCG@K) for a single user group."""
    g_sorted = g.sort_values("score", ascending=False).head(k)
    dcg = dcg_at_k(g_sorted["label"].values)
    ideal = dcg_at_k(sorted(g["label"].values, reverse=True)[:k])
    return dcg / ideal if ideal > 0 else 0.0

# Compute average metrics across users
p_at_10 = val_df.groupby("user_id").apply(precision_at_k, k=10).mean()
ndcg_10 = val_df.groupby("user_id").apply(ndcg_at_k, k=10).mean()

print(f"Precision@10: {p_at_10:.4f} | NDCG@10: {ndcg_10:.4f}")
