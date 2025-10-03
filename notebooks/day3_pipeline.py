from recsys.data_pipeline.ingest import load_movielens_100k
from recsys.retrieval.item2item import build_item_cooccur, recommend_item2item
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# 1. load data and preprocess
df = load_movielens_100k()
df["user_id_enc"] = df["user_id"].astype("category").cat.codes
df["item_id_enc"] = df["item_id"].astype("category").cat.codes

X = df[["user_id_enc", "item_id_enc"]]
y = df["label"]

# 2. train ranking model
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
train_ds = lgb.Dataset(X_train, label=y_train)
val_ds = lgb.Dataset(X_val, label=y_val)

params = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=31)
model = lgb.train(
    params,
    train_ds,
    valid_sets=[val_ds],
    num_boost_round=200,
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
)

model.save_model("model.txt")

# 3. build item co-occurrence and recommend candidates
pos_df = df[df["label"] == 1]
cooccur = build_item_cooccur(pos_df)

user_id = 1
user_history = pos_df[pos_df["user_id"] == user_id]["item_id"].tolist()
candidates = recommend_item2item(user_history, cooccur, topk=100)

# 4. prepare candidate features for ranking
user_enc = df[df["user_id"] == user_id]["user_id_enc"].iloc[0]
cand_df = pd.DataFrame({
    "user_id_enc": [user_enc] * len(candidates),
    "item_id_enc": [df[df["item_id"] == i]["item_id_enc"].iloc[0] for i in candidates]
})

# 5. rank candidates
scores = model.predict(cand_df)
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:10]

print(f"User {user_id} history: {user_history[:5]}")
print("Final Top-10 Recommendations (Recall+Ranking):")
print(ranked)
