# from fastapi import FastAPI
# from recsys.data_pipeline.ingest import load_movielens_100k
# from recsys.retrieval.item2item import build_item_cooccur, recommend_item2item
# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split

# app = FastAPI(title="Video Recommendation API")

# @app.get("/")
# def root():
#     return {
#         "message": "VideoRec API is running 🚀",
#         "try": "/recommend/{user_id}?topk=10",
#         "docs": "/docs"
#     }


# # ----------------------
# # Load Data & Train Model
# # ----------------------
# df = load_movielens_100k()
# df["user_id_enc"] = df["user_id"].astype("category").cat.codes
# df["item_id_enc"] = df["item_id"].astype("category").cat.codes

# X = df[["user_id_enc", "item_id_enc"]]
# y = df["label"]

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# train_ds = lgb.Dataset(X_train, label=y_train)
# val_ds = lgb.Dataset(X_val, label=y_val)

# params = dict(objective="binary", metric="auc", learning_rate=0.1, num_leaves=31)
# model = lgb.train(
#     params,
#     train_ds,
#     valid_sets=[val_ds],
#     num_boost_round=200,
#     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
# )

# # Build co-occurrence matrix
# pos_df = df[df["label"] == 1]
# cooccur = build_item_cooccur(pos_df)

# # ----------------------
# # API Endpoint
# # ----------------------
# @app.get("/recommend/{user_id}")
# def recommend(user_id: int, topk: int = 10):
#     # User history
#     user_history = pos_df[pos_df["user_id"] == user_id]["item_id"].tolist()
#     if not user_history:
#         return {"user_id": user_id, "recommendations": []}

#     # Recall candidates
#     candidates = recommend_item2item(user_history, cooccur, topk=100)

#     # Ranking
#     user_enc = df[df["user_id"] == user_id]["user_id_enc"].iloc[0]
#     cand_df = pd.DataFrame({
#         "user_id_enc": [user_enc] * len(candidates),
#         "item_id_enc": [df[df["item_id"] == i]["item_id_enc"].iloc[0] for i in candidates]
#     })
#     scores = model.predict(cand_df)
#     ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:topk]
#     ranked = [{"item_id": int(i), "score": float(s)} for i, s in ranked]

#     return {"user_id": user_id, "recommendations": ranked}


from fastapi import FastAPI
from recsys.data_pipeline.ingest import load_movielens_100k
from recsys.retrieval.item2item import build_item_cooccur, recommend_item2item
import pandas as pd
import lightgbm as lgb

app = FastAPI(title="Video Recommendation API")

@app.get("/")
def root():
    return {
        "message": "VideoRec API is running 🚀",
        "try": "/recommend/{user_id}?topk=10",
        "docs": "/docs"
    }

# ----------------------
# Load Data & Load Model
# ----------------------
df = load_movielens_100k()
df["user_id_enc"] = df["user_id"].astype("category").cat.codes
df["item_id_enc"] = df["item_id"].astype("category").cat.codes

# 直接加载保存好的模型（避免启动时重新训练）
# model = lgb.train(...)   # ❌ 不要在API里训练
# 直接加载保存好的模型
model = lgb.Booster(model_file="/app/model.txt")


# 构建 co-occurrence 矩阵
pos_df = df[df["label"] == 1]
cooccur = build_item_cooccur(pos_df)
item_map = df.set_index("item_id")["item_id_enc"].to_dict()

# ----------------------
# API Endpoint
# ----------------------
@app.get("/recommend/{user_id}")
def recommend(user_id: int, topk: int = 10):
    user_history = pos_df[pos_df["user_id"] == user_id]["item_id"].tolist()
    if not user_history:
        return {"user_id": user_id, "recommendations": []}

    # 召回候选
    candidates = recommend_item2item(user_history, cooccur, topk=100)

    # 构建排序特征（安全取 item_id 映射）
    user_enc = df[df["user_id"] == user_id]["user_id_enc"].iloc[0]
    cand_df = pd.DataFrame({
        "user_id_enc": [user_enc] * len(candidates),
        "item_id_enc": [item_map.get(i, -1) for i in candidates]
    })
    cand_df = cand_df[cand_df["item_id_enc"] != -1]  # 过滤无效候选

    if cand_df.empty:
        return {"user_id": user_id, "recommendations": []}

    # 排序
    scores = model.predict(cand_df)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:topk]
    ranked = [{"item_id": int(i), "score": float(s)} for i, s in ranked]

    return {"user_id": user_id, "recommendations": ranked}
