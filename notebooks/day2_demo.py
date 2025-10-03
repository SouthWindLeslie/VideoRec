from recsys.data_pipeline.ingest import load_movielens_100k
from recsys.retrieval.item2item import build_item_cooccur, recommend_item2item
from eval.metrics import recall_at_k, hit_rate

# 1. 加载数据
df = load_movielens_100k()

# 2. 只取正反馈样本
pos_df = df[df["label"] == 1]

# 3. 构建物品共现矩阵
cooccur = build_item_cooccur(pos_df)

# 4. 给某个用户推荐
user_id = 1
user_items = pos_df[pos_df["user_id"] == user_id]["item_id"].tolist()
recommendations = recommend_item2item(user_items, cooccur, topk=10)

print(f"User {user_id} history: {user_items[:5]}")
print(f"Recommendations: {recommendations}")

# 5. 评估 (假设我们用最后一次交互作为测试集)
test_items = user_items[-1:]  # 留一个item做验证
train_items = user_items[:-1]
recos = recommend_item2item(train_items, cooccur, topk=10)

recall = recall_at_k(test_items, recos, k=10)
hit = hit_rate(test_items, recos, k=10)
print(f"Recall@10: {recall:.4f}, HitRate@10: {hit:.4f}")
