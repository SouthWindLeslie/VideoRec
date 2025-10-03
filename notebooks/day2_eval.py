from recsys.data_pipeline.ingest import load_movielens_100k
from recsys.retrieval.item2item import build_item_cooccur, recommend_item2item
from eval.metrics import recall_at_k, hit_rate
import numpy as np

# 1. 加载数据
df = load_movielens_100k()
pos_df = df[df["label"] == 1]   # 只取正样本

# 2. 构建物品共现矩阵
cooccur = build_item_cooccur(pos_df)

# 3. 全局评估
all_recalls, all_hits = [], []

for user_id, items in pos_df.groupby("user_id")["item_id"]:
    items = list(items)
    if len(items) < 2:   # 跳过交互太少的用户
        continue

    train_items = items[:-1]   # 留最后一个作为测试集
    test_items = items[-1:]

    recos = recommend_item2item(train_items, cooccur, topk=10)

    all_recalls.append(recall_at_k(test_items, recos, k=10))
    all_hits.append(hit_rate(test_items, recos, k=10))

print(f"Overall Recall@10: {np.mean(all_recalls):.4f}")
print(f"Overall HitRate@10: {np.mean(all_hits):.4f}")
