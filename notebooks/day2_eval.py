from recsys.data_pipeline.ingest import load_movielens_100k
from recsys.retrieval.item2item import build_item_cooccur, recommend_item2item
from eval.metrics import recall_at_k, hit_rate
import numpy as np

# 1. load data
df = load_movielens_100k()
pos_df = df[df["label"] == 1]   # select positive feedback only

# 2. construct item co-occurrence matrix
cooccur = build_item_cooccur(pos_df)

# 3. evaluate over all users
all_recalls, all_hits = [], []

for user_id, items in pos_df.groupby("user_id")["item_id"]:
    items = list(items)
    if len(items) < 2:   # skip users with less than 2 items
        continue

    train_items = items[:-1]   # set last one as test
    test_items = items[-1:]

    recos = recommend_item2item(train_items, cooccur, topk=10)

    all_recalls.append(recall_at_k(test_items, recos, k=10))
    all_hits.append(hit_rate(test_items, recos, k=10))

print(f"Overall Recall@10: {np.mean(all_recalls):.4f}")
print(f"Overall HitRate@10: {np.mean(all_hits):.4f}")
