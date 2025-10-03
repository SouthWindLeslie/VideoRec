import pandas as pd
from collections import defaultdict

def build_item_cooccur(df: pd.DataFrame):
    """
    Build item co-occurrence matrix from user-item interactions.
    df: DataFrame with columns [user_id, item_id]
    """
    cooccur = defaultdict(lambda: defaultdict(int))
    for user, items in df.groupby("user_id")["item_id"]:
        items = list(items)
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                cooccur[items[i]][items[j]] += 1
                cooccur[items[j]][items[i]] += 1
    return cooccur

def recommend_item2item(user_items, cooccur, topk=10):
    """
    Recommend items for a user based on item co-occurrence.
    user_items: list of items the user interacted with
    cooccur: co-occurrence matrix
    topk: number of items to recommend
    """
    scores = defaultdict(int)
    for item in user_items:
        for related, count in cooccur[item].items():
            if related not in user_items:  # 不推荐已看过的
                scores[related] += count
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:topk]]
