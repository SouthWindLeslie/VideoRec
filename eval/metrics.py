def recall_at_k(actual, predicted, k=10):
    """Recall@K: fraction of relevant items found among top-K."""
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return len(actual_set & predicted_set) / len(actual_set) if actual_set else 0.0

def hit_rate(actual, predicted, k=10):
    """HitRate@K: whether at least one relevant item is in top-K."""
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    return 1.0 if len(actual_set & predicted_set) > 0 else 0.0
