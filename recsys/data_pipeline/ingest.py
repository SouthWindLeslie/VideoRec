import pandas as pd

def load_movielens_100k(path="data/ml-100k/u.data"):
    cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="\t", names=cols)
    df["label"] = (df["rating"] >= 4).astype(int)
    return df[["user_id", "item_id", "label", "timestamp"]]

if __name__ == "__main__":
    df = load_movielens_100k()
    print(df.head())
    print("Label distribution:\n", df.label.value_counts(normalize=True).round(3))
