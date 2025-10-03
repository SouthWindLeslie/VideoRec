import os
import requests
import zipfile
import io

def download_movielens_100k(target_dir="data/ml-100k"):
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    udata_path = os.path.join(target_dir, "u.data")

    # 如果已经存在就跳过
    if os.path.exists(udata_path):
        print(f"✅ Found existing dataset at {udata_path}, skip download.")
        return

    print("📥 Downloading MovieLens 100k dataset...")
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception("❌ Failed to download dataset.")

    # 解压到 data/ 文件夹
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("data/")
    print(f"✅ Dataset extracted to {target_dir}")

if __name__ == "__main__":
    download_movielens_100k()
