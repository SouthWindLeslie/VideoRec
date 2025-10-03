# VideoRec – Real-time Video Recommendation System

A two-stage video recommendation system built on **MovieLens-100k**, featuring **candidate recall + ranking**, offline evaluation, and a **FastAPI** serving layer (Docker-ready).  
This repo is a demo prototype that mirrors production architecture patterns.

---

## ✨ Features

- **Data Pipeline**
  - Load `data/ml-100k/u.data` (user_id, item_id, rating, timestamp)
  - Labeling: `label = 1 if rating >= 4 else 0`
  - Categorical encoding for `user_id` / `item_id`

- **Candidate Generation (Recall)**
  - Item-to-Item co-occurrence (simple, fast baseline)
  - For each user, recall Top-K candidates from history

- **Ranking**
  - **LightGBM** binary model
  - Minimal features: `user_id_enc`, `item_id_enc`
  - Outputs interaction probability for each (user, item)

- **Evaluation**
  - Offline metrics: **AUC**, **Precision@K**, **NDCG@K**, **Recall@K**, **HitRate@K**

- **Serving**
  - **FastAPI** endpoint: `/recommend/{user_id}?topk=10`
  - Swagger UI: `http://localhost:8000/docs`
  - Dockerfile included

---

## 🧭 Architecture

Demo (this repo):

```
MovieLens (u.data) → ETL (Pandas) → Recall (Item2Item) → Ranking (LightGBM)
→ FastAPI (Uvicorn) → REST API
```

Production reference (design vision):

```
User Logs → Kafka → Spark Streaming / Batch (Airflow) → Feature Store
→ Recall (ANN: Elasticsearch/Faiss + Popularity Priors)
→ Ranking (GBDT / Deep Models)
→ Serving (FastAPI) + Redis Cache → Online A/B Testing
```

---

## 📁 Project Structure

```
VideoRec/
├── data/                      # (ignored) run scripts/download_data.py
├── eval/
│   └── metrics.py             # Recall/HitRate/Precision/NDCG utilities
├── notebooks/
│   ├── day2_eval.py           # Global recall evaluation
│   ├── day3_pipeline.py       # Two-stage (recall + ranking) demo; saves model.txt
│   └── day1_baseline.py       # (optional) simple baseline
├── recsys/
│   ├── data_pipeline/
│   │   └── ingest.py          # load_movielens_100k()
│   ├── retrieval/
│   │   └── item2item.py       # co-occurrence recall
│   ├── ranking/
│   │   └── baseline_lightgbm.py
│   └── serving/
│       └── api.py             # FastAPI: / and /recommend/{user_id}
├── scripts/
│   └── download_data.py       # fetch & unzip MovieLens-100k → data/ml-100k/
├── model.txt                  # LightGBM model (saved by day3_pipeline)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

1) Install deps

```
pip install -r requirements.txt
```

2) Download dataset

```
python3 scripts/download_data.py
# Data goes to: data/ml-100k/u.data
```

3) Train & save model (creates `model.txt` in project root)

```
python3 -m notebooks.day3_pipeline
```

4) Run API

```
uvicorn recsys.serving.api:app --host 0.0.0.0 --port 8000
```

5) Test

```
curl "http://localhost:8000/"
curl "http://localhost:8000/recommend/1?topk=10"
# Swagger UI: http://localhost:8000/docs
```

Example response:

```json
{
  "user_id": 1,
  "recommendations": [
    {"item_id": 483, "score": 0.8927},
    {"item_id": 484, "score": 0.8927},
    {"item_id": 480, "score": 0.8651}
  ]
}
```

---

## 📈 Example Results (MovieLens-100k)

- AUC (ranking): ~ **0.71**
- Precision@10: ~ **0.65**
- NDCG@10: ~ **0.78**
- Recall@10 (recall layer): ~ **0.19**
- HitRate@10 (recall layer): ~ **0.19**

(Exact numbers may vary by random seed and splits.)

---

## 🐳 Docker

Build:

```
docker build -t videorec:latest .
```

Run:

```
docker run -it --rm   -v $(pwd):/app   -p 8000:8000   videorec:latest uvicorn recsys.serving.api:app --host 0.0.0.0 --port 8000
```

---

## 🔬 Metrics Glossary

- **AUC**: ranking discrimination quality
- **Precision@K**: fraction of top-K that are relevant
- **Recall@K**: fraction of relevant items covered by top-K
- **HitRate@K**: whether at least one relevant item appears in top-K
- **NDCG@K**: position-aware gain (higher rank → higher weight)

---

## 🛣️ Roadmap / Extensions

- Add content features (movie genres from `u.item`, user profile from `u.user`)
- ANN-based recall (Faiss / Elasticsearch)
- Caching layer (Redis) for hot users/items
- Feature store + scheduled ETL (Airflow), streaming ingestion (Kafka/Spark)
- Cold-start strategies (popularity, content-based)
- Online A/B testing (requires real traffic)

---

## 📚 References

- MovieLens 100k: https://grouplens.org/datasets/movielens/100k/
- LightGBM: https://github.com/microsoft/LightGBM
- FastAPI: https://fastapi.tiangolo.com/
- Recommender metrics (NDCG/Precision/Recall): standard IR definitions