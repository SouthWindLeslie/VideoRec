# VideoRec - Video Recommendation System

An end-to-end recommendation system project built on the MovieLens-100k dataset.

## Day1: Baseline
- Implemented a LightGBM baseline ranking model
- Encoded user and item IDs as categorical features
- Evaluated with AUC, Precision@10, and NDCG@10
- Results: AUC=0.71, P@10=0.65, NDCG@10=0.79

## Next Steps
- Day2: Candidate Retrieval (item2item, ANN)
- Day3: Serving API with FastAPI + Redis
