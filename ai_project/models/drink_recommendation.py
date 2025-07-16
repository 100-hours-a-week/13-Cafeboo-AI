# ai_project/models/drink_recommendation.py

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent

# ─── 파일 경로 ────────────────────────────────────────────────────────────────
MODEL_PT    = BASE / "data/models/drink_recommend_model_v1.pkl"
PREPROC_PKL = BASE / "data/models/user_preprocessor.pkl"
EMB_CSV     = BASE / "data/drink_embeddings.csv"

# ─── 1) 전처리기 로드 ────────────────────────────────────────────────────────────
user_preprocessor = joblib.load(PREPROC_PKL)

# ─── 2) 음료 임베딩 & 메타 정보 로드 ─────────────────────────────────────────────
emb_df = pd.read_csv(EMB_CSV)
drink_ids        = emb_df["drink_id"].values
drink_embeddings = emb_df.drop(columns=["drink_id"]).values  # (N_drink, emb_dim)

# ─── 3) dummy DataFrame 으로 input_dim 계산 ────────────────────────────────────
# ColumnTransformer 내부 구조에서 숫자/범주형 피처 목록 및 카테고리 획득
num_features = user_preprocessor.transformers_[0][2]
cat_features = user_preprocessor.transformers_[1][2]
ohe          = user_preprocessor.named_transformers_['cat']
categories   = ohe.categories_

# 숫자형은 0, 범주형은 각 카테고리의 첫 번째 값으로 채움
dummy = {}
for col in num_features:
    dummy[col] = 0
for col, cats in zip(cat_features, categories):
    dummy[col] = cats[0]

dummy_df  = pd.DataFrame([dummy])
input_dim = user_preprocessor.transform(dummy_df).shape[1]
emb_dim   = drink_embeddings.shape[1]

# ─── 4) UserTower 클래스 정의 ─────────────────────────────────────────────────
class UserTower(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, emb_dim),   nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

# ─── 5) UserTower 초기화 & 가중치 로드 ─────────────────────────────────────────
user_tower = UserTower(input_dim=input_dim, emb_dim=emb_dim)
state      = torch.load(MODEL_PT, map_location="cpu")

# Two-Tower 전체 state_dict 중 user_mlp 파트만 골라 로드
utm = {
    k.replace("user_mlp.", ""): v
    for k, v in state.items()
    if k.startswith("user_mlp.")
}
user_tower.net.load_state_dict(utm)
user_tower.eval()

# ─── 6) 외부에서 import 할 객체들 ────────────────────────────────────────────────
__all__ = [
    "user_preprocessor",
    "user_tower",
    "drink_ids",
    "drink_embeddings"
]
