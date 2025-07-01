import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class DrinkRecommendationLoader:
    def __init__(self):
        BASE = Path(__file__).resolve().parent.parent

        # 1) User 전처리기
        self.preproc_path = BASE / "data/models/user_preprocessor.pkl"
        self.user_preproc = joblib.load(self.preproc_path)

        # 2) Drink 임베딩 + 메타로드
        emb_csv = BASE / "data/drink_embeddings.csv"
        emb_df = pd.read_csv(emb_csv)
        self.drink_ids        = emb_df["drink_id"].values
        self.drink_embeddings = emb_df.drop(columns=["drink_id"]).values  # shape (N_drink, emb_dim)
        emb_dim = self.drink_embeddings.shape[1]

        # 3) UserTower 모델 정의 (학습 시와 동일 구조)
        #    Two-Tower 전체를 저장한 경우 user_mlp 키만 걸러서 load 합니다.
        class UserTower(nn.Module):
            def __init__(self, input_dim: int, emb_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(),
                    nn.Linear(64, emb_dim),   nn.ReLU()
                )
            def forward(self, x):
                return self.net(x)

        # 4) input_dim 계산 (dummy DataFrame→transform→shape)
        # 전처리기의 transformers_ 속성에서 컬럼 목록과 카테고리 정보를 가져옵니다.
        num_features = self.user_preproc.transformers_[0][2]
        cat_features = self.user_preproc.transformers_[1][2]
        ohe          = self.user_preproc.named_transformers_['cat']
        categories   = ohe.categories_

        # 숫자형은 0, 범주형은 각 카테고리의 첫 번째 값으로 채움
        dummy = {}
        for col in num_features:
            dummy[col] = 0
        for col, cats in zip(cat_features, categories):
            dummy[col] = cats[0]

        dummy_df  = pd.DataFrame([dummy])
        input_dim = self.user_preproc.transform(dummy_df).shape[1]

        # 5) UserTower 인스턴스 생성 & state_dict 로드
        self.user_tower = UserTower(input_dim=input_dim, emb_dim=emb_dim)
        state = torch.load(BASE / "data/models/drink_recommend_model_v1.pkl",
                           map_location="cpu")
        # user_mlp.* 키만 골라 내어 net에 매핑
        utm = {
            k.replace("user_mlp.", ""): v
            for k, v in state.items()
            if k.startswith("user_mlp.")
        }
        self.user_tower.net.load_state_dict(utm)
        self.user_tower.eval()
        self.device = torch.device("cpu")
        self.user_tower.to(self.device)

    def recommend(self, user_dict: dict, top_n: int = 3) -> list[dict]:
        # 1) 누락된 컬럼 0으로 채워 DataFrame 생성
        record = {c: user_dict.get(c, 0) for c in self.user_preproc.feature_names_in_}
        df_u = pd.DataFrame([record])

        # 2) 전처리 → tensor
        X_u = self.user_preproc.transform(df_u).astype(np.float32)
        t_u = torch.from_numpy(X_u).to(self.device)

        # 3) user embedding
        with torch.no_grad():
            u_emb = self.user_tower(t_u).cpu().numpy()[0]  # (emb_dim,)

        # 4) 내적 연산 → scores
        scores = self.drink_embeddings.dot(u_emb)       # (N_drink,)

        # 5) Top-N 인덱스
        idx = np.argsort(scores)[-top_n:][::-1]

        # 6) 결과 조립
        return [
            {"drink_id": int(self.drink_ids[i]), "score": float(scores[i])}
            for i in idx
        ]

# 앱 시작 시 한 번만 로드
drink_recommender = DrinkRecommendationLoader()
