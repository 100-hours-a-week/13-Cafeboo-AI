import numpy as np
import pandas as pd
import torch

from ai_project.models.drink_recommendation import (
    user_preprocessor, user_tower,
    drink_ids, drink_embeddings
)

device = torch.device("cpu")  # CPU inference

def recommend_top3_drinks(user_dict: dict, top_n: int = 3) -> list[dict]:
    # 1) 사용자 전처리 → torch.Tensor
    df_u = pd.DataFrame([user_dict])
    X_u  = user_preprocessor.transform(df_u).astype(np.float32)
    t_u  = torch.from_numpy(X_u).to(device)

    # 2) user embedding
    with torch.no_grad():
        u_emb = user_tower(t_u).cpu().numpy()[0]  # (emb_dim,)

    # 3) 내적 연산으로 점수 계산
    scores = drink_embeddings.dot(u_emb)        # shape: (N_drink,)

    # 4) 상위 N개 인덱스 추출
    idx = np.argsort(scores)[-top_n:][::-1]

    # 5) 결과 조립
    return [
        {"drink_id": int(drink_ids[i]), "score": float(scores[i])}
        for i in idx
    ]
