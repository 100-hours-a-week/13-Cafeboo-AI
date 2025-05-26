import joblib
import numpy as np
import os
from pathlib import Path

class CoffeeRecommendationModel:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent  # ai_project/models/ → ai_project/
        model_path = base_dir / "data/models/can_drink_model_v02.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        self.model = joblib.load(model_path)

    def predict(self, user_info: dict) -> dict:
        gender = 1 if user_info["gender"] == "M" else 0
        features = np.array([[
            gender,
            ##user_info["gender"],
            user_info["age"],
            user_info["height"],
            user_info["weight"],
            user_info["is_smoker"],
            user_info["take_hormonal_contraceptive"],
            user_info["caffeine_sensitivity"],
            user_info["current_caffeine"],
            user_info["caffeine_limit"],
            user_info["residual_at_sleep"],
            user_info["target_residual_at_sleep"],
           ## user_info["planned_caffeine_intake"],
            user_info["current_time"],
            user_info["sleep_time"],
        ]])
        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0][1]
        return {
            "can_drink": bool(pred),
            "probability": round(prob, 3)
        }
