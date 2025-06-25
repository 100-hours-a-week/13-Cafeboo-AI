import joblib
import numpy as np
import os
from pathlib import Path

class CaffeineLimitModel:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent  # ai_project/models/ → ai_project/
        model_path = base_dir / "data/models/caffeine_limit_model_v04.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        self.model = joblib.load(model_path)


    def preprocess(self, user_info: dict) -> np.ndarray:
        gender = 1 if user_info["gender"] == "M" else 0
        sleep_quality_map = {"bad": 0, "normal": 1, "good": 2}

        return np.array([[
            gender,
            user_info["age"],
            user_info["height"],
            user_info["weight"],
            user_info["is_smoker"],
            user_info["take_hormonal_contraceptive"],
            user_info["caffeine_sensitivity"]
           ## user_info["total_caffeine_today"],
            ##user_info["caffeine_intake_count"],
            ##user_info["first_intake_hour"],
            ##user_info["last_intake_hour"],
            ##user_info["sleep_duration"],
            ##sleep_quality_map.get(user_info["sleep_quality"], 1)
        ]])

    def predict(self, user_info: dict) -> float:
        features = self.preprocess(user_info)
        prediction = self.model.predict(features)
        return float(prediction[0])
