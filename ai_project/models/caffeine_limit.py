import joblib
import numpy as np
import os

class CaffeineLimitModel:
    """
    LightGBM 기반 하루 최대 카페인 권장량 예측 모델
    """

    ##def __init__(self, model_path: str = "ai_project/service/models/caffeine_limit_model.pkl"):
    def __init__(self, model_path = os.path.join("service", "models", "caffeine_limit_model.pkl")):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        self.model = joblib.load(model_path)
        #print("✅ 모델 로딩 성공:", type(self.model))
        print("✅ 모델 로딩 성공:", model_path)

    def preprocess(self, user_info: dict) -> np.ndarray:
        """
        사용자 딕셔너리를 모델 입력 형식의 feature 배열로 변환
        """
        gender = 1 if user_info["gender"] == "M" else 0
        return np.array([[
            gender,
            user_info["age"],
            user_info["height"],
            user_info["weight"],
            user_info["is_smoker"],
            user_info["take_hormonal_contraceptive"],
            user_info["caffeine_sensitivity"],
            user_info["total_caffeine_today"],
            user_info["caffeine_intake_count"],
            user_info["first_intake_hour"],
            user_info["last_intake_hour"],
            user_info["sleep_duration"],
            {"bad": 0, "normal": 1, "good": 2}[user_info["sleep_quality"]]
        ]])

    def predict(self, user_info: dict) -> float:
        """
        예측된 최대 권장 카페인량 (mg) 반환
        
        ## features = self.preprocess(user_info)
        ## prediction = self.model.predict(features)
        ## return float(np.clip(prediction[0], 100, 600))  # 안전한 범위 제한
        """
        try:
            features = self.preprocess(user_info)
            return float(np.clip(self.model.predict(features)[0], 100, 600))
        except Exception as e:
            print("❌ 예측 중 오류 발생:", str(e))
            return None