from models.caffeine_limit import CaffeineLimitModel

# 가상 사용자 정보
user_info = {
    "gender": "M",
    "age": 28,
    "height": 175,
    "weight": 70,
    "is_smoker": 0,
    "take_hormonal_contraceptive": 0,
    "caffeine_sensitivity": 60,
    "total_caffeine_today": 200,
    "caffeine_intake_count": 2,
    "first_intake_hour": 9,
    "last_intake_hour": 15,
    "sleep_duration": 7,
    "sleep_quality": "normal"
}

# 모델 인스턴스 생성
model = CaffeineLimitModel()

# 예측 실행
recommended_limit = model.predict(user_info)

# 결과 출력
print("모델 타입:", type(model.model))
print(f"예측된 하루 카페인 권장량: {recommended_limit:.2f} mg")
