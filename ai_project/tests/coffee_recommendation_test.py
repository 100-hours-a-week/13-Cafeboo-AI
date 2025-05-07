from models.coffee_recommendation import CoffeeRecommendationModel

user_input = {
    "gender": 1,
    "age": 27,
    "height": 165.0,
    "weight": 45.1,
    "is_smoker": 1,
    "take_hormonal_contraceptive": 0,
    "caffeine_sensitivity": 90,
    "current_caffeine": 340,
    "caffeine_limit": 400,
    "residual_at_sleep": 38.7,
    "target_residual_at_sleep": 50.0,
    "planned_caffeine_intake": 125,
    "current_time": 15.5,
    "sleep_time": 17.5
}

model = CoffeeRecommendationModel()
result = model.predict(user_input)

print("🧠 예측 결과:")
print(f" - 마셔도 됨 여부: {'YES' if result['can_drink'] else 'NO'}")
print(f" - 확률(1일 확률): {result['probability']:.2f}")
