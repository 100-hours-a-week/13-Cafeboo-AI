import requests

url = "http://127.0.0.1:8000/v1/caffeine-limit/predict"
payload = {
    "user_id": "user001",
    "gender": "M",
    "age": 29,
    "height": 175.0,
    "weight": 70.0,
    "is_smoker": 0,
    "take_hormonal_contraceptive": 0,
    "caffeine_sensitivity": 60,
    "total_caffeine_today": 200,
    "caffeine_intake_count": 2,
    "first_intake_hour": 8,
    "last_intake_hour": 15,
    "sleep_duration": 7.0,
    "sleep_quality": "normal"
}
response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
