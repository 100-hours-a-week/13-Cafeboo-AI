from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
##from ai_project.routers.v1.pdf_router import router as pdf_router 
##from ai_project.routers.v1.caffeine_limit import router as caffeine_limit
##from ai_project.routers.v1.real_time_coffee import router as real_time_coffee
##from ai_project.routers.v1.input_moderation import router as input_moderation
##from ai_project.routers.v1.weekly_report import router as weekly_report
from ai_project.routers.v1.coffee_recommendation import router as coffee_recommendation
from ai_project.models.coffee_recommendation import CoffeeRecommendation
from ai_project.models.caffeine_limit import CaffeineLimit
from pydantic import BaseModel


app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 프론트엔드 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
##app.include_router(caffeine_limit)
##app.include_router(real_time_coffee)
##app.include_router(input_moderation)
##app.include_router(weekly_report)
app.include_router(coffee_recommendation) 
## app.include_router(pdf_router)

class UserProfile(BaseModel):
    gender: int
    age: int
    height: float
    weight: float
    is_smoker: int
    take_hormonal_contraceptive: int
    caffeine_sensitivity: int
    current_caffeine: int
    caffeine_limit: int
    residual_at_sleep: float
    target_residual_at_sleep: float
    planned_caffeine_intake: int
    current_time: float
    sleep_time: float

@app.post("/caffeine-limit")
async def get_caffeine_limit(profile: UserProfile):
    limit_model = CaffeineLimit()
    result = limit_model.calculate(profile.dict())
    return result

@app.post("/coffee-recommendation")
async def get_coffee_recommendation(profile: UserProfile):
    recommender = CoffeeRecommendation()
    recommendations = recommender.recommend(profile.dict())
    return recommendations