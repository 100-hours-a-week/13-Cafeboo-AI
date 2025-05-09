from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
##from ai_project.routers.v1.pdf_router import router as pdf_router 
##from ai_project.routers.v1.caffeine_limit import router as caffeine_limit
##from ai_project.routers.v1.real_time_coffee import router as real_time_coffee
##from ai_project.routers.v1.input_moderation import router as input_moderation
##from ai_project.routers.v1.weekly_report import router as weekly_report
from ai_project.routers.v1.coffee_recommendation import router as coffee_recommendation
from ai_project.routers.v1.caffeine_limit import router as caffeine_limit_router
from ai_project.models.coffee_recommendation import CoffeeRecommendation
from ai_project.models.caffeine_limit import CaffeineLimit
from pydantic import BaseModel

app = FastAPI()

# 라우터 등록
##app.include_router(caffeine_limit)
##app.include_router(real_time_coffee)
##app.include_router(input_moderation)
##app.include_router(weekly_report)
app.include_router(coffee_recommendation)
app.include_router(caffeine_limit_router)
## app.include_router(pdf_router)
