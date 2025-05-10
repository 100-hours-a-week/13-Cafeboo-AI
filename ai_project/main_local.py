from fastapi import FastAPI
from ai_project.routers.v1 import (
   caffeine_limit,
   ## real_time_coffee,
   ## input_moderation,
   ## weekly_report,
   coffee_recommendation
)

app = FastAPI()

# 라우터 등록
app.include_router(caffeine_limit.router)
## app.include_router(real_time_coffee.router)
## app.include_router(input_moderation.router)
## app.include_router(weekly_report.router)
app.include_router(coffee_recommendation.router) 