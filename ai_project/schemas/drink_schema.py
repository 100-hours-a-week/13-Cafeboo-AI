#from pydantic import BaseModel
from typing import Optional

from pydantic import BaseModel
#from typing import List, Literal


class UserFeatures(BaseModel):
    gender: str
    age: int
    height: float
    weight: float
    is_pregnant: int
    is_taking_birth_pill: int
    is_smoking: int
    caffeine_sensitivity: int
    avg_intake_per_day: float
    avg_caffeine_amount: float
    daily_caffeine_limit: float


class DrinkScore(BaseModel):
    drink_id: int
    score: float
