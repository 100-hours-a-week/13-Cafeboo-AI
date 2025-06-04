from pydantic import BaseModel

class ErrorData(BaseModel):
    code: str
    detail: str

class ErrorResponse(BaseModel):
    status: str
    message: str
    data: ErrorData