from ai_project.service.moderation_model_loader import moderation_model
from fastapi import HTTPException
class ModerationService:
    def moderate(selt, user_input: str) -> dict:
        try:
            pred = moderation_model.predict(user_input)
            if pred == 1:
                return {"is_toxic": pred, "message": "유해하지 않은 문장입니다."}
            else:
                return {"is_toxic": pred, "message": "유해한 문장입니다."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
moderation_service = ModerationService()
