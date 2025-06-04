from ai_project.service.moderation_model_loader import moderation_model

class ModerationService:
    def moderate(selt, user_input: str) -> dict:
        pred = moderation_model.predict(user_input)
        print(pred.dtype)
        return {"is_toxic": pred}
    

