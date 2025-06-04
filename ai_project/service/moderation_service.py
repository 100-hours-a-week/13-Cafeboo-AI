from ai_project.service.moderation_model_loader import moderation_model

class ModerationService:
    def moderate(selt, user_input: str) -> int:
        pred = moderation_model.predict(user_input)
        return pred
    

