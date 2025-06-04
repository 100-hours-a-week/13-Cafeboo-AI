from transformers import BertForSequenceClassification, AutoTokenizer
import torch

class ModerationModelLoader:
    def __init__(self):

        self.model_name = "klue/bert-base"
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.state_dict = torch.load("ai_project/models/best_model.pt", map_location=torch.device("cpu"))
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

    def predict(self, text: str) -> int:
        user_input = text
        encoded_input = self.tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            logits = outputs.logits 
            pred = torch.argmax(logits, dim=1).item()
        return pred

moderation_model = ModerationModelLoader()
