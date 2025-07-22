import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

class ModerationModelLoaderV2:
    def __init__(self):
        self.sess_options = ort.SessionOptions()
        self.model_roberta_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
        self.model_electra_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
        self.electra_sess = ort.InferenceSession("ai_project/models/ELECTRA.quant.onnx", providers=["CPUExecutionProvider"])
        self.roberta_sess = ort.InferenceSession("ai_project/models/roberta.quant.onnx", providers=["CPUExecutionProvider"])
    

    

    def prepare_onnx_input(self, text, tokenizer):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=300)
        return {
            "input_ids": inputs["input_ids"].cpu().numpy(),
            "attention_mask": inputs["attention_mask"].cpu().numpy()
        }
    def softmax(self, x, axis=None):
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
    def ensemble_check_moderation_onnx(self, text, model_weights=None, threshold=0.5):
        if model_weights is None:
            model_weights = {
                "electra": 0.27,
                "roberta": 0.36,
            }

        onnx_models = {
            "electra": (self.electra_sess, self.model_electra_tokenizer),
            "roberta": (self.roberta_sess, self.model_roberta_tokenizer),
        }

        total_score = 0

        for name, weight in model_weights.items():
            session, tokenizer = onnx_models[name]
            ort_inputs = self.prepare_onnx_input(text, tokenizer)
            ort_outs = session.run(None, ort_inputs)
            logits = ort_outs[0]
            probs = self.softmax(logits, axis=-1)
            hate_prob = probs[0][0]  # 클래스 0 = 혐오
            total_score += weight * hate_prob

        
        final_pred = 0 if total_score > threshold else 1
        

        return final_pred

moderation_model = ModerationModelLoaderV2()