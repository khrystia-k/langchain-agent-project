import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, RobertaTokenizer


class TopicClassifier:
    def __init__(
        self,
        base_model="roberta-base",
        saved_model_path="./saved_model",
        num_labels=5,
        threshold=0.6,
    ):
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)
        base_model_instance = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=num_labels
        )
        self.model = PeftModel.from_pretrained(base_model_instance, saved_model_path)
        self.model.eval()  # set model to evaluation mode
        self.class_names = [
            "Politics",
            "Sport",
            "Technology",
            "Entertainment",
            "Business",
        ]
        self.id2label = {i: label for i, label in enumerate(self.class_names)}
        self.threshold = threshold

    def classify(self, query):
        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():  # disable gradient computations for faster inference
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        max_prob, predicted_class = torch.max(probabilities, dim=-1)
        max_prob = max_prob.item()

        if max_prob < self.threshold:
            return "Unknown"
        return self.id2label[predicted_class.item()]


if __name__ == "__main__":
    classifier = TopicClassifier()
    sample_query = "The latest advancements in AI have taken the tech world by storm."
    result = classifier.classify(sample_query)
    print(f"Predicted Topic: {result}")
