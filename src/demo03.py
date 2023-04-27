from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")

encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)