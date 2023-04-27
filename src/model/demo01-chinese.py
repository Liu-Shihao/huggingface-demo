from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline


model_name = "bert-base-chinese"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
results = classifier(["我爱你", "我恨你"])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")