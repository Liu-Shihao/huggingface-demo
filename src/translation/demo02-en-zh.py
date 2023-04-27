from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

input_text = "My name is Wolfgang and I live in Berlin."
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids=input_ids, max_length=128, num_beams=4, early_stopping=True)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)