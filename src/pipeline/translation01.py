from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

zh_en_model="/Users/liushihao/PycharmProjects/hugging-face-demo/model/Helsinki-NLP/opus-mt-zh-en"
en_zh_model="/Users/liushihao/PycharmProjects/hugging-face-demo/model/Helsinki-NLP/opus-mt-en-zh"

tokenizer = AutoTokenizer.from_pretrained(en_zh_model)
model = AutoModelForSeq2SeqLM.from_pretrained(en_zh_model)


translation = pipeline("translation",
               model=model,
               tokenizer=tokenizer)

results = translation("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(results)




