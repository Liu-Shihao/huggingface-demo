from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

"""
max_length参数控制生成的翻译文本的最大长度，
num_beams参数控制beam search算法中生成的翻译候选句子的数量。
生成的翻译文本将存储在translated_text变量中并打印出来。
"""

model_name = '/Users/liushihao/PycharmProjects/hugging-face-demo/model/Helsinki-NLP/opus-mt-zh-en'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


input_text = "今天天气真不错！"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids=input_ids, max_length=128, num_beams=4, early_stopping=True)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)