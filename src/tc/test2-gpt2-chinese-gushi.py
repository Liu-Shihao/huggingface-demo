from transformers import pipeline

generator = pipeline("text-generation", model="/Users/liushihao/PycharmProjects/hugging-face-demo/model/uer/gpt2-chinese-poem")
results = generator(
    "[CLS] 万 叠 春 山 积 雨 晴 ，",
    max_length=40,
    num_return_sequences=2,
)
print(results)