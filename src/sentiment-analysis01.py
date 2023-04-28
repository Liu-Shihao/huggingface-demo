from transformers import pipeline
'''
借助情感分析 pipeline，我们只需要输入文本，就可以得到其情感标签（积极/消极）以及对应的概率：
pipeline 模型会自动完成以下三个步骤：

将文本预处理为模型可以理解的格式；
将预处理好的文本送入模型；
对模型的预测值进行后处理，输出人类可以理解的格式。
pipeline 会自动选择合适的预训练模型来完成任务。例如对于情感分析，默认就会选择微调好的英文情感模型 distilbert-base-uncased-finetuned-sst-2-english。
'''
classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
print(result)
results = classifier(
  ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(results)