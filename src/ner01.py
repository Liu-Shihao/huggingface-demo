from transformers import pipeline
'''
命名实体识别 (NER) pipeline 负责从文本中抽取出指定类型的实体，例如人物、地点、组织等等。
'''

ner = pipeline("ner",
               model="/Users/liushihao/PycharmProjects/hugging-face-demo/model/dbmdz/bert-large-cased-finetuned-conll03-english",
               grouped_entities=True)
results = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(results)
'''
[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18}, 
{'entity_group': 'ORG', 'score': 0.9796021, 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
{'entity_group': 'LOC', 'score': 0.9932106, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
可以看到，模型正确地识别出了 Sylvain 是一个人物，Hugging Face 是一个组织，Brooklyn 是一个地名。
这里通过设置参数 grouped_entities=True，使得 pipeline 自动合并属于同一个实体的多个子词 (token)，例如这里将“Hugging”和“Face”合并为一个组织实体，实际上 Sylvain 也进行了子词合并，因为分词器会将 Sylvain 切分为 S、##yl 、##va 和 ##in 四个 token。
'''