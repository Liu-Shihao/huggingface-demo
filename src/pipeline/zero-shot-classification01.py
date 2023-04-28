from transformers import pipeline
'''
零训练样本分类 
pipeline 允许我们在不提供任何标注数据的情况下自定义分类标签。
pipeline 自动选择了预训练好的 facebook/bart-large-mnli 模型来完成任务。


'''
classifier = pipeline("zero-shot-classification")
result = classifier(
"This is a course about the Transformers library",
candidate_labels=["education", "politics", "business"],
)
print(result)