from transformers import pipeline
'''
自动问答 pipeline 可以根据给定的上下文回答问题
可以看到，pipeline 自动选择了在 SQuAD 数据集上训练好的 distilbert-base 模型来完成任务。这里的自动问答 pipeline 实际上是一个抽取式问答模型，即从给定的上下文中抽取答案，而不是生成答案。

根据形式的不同，自动问答 (QA) 系统可以分为三种：

抽取式 QA (extractive QA)：假设答案就包含在文档中，因此直接从文档中抽取答案；
多选 QA (multiple-choice QA)：从多个给定的选项中选择答案，相当于做阅读理解题；
无约束 QA (free-form QA)：直接生成答案文本，并且对答案文本格式没有任何限制。
'''
question_answerer = pipeline("question-answering",
                             model="/Users/liushihao/PycharmProjects/hugging-face-demo/model/distilbert-base-cased-distilled-squad")
answer = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
print(answer)