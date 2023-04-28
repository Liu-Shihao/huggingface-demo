from transformers import pipeline
'''
前缀模板 (Preﬁx Prompt)
我们首先根据任务需要构建一个模板 (prompt)，然后将其送入到模型中来生成后续文本。注意，由于文本生成具有随机性，因此每次运行都会得到不同的结果。
'''
generator = pipeline("text-generation")
results = generator("In this course, we will teach you how to")
print(results)
results = generator(
    "In this course, we will teach you how to",
    num_return_sequences=2,
    max_length=50
)
print(results)

'''
可以看到，pipeline 自动选择了预训练好的 gpt2 模型来完成任务。我们也可以指定要使用的模型。
对于文本生成任务，我们可以在 Model Hub 页面左边选择 Text Generation tag 查询支持的模型。例如，我们在相同的 pipeline 中加载 distilgpt2 模型：
'''
generator = pipeline("text-generation", model="distilgpt2")
results = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(results)

'''
还可以通过左边的语言 tag 选择其他语言的模型。例如加载专门用于生成中文古诗的 gpt2-chinese-poem 模型：
'''
generator = pipeline("text-generation", model="/Users/liushihao/PycharmProjects/hugging-face-demo/model/uer/gpt2-chinese-poem")
results = generator(
    "[CLS] 万 叠 春 山 积 雨 晴 ，",
    max_length=40,
    num_return_sequences=2,
)
print(results)