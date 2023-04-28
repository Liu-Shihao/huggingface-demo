import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

'''
情感分析 模型&分词器

因为神经网络模型无法直接处理文本，因此首先需要通过预处理环节将文本转换为模型可以理解的数字。具体地，我们会使用每个模型对应的分词器 (tokenizer) 来进行
'''
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# 使用分词器进行预处理
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

'''
{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
        
可以看到，输出中包含两个键 input_ids 和 attention_mask，
其中 input_ids 对应分词之后的 tokens 映射到的数字编号列表，
而 attention_mask 则是用来标记哪些 tokens 是被填充的（这里“1”表示是原文，“0”表示是填充字符）。
'''

#将预处理好的输入送入模型
'''
预训练模型的本体只包含基础的 Transformer 模块，对于给定的输入，它会输出一些神经元的值，称为 hidden states 或者特征 (features)。
对于 NLP 模型来说，可以理解为是文本的高维语义表示。
这些 hidden states 通常会被输入到其他的模型部分（称为 head），以完成特定的任务，例如送入到分类头中完成文本分类任务。
'''
model = AutoModel.from_pretrained(checkpoint)

# 打印出这里使用的 distilbert-base 模型的输出维度:
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
'''
torch.Size([2, 16, 768]) 例如 Bert 模型 base 版本的输出为 768 维
Transformers 模型的输出格式类似 namedtuple 或字典，可以像上面那样通过属性访问，也可以通过键（outputs["last_hidden_state"]），甚至索引访问（outputs[0]）。

'''

'''
Transformers 库封装了很多不同的结构，常见的有：

*Model （返回 hidden states）
*ForCausalLM （用于条件语言模型）
*ForMaskedLM （用于遮盖语言模型）
*ForMultipleChoice （用于多选任务）
*ForQuestionAnswering （用于自动问答任务）
*ForSequenceClassification （用于文本分类任务）
*ForTokenClassification （用于 token 分类任务，例如 NER）
'''
# 对于情感分析任务，很明显我们最后需要使用的是一个文本分类 head。因此，实际上我们不会使用 AutoModel 类，而是使用 AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
'''
torch.Size([2, 2])
可以看到，对于 batch 中的每一个样本，模型都会输出一个两维的向量（每一维对应一个标签，positive 或 negative）。
'''

# 对模型输出进行后处理
print(outputs.logits)
'''
由于模型的输出只是一些数值，因此并不适合人类阅读。例如我们打印出上面例子的输出：
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward0>)
所有 Transformers 模型都会输出 logits 值，因为训练时的损失函数通常会自动结合激活函数（例如 SoftMax）与实际的损失函数（例如交叉熵 cross entropy）。

'''

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)
