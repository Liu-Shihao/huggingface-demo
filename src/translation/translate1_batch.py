from transformers import AutoTokenizer

from src.translation.TRANS import train_data

"""
接下来我们就需要通过 DataLoader 库来按 batch 加载数据，将文本转换为模型可以接受的 token IDs。
对于翻译任务，我们需要运用分词器同时对源文本和目标文本进行编码，
这里我们选择 Helsinki-NLP 提供的汉英翻译模型 opus-mt-zh-en 对应的分词器

这个代码块
首先使用 tokenizer 对中文句子进行编码，并将编码结果保存到 inputs 变量中
然后它使用 tokenizer 对英文句子进行编码，并将编码结果保存到 targets 变量中。
在编码英文句子时，我们将中文句子传递给 text_pair 参数，以确保 tokenizer 生成正确的输入和输出序列。
最后，我们将 padding 和 truncation 参数都设置为 True，以便对所有句子进行相同的预处理，以便可以将它们传递给模型进行训练。
"""
model_checkpoint = "/Users/liushihao/PycharmProjects/hugging-face-demo/model/Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
"""
pt:PyTorch  tf:TensorFlow
当使用tokenizer处理文本时，有许多可用的参数。以下是本文提到的参数和代码的解释：

zh_sentence 和 en_sentence：这些是要处理的中文和英文句子，分别表示中文句子和对应的英文翻译句子。
padding=True：这个参数告诉 tokenizer 在对句子进行编码时添加填充（padding）到每个句子的结尾，以使它们具有相同的长度。这在将句子传递给模型进行训练时很有用。
truncation=True：这个参数告诉 tokenizer 如果句子超过了指定的最大长度，则截断它。这也有助于使所有输入句子具有相同的长度。
return_tensors="pt"：这个参数告诉 tokenizer 返回 PyTorch 张量格式的编码结果，以便可以将它们传递给 PyTorch 模型。如果使用 TensorFlow 模型，可以将其设置为 "tf"。
text_pair=zh_sentence：这个参数告诉 tokenizer 与中文句子相对应的英文句子是什么。在这种情况下，我们将中文句子作为主要输入，然后将英文句子作为配对文本（text pair）传递给 tokenizer，以便 tokenizer 可以生成正确的输入和输出序列。
"""

# 默认情况下分词器会采用源语言的设定来编码文本，要编码目标语言则需要通过上下文管理器
zh_sentence = train_data[0]["chinese"]
en_sentence = train_data[0]["english"]

# as_target_tokenizer 上下文管理器已被弃用 ，可以将 text_target 参数设置为需要对齐的源文本（即中文）
# inputs = tokenizer(zh_sentence)
# with tokenizer.as_target_tokenizer():
    # targets = tokenizer(en_sentence, padding=True, truncation=True, return_tensors="pt", text_pair=zh_sentence)
    # targets = tokenizer(en_sentence)
# 如果你忘记添加上下文管理器，就会使用源语言分词器对目标语言进行编码，产生糟糕的分词结果
# inputs = tokenizer(zh_sentence)
# targets = tokenizer(en_sentence, text_target=zh_sentence)

inputs = tokenizer(zh_sentence)
targets = tokenizer(en_sentence, padding=True, truncation=True, return_tensors="pt", text_target=zh_sentence)

wrong_targets = tokenizer(en_sentence)

print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
print(tokenizer.convert_ids_to_tokens(targets["input_ids"].tolist()[0]))
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))