https://huggingface.co/

https://github.com/huggingface/transformers
https://github.com/huggingface/transformers/blob/main/README_zh-hans.md
https://huggingface.co/docs/transformers/v4.28.1/zh/index

# pipelines
Transformers 库将目前的 NLP 任务归纳为几下几类：
- 文本分类：例如情感分析、句子对关系判断等；
- 对文本中的词语进行分类：例如词性标注 (POS)、命名实体识别 (NER) 等；
- 文本生成：例如填充预设的模板 (prompt)、预测文本中被遮掩掉 (masked) 的词语；
- 从文本中抽取答案：例如根据给定的问题从一段文本中抽取出对应的答案；
- 根据输入文本生成新的句子：例如文本翻译、自动摘要等。

Transformers 库最基础的对象就是 pipeline() 函数，它封装了预训练模型和对应的前处理和后处理环节。我们只需输入文本，就能得到预期的答案。

目前常用的 pipelines 有：
使用pipeline()是利用预训练模型进行推理的最简单的方式. 你能够将pipeline()开箱即用地用于跨不同模态的多种任务.
- 文本分类	为给定的文本序列分配一个标签	NLP	pipeline(task=“sentiment-analysis”)
- 文本生成	根据给定的提示生成文本	NLP	pipeline(task=“text-generation”)
- 命名实体识别	为序列里的每个token分配一个标签(人, 组织, 地址等等)	NLP	pipeline(task=“ner”)
- 问答系统	通过给定的上下文和问题, 在文本中提取答案	NLP	pipeline(task=“question-answering”)
- 掩盖填充	预测出正确的在序列中被掩盖的token	NLP	pipeline(task=“fill-mask”)
- 文本摘要	为文本序列或文档生成总结	NLP	pipeline(task=“summarization”)
- 文本翻译	将文本从一种语言翻译为另一种语言	NLP	pipeline(task=“translation”)
- 图像分类	为图像分配一个标签	Computer vision	pipeline(task=“image-classification”)
- 图像分割	为图像中每个独立的像素分配标签(支持语义、全景和实例分割)	Computer vision	pipeline(task=“image-segmentation”)
- 目标检测	预测图像中目标对象的边界框和类别	Computer vision	pipeline(task=“object-detection”)
- 音频分类	给音频文件分配一个标签	Audio	pipeline(task=“audio-classification”)
- 自动语音识别	将音频文件中的语音提取为文本	Audio	pipeline(task=“automatic-speech-recognition”)
- 视觉问答	给定一个图像和一个问题，正确地回答有关图像的问题	Multimodal	pipeline(task=“vqa”)

![pipeling-task](imgs/pipeline_task.png)

## pipeline 处理流程
这些简单易用的 pipeline 模型实际上封装了许多操作，下面我们就来了解一下它们背后究竟做了啥
实际上它的背后经过了三个步骤：

1. 预处理 (preprocessing)，将原始文本转换为模型可以接受的输入格式；
2. 将处理好的输入送入模型；
3. 对模型的输出进行后处理 (postprocessing)，将其转换为人类方便阅读的格式。
# 使用分词器进行预处理
因为神经网络模型无法直接处理文本，因此首先需要通过预处理环节将文本转换为模型可以理解的数字。具体地，我们会使用每个模型对应的分词器 (tokenizer) 来进行：

1. 将输入切分为词语、子词或者符号（例如标点符号），统称为 tokens；
2. 根据模型的词表将每个 token 映射到对应的 token 编号（就是一个数字）；
3. 根据模型的需要，添加一些额外的输入。

我们对输入文本的预处理需要与模型自身预训练时的操作完全一致，只有这样模型才可以正常地工作。
注意，每个模型都有特定的预处理操作，如果对要使用的模型不熟悉，可以通过 [https://huggingface.co/models](https://huggingface.co/models) 查询。

这里我们使用 AutoTokenizer 类和它的 from_pretrained() 函数，它可以自动根据模型 checkpoint 名称来获取对应的分词器。