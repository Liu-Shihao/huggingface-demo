import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, TextDataset

# 初始化模型和tokenizer
from src.gtp2.MyDataset import MyDataset

model_name = '/Users/liushihao/PycharmProjects/hugging-face-demo/model/gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)


# 将数据集转换为问答对的形式
test_data = MyDataset('/Users/liushihao/PycharmProjects/hugging-face-demo/data/demo/test.json')
print(f'test set size: {len(test_data)}')
# 将问答对转换为模型输入格式
def prepare_dataset(qa_pairs):
    inputs = []
    for pair in qa_pairs:
        question, answer = pair['question'], pair['answer']
        input_text = f"Question: {question} Answer: {answer} "
        input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        inputs.append({'input_ids': input_ids})
    return inputs
train_inputs = prepare_dataset(test_data)
# 设置训练参数
batch_size = 4
learning_rate = 2e-5
epochs = 10
print(train_inputs)
# 设置训练参数 设置TrainingArguments，用于指定训练参数的详细信息，如输出目录、评估策略、每个设备的batch大小、训练周期数等。这些参数也需要根据数据集大小和模型复杂度进行调整。
# 初始化训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 训练模型和日志的输出目录
    overwrite_output_dir=True,       # 如果输出目录已经存在，则覆盖
    num_train_epochs=1,              # 训练的轮数
    per_device_train_batch_size=4,   # 每个设备上的训练批次大小
    per_device_eval_batch_size=4,    # 每个设备上的评估批次大小
    gradient_accumulation_steps=16,  # 梯度累积的步数
    save_steps=1000,                 # 每隔多少个步骤保存模型
    eval_steps=1000,                 # 每隔多少个步骤评估模型
    save_total_limit=10,             # 最多保存多少个模型
    learning_rate=2e-5,              # 学习率
    weight_decay=0.01,               # 权重衰减
    push_to_hub=False,               # 是否上传模型到 Hugging Face Hub
    logging_dir='./logs',            # 存储训练日志的目录
    logging_steps=500,               # 每隔多少个步骤记录训练日志
    save_strategy='steps',           # 模型保存策略
    evaluation_strategy='steps',
    load_best_model_at_end=True,     # 是否在训练结束时加载最佳模型
    metric_for_best_model='eval_loss',
    greater_is_better=False
)



# 设置Trainer
"""
使用Hugging Face库中的Trainer类来进行模型训练的初始化。
其中model是需要训练的模型：
training_args是训练参数，
train_data是训练数据集，
eval_data是验证数据集，
data_collator是一个函数，用于将数据转换为模型所需的格式。具体来说，这个data_collator函数将数据中的input_ids、attention_mask和labels打包成一个字典。其中，input_ids是表示输入文本的token ids，attention_mask是用于掩码输入序列中填充位置的二进制掩码，labels是用于训练的标签。函数中通过torch.stack将这些数据堆叠起来，形成一个tensor。最后，Trainer类会根据参数进行训练，并返回训练结果。
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_inputs,
    # eval_dataset=eval_data,
    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'labels': torch.stack([f[2] for f in data])}
)

# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained('my_model')
tokenizer.save_pretrained('my_model')
