# 导入必要的库
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class MyDataset(Dataset):
    # 初始化数据集参数，这里设置了图像的存储目录、标签（通过读取标签 csv 文件）以及样本和标签的数据转换函数；
    def __init__(self, data_file):
        self.data = []
        with open(data_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    self.data.append(line)

    # 返回数据集中样本的个数；
    def __len__(self):
        return len(self.data)
    # 映射型数据集的核心，根据给定的索引 idx 返回样本。这里会根据索引从目录下通过
    def __getitem__(self, index):
        return self.data[index]

# 初始化模型和tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 设置训练参数
batch_size = 4
learning_rate = 2e-5
epochs = 10

# 准备数据集
dataset = MyDataset('/Users/liushihao/PycharmProjects/hugging-face-demo/data/baike2018qa/baike_qa_train.json')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 设置优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
for epoch in range(epochs):
    for batch in dataloader:
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        outputs = model(inputs['input_ids'], labels=inputs['input_ids'])
        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), inputs['input_ids'].view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('Epoch: {}, Loss: {}'.format(epoch+1, loss.item()))

# 保存模型
model.save_pretrained('my_model')
tokenizer.save_pretrained('my_model')
