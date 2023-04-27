from torch.utils.data import Dataset, random_split
import json
"""
首先编写继承自 Dataset 类的自定义数据集类用于组织样本和标签
考虑到 translation2019zh 并没有提供测试集，而且使用五百多万条样本进行训练耗时过长，
这里我们只抽取训练集中的前 22 万条数据，并从中划分出 2 万条数据作为验证集，
然后将 translation2019zh 中的验证集作为测试集：
"""
max_dataset_size = 220000
train_set_size = 200000
valid_set_size = 20000

# 构建数据集
class TRANS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = TRANS('../../data/translation2019zh/translation2019zh_train.json')
train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
test_data = TRANS('../../data/translation2019zh/translation2019zh_valid.json')

print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data)))


