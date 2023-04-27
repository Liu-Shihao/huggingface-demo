from torch.utils.data import Dataset
import json
'''
继承自 Dataset 类的自定义数据集类用于组织样本和标签
'''

max_dataset_size = 1000
train_set_size = 100
valid_set_size = 100
class MyDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        qa_pairs = []
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line)
                # print(f"question：{sample['question']} answer：{sample['answer']}")
                qa_pairs.append({'question': {sample['question']}, 'answer': {sample['answer']}})
        return qa_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# test_data = MyDataset('/Users/liushihao/PycharmProjects/hugging-face-demo/data/demo/test.json')
# print(f'test set size: {len(test_data)}')


