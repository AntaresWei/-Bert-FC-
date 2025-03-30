import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer  # 确保导入 BertModel
import pandas as pd

# 定义模型（与训练时的定义一致）
class BertForTextClassification(nn.Module):
    def __init__(self, num_labels):
        super(BertForTextClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] token 的输出
        logits = self.classifier(cls_output)
        return logits

# 自定义数据集（与训练时的定义一致）
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

# 加载测试数据
def load_test_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    texts = data['text_a'].tolist()  # 假设测试数据有 'text_a' 列
    return texts[0:9] # 只检测前10行

# 加载模型
def load_model(model_path, num_labels):
    model = BertForTextClassification(num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # 小电脑没有gpu……
    model.eval()  # 设置为评估模式
    return model

# 预测函数
def predict(model, tokenizer, test_texts, batch_size=16):
    # 创建测试数据集和数据加载器
    test_dataset = TextDataset(test_texts, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 预测
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 前向传播
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 获取预测类别
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

# 主函数
def main():
    # 文件路径
    test_file_path = './chnsenticorp/test.tsv'  # 测试数据路径
    model_path = './save_model/bert_text_classification.pth'  # 训练好的模型路径

    # 加载测试数据
    test_texts = load_test_data(test_file_path)
    label_map = {
        0:"Negative",
        1:"Positive"
    }

    # 加载模型
    num_labels = 2  # 类别数（与训练时一致）
    model = load_model(model_path, num_labels)

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 进行预测
    predictions = predict(model, tokenizer, test_texts)

    # 输出预测结果
    flag = 10 # 只展示前10项
    for text, pred in zip(test_texts, predictions):
        flag = flag - 1
        print(f"Text: {text}")
        print(f"Predicted Label: {label_map[pred]}")
        print("-" * 50)
        if flag ==0:
            break

if __name__ == '__main__':
    main()