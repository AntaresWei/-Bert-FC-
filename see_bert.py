import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# 定义模型
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

# 初始化模型
num_labels = 2  # 二分类任务
model = BertForTextClassification(num_labels)

# 示例输入
text = "这是一个中文句子。"
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
inputs = tokenizer(text, return_tensors='pt')

# 前向传播
logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
print("Logits:", logits)

# 转换为概率
probs = F.softmax(logits, dim=1)
print("Probabilities:", probs)

# 获取预测类别
predicted_class = torch.argmax(probs, dim=1)
print("Predicted class:", predicted_class)