import torch
import torch.nn as nn

class LongformerClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(LongformerClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features.last_hidden_state[:, 0, :]  # the [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LongformerForSequenceClassification(nn.Module):
    def __init__(self, longformer, hidden_size, num_labels):
        super(LongformerForSequenceClassification, self).__init__()
        self.longformer = longformer
        self.classification_head = LongformerClassificationHead(hidden_size, num_labels)

    def forward(self, x, attention_mask=None):
        x = self.longformer(x, attention_mask)
        x = self.classification_head(x)
        return x
