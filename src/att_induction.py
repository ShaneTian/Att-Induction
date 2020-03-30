import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class AttentionInductionNetwork(nn.Module):
    def __init__(self, encoder, relation, hidden_size, max_length, induction_iters,
                 n_heads, dropout=0.1, current_device=torch.device("cpu")):
        super(AttentionInductionNetwork, self).__init__()
        self.encoder = encoder
        self.relation = relation
        self.hidden_size = hidden_size  # D
        self.max_length = max_length
        self.induction_iters = induction_iters
        self.current_device = current_device

        # Attention score: Multi-head attention + dropout + layernorm
        self.self_att = nn.MultiheadAttention(self.hidden_size, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        # Dynamic routing: Linear
        # self.fc_induction = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # 新transform
        self.fc_induction = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # self.CE_loss = nn.CrossEntropyLoss(reduction="mean")
        self.__init_params()

    def __init_params(self):
        nn.init.xavier_uniform_(self.fc_induction.weight)

    def __squash(self, inputs):
        # inputs: [B, *, D]
        # c: [B, totalQ, N, 1, D]
        inputs_norm = inputs.norm(dim=-1, keepdim=True)  # [B, *, 1]   c: [B, totalQ, N, 1, 1]
        return torch.div(torch.mul(inputs, inputs_norm),
                         torch.ones(inputs_norm.size(),
                                    device=self.current_device, requires_grad=False)
                         + inputs_norm.pow(2))
    
    def loss(self, predict_proba, label):
        # MSE loss
        N = predict_proba.size(-1)
        label_one_hot = F.one_hot(label.type(torch.long), num_classes=N)
        return F.mse_loss(predict_proba.view(-1, N),
                          label_one_hot.view(-1, N).type(torch.float),
                          reduction="sum")
    
    # def loss(self, predict_proba, label):
        # CE loss
    #     N = predict_proba.size(-1)
    #     return self.CE_loss(predict_proba.view(-1, N), label.view(-1))
    
    def mean_accuracy(self, predict_label, label):
        return torch.mean((predict_label.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def forward(self, support, support_mask, query, query_mask):
        """Attention Capsule Network forward.

        Args:
            support: torch.Tensor, [-1, N, K, max_length]
            support_mask: torch.Tensor, [-1, N, K, max_length]
            query: torch.Tensor, [-1, totalQ, max_length]
            query_mask: torch.Tensor, [-1, totalQ, max_length]
            
        Returns:
            relation_score: torch.Tensor, [B, totalQ, N]
            predict_label: torch.Tensor, [B, totalQ]"""
        B, N, K = support.size()[:3]
        totalQ = query.size()[1]  # Number of query instances for each batch
        
        # 1. Encoder
        support = support.view(-1, self.max_length)  # [B * N * K, max_length]
        support_mask = support_mask.view(-1, self.max_length)
        query = query.view(-1, self.max_length)  # [B * totalQ, max_length]
        query_mask = query_mask.view(-1, self.max_length)

        support = self.encoder(support, support_mask)  # [B * N * K, D]
        query = self.encoder(query, query_mask)  # [B * totalQ, D]
        support = support.view(-1, N, K, self.hidden_size)  # [B, N, K, D]
        query = query.view(-1, totalQ, self.hidden_size)  # [B, totalQ, D]

        # 2. Induction
        # 2.1 Attention score
        support_att = support.unsqueeze(1).expand(-1, totalQ, -1, -1, -1)  # [B, totalQ, N, K, D]
        query_att = query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, -1, -1) # [B, totalQ, N, 1, D]
        support_query_att = torch.cat((query_att, support_att), dim=3)  # [B, totalQ, N, 1 + K, D]
        support_query_att = support_query_att.view(-1, 1 + K, self.hidden_size).transpose(0, 1)  # [1 + K, B * totalQ * N , D]
        att_score = self.self_att(support_query_att, support_query_att, support_query_att)[0]  # [1 + K, B * totalQ * N , D]
        att_score = support_query_att + self.dropout(att_score)
        att_score = self.layer_norm(att_score)
        att_score = att_score[0].view(-1, totalQ, N, self.hidden_size)  # [B, totalQ, N, D]
        att_score = att_score.unsqueeze(3)  # [B, totalQ, N, 1, D]

        # 2.2 Attention capsule
        # support_hat = self.fc_induction(support).unsqueeze(1).expand(-1, totalQ, -1, -1, -1)  # [B, totalQ, N, K, D]
        # 新transform
        support_hat = self.__squash(self.fc_induction(support).unsqueeze(1).expand(-1, totalQ, -1, -1, -1))  # [B, totalQ, N, K, D]
        b = torch.zeros(B, totalQ, N, K, 1, device=self.current_device, requires_grad=False)  # [B, totalQ, N, K, 1]
        for _ in range(self.induction_iters):
            d = F.softmax(b, dim=3)  # [B, totalQ, N, K, 1]
            c_hat = torch.mul(d, support_hat).sum(3, keepdims=True)  # [B, totalQ, N, 1, D]
            c = self.__squash(c_hat)  # [B, totalQ, N, 1, D]
            # f1 = tanh or relu
            b = b + torch.mul(att_score, torch.tanh(torch.mul(support_hat, c))).sum(-1, keepdims=True)  # [B, totalQ, N, K, 1]
        
        # 3. Relation
        c = c.squeeze(3)  # [B, totalQ, N, D]
        query = query.unsqueeze(2).expand(-1, -1, N, -1)  # [B, totalQ, N, D]
        query = query.contiguous()
        relation_score = self.relation(c, query)  # [B, totalQ, N]

        predict_label = relation_score.argmax(dim=-1, keepdims=False)  # [B, totalQ]

        return relation_score, predict_label