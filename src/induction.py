import torch
import torch.nn as nn
from torch.nn import functional as F


class InductionNetwork(nn.Module):
    def __init__(self, encoder, relation, hidden_size, max_length, induction_iters,
                 current_device=torch.device("cpu")):
        super(InductionNetwork, self).__init__()
        self.encoder = encoder
        self.relation = relation
        self.hidden_size = hidden_size  # D
        self.max_length = max_length
        self.induction_iters = induction_iters
        self.current_device = current_device

        # Dynamic routing: Linear
        self.fc_induction = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.__init_params()

    def __init_params(self):
        nn.init.xavier_uniform_(self.fc_induction.weight)

    def __squash(self, inputs):
        # inputs: [B, totalQ, N, 1, D]
        inputs_norm = inputs.norm(dim=-1, keepdim=True)  # [B, totalQ, N, 1, 1]
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
        
    def mean_accuracy(self, predict_label, label):
        return torch.mean((predict_label.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def forward(self, support, support_mask, query, query_mask):
        """Induction Networks forward.

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
        support_hat = self.__squash(self.fc_induction(support))  # [B, N, K, D]
        b = torch.zeros(B, N, K, 1, device=self.current_device, requires_grad=False)  # [B, N, K, 1]
        for _ in range(self.induction_iters):
            d = F.softmax(b, dim=2)  # [B, N, K, 1]
            c_hat = torch.mul(d, support_hat).sum(2, keepdims=True)  # [B, N, 1, D]
            c = self.__squash(c_hat)  # [B, N, 1, D]
            b = b + torch.mul(support_hat, c).sum(-1, keepdims=True)  # [B, N, K, 1]
        
        # 3. Relation
        c = c.squeeze(2).unsqueeze(1).expand(-1, totalQ, -1, -1).contiguous()  # [B, totalQ, N, D]
        query = query.unsqueeze(2).expand(-1, -1, N, -1).contiguous()  # [B, totalQ, N, D]
        relation_score = self.relation(c, query)  # [B, totalQ, N]

        predict_label = relation_score.argmax(dim=-1, keepdims=False)  # [B, totalQ]

        return relation_score, predict_label