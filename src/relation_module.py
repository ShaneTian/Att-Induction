import torch
import torch.nn as nn
from torch.nn import functional as F


class NeuralTensorNetwork(nn.Module):
    def __init__(self, hidden_size, relation_size):
        super(NeuralTensorNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, relation_size, bias=False)
        self.linear = nn.Linear(2 * hidden_size, relation_size, bias=True)
        self.linear_out = nn.Linear(relation_size, 1, bias=True)
        self.__init_params()

    def __init_params(self):
        nn.init.kaiming_uniform_(self.bilinear.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.linear_out.weight, gain=nn.init.calculate_gain("sigmoid"))

    def forward(self, support, query):
        """
        Args:
            support: torch.Tensor, [B, totalQ, N, D]
            query: torch.Tensor, [B, totalQ, N, D]
            
        Returns:
            relation_score: torch.Tensor, [B, totalQ, N]"""
        relation_score = F.relu(
            self.bilinear(support, query) +
            self.linear(torch.cat((support, query), dim=-1))
        )
        relation_score = torch.sigmoid(self.linear_out(relation_score)).squeeze(-1)
        return relation_score


class BiLinear(nn.Module):
    def __init__(self, hidden_size, relation_size):
        super(BiLinear, self).__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, relation_size, bias=False)
        self.linear_out = nn.Linear(relation_size, 1, bias=True)
        self.__init_params()

    def __init_params(self):
        nn.init.kaiming_uniform_(self.bilinear.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.linear_out.weight, gain=nn.init.calculate_gain("sigmoid"))
    
    def forward(self, support, query):
        """
        Args:
            support: torch.Tensor, [B, totalQ, N, D]
            query: torch.Tensor, [B, totalQ, N, D]
            
        Returns:
            relation_score: torch.Tensor, [B, totalQ, N]"""
        relation_score = F.relu(self.bilinear(support, query))
        relation_score = torch.sigmoid(self.linear_out(relation_score)).squeeze(-1)
        return relation_score


class Linear(nn.Module):
    def __init__(self, hidden_size, relation_size):
        super(Linear, self).__init__()
        self.linear = nn.Linear(2 * hidden_size, relation_size, bias=True)
        self.linear_out = nn.Linear(relation_size, 1, bias=True)
        self.__init_params()

    def __init_params(self):
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.linear_out.weight, gain=nn.init.calculate_gain("sigmoid"))

    def forward(self, support, query):
        """
        Args:
            support: torch.Tensor, [B, totalQ, N, D]
            query: torch.Tensor, [B, totalQ, N, D]
            
        Returns:
            relation_score: torch.Tensor, [B, totalQ, N]"""
        relation_score = F.relu(self.linear(torch.cat((support, query), dim=-1)))
        relation_score = torch.sigmoid(self.linear_out(relation_score)).squeeze(-1)
        return relation_score


class L2Distance(nn.Module):
    def __init__(self):
        super(L2Distance, self).__init__()

    def forward(self, support, query):
        """
        Args:
            support: torch.Tensor, [B, totalQ, N, D]
            query: torch.Tensor, [B, totalQ, N, D]
            
        Returns:
            relation_score: torch.Tensor, [B, totalQ, N]"""
        l2_distance = torch.pow(support - query, 2).sum(-1, keepdim=False)  # [B, totalQ, N]
        return F.softmax(-l2_distance, dim=-1)


class CosineDistance(nn.Module):
    def __init__(self):
        super(CosineDistance, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def forward(self, support, query):
        """
        Args:
            support: torch.Tensor, [B, *, D]
            query: torch.Tensor, [B, *, D]
            
        Returns:
            relation_score: torch.Tensor, [B, *]"""
        cos_distance = self.cos_sim(support, query)  # [B, *]
        return F.softmax(cos_distance, dim=-1)
