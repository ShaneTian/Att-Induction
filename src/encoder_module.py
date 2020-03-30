import os
import json
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer, XLNetModel, XLNetTokenizer, AlbertModel, AlbertTokenizer, RobertaModel, RobertaTokenizer


class AttBiLSTMEncoder(nn.Module):
    """Encoder indices of sentences by Self-Attention Bi-LSTM."""
    def __init__(self, pretrain_path, max_length, hidden_size, att_dim):
        super(AttBiLSTMEncoder, self).__init__()
        self.token2idx = json.load(open(os.path.join(
            pretrain_path,
            "att-bi-lstm",
            "token2idx.json"
        ), "r"))
        word_vec = torch.from_numpy(np.load(os.path.join(
            pretrain_path,
            "att-bi-lstm",
            "word_vec.npy"
        )))
        self.word_count, self.word_vec_dim = word_vec.shape[0], word_vec.shape[1]

        # Unknown, Blank
        self.unk_idx, self.blk_idx = self.word_count, self.word_count + 1
        unk = torch.randn(1, self.word_vec_dim, dtype=torch.double) / math.sqrt(self.word_vec_dim)
        blk = torch.zeros(1, self.word_vec_dim, dtype=torch.double)

        # Embedding layer
        self.word_embedding = nn.Embedding(self.word_count + 2, self.word_vec_dim, padding_idx=self.blk_idx)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec, unk, blk), 0))

        # Self-Att Bi-LSTM layer
        self.bilstm = nn.LSTM(self.word_vec_dim, hidden_size, batch_first=True, bidirectional=True)
        self.att1 = nn.Linear(2 * hidden_size, att_dim, bias=False)
        self.att2 = nn.Linear(att_dim, 1, bias=False)
        self.max_length = max_length

    def forward(self, tokens, lengths):
        """Self-Attention Bi-LSTM encoder forward.
        
        Args:
            tokens: torch.Tensor, [-1, max_length]
            lengths: torch.Tensor, [-1, max_length]
        
        Returns:
            sentence_embedding: torch.Tensor, [-1, 2 * hidden_size]"""
        embedding = self.word_embedding(tokens)  # [-1, max_length, word_vec_dim]
        lengths = lengths[:, 0]  # [-1]
        embedding = nn.utils.rnn.pack_padded_sequence(
            embedding,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        output, _ = self.bilstm(embedding)  # packed sequence. Throw away (h_n, c_n)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output,
            batch_first=True,
            padding_value=0.0,
            total_length=self.max_length
        )  # [-1, max_length, 2 * hidden_size]. Throw away lengths tensor
        att_score = torch.tanh(self.att1(output))  # [-1, max_length, att_dim]
        att_score = F.softmax(self.att2(att_score), dim=1)  # [-1, max_length, 1]
        output = torch.mul(att_score, output).sum(dim=1, keepdim=False)  # [-1, 2 * hidden_size]
        return output

    def tokenize(self, text):
        """Att-BiLSTM tokenizer.
        
        Args:
            text: str
            
        Returns:
            ids: list, [max_length]
            length: list, [max_length]"""
        ids = []
        for char in text.strip().split():
            char = char.lower()
            char_idx = self.token2idx[char] if char in self.token2idx else self.unk_idx
            ids.append(char_idx)
        # 'length' is the true length of ids, each element is equal true length
        # eg. text = "Hello World"  ==>  length = [2] * max_length
        length = self.max_length * [min(len(ids), self.max_length)]
        # Padding
        while len(ids) < self.max_length:
            ids.append(self.blk_idx)
        ids = ids[:self.max_length]
        return ids, length


class BERTEncoder(nn.Module):
    """Encoder indices of sentences in BERT last hidden states."""
    def __init__(self, model_shortcut_name, pretrain_path, max_length):
        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(
            model_shortcut_name,
            cache_dir=os.path.join(pretrain_path, model_shortcut_name)
        )
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            model_shortcut_name,
            cache_dir=os.path.join(pretrain_path, model_shortcut_name)
        )
        self.max_length = max_length
    
    def forward(self, tokens, mask):
        """BERT encoder forward.

        Args:
            tokens: torch.Tensor, [-1, max_length]
            mask: torch.Tensor, [-1, max_length]
            
        Returns:
            sentence_embedding: torch.Tensor, [-1, hidden_size]"""
        # last_hidden_state: [-1, max_length, hidden_size]
        last_hidden_state = self.bert(tokens, attention_mask=mask)
        return last_hidden_state[0][:, 0, :]  # The last hidden-state of <CLS>

    def tokenize(self, text):
        """BERT tokenizer.
        
        Args:
            text: str
        
        Returns:
            ids: list, [max_length]
            mask: list, [max_length]"""
        ids = self.bert_tokenizer.encode(text, add_special_tokens=True,
                                         max_length=self.max_length)
        # attention mask: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        mask = [1] * len(ids)
        # Padding
        while len(ids) < self.max_length:
            ids.append(0)
            mask.append(0)
        # truncation
        ids = ids[:self.max_length]
        mask = mask[:self.max_length]
        return ids, mask


class XLNetEncoder(nn.Module):
    """Encoder indices of sentences in XLNet last hidden states."""
    def __init__(self, model_shortcut_name, pretrain_path, max_length):
        super(XLNetEncoder, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(
            model_shortcut_name,
            cache_dir=os.path.join(pretrain_path, model_shortcut_name)
        )
        self.xlnet_tokenizer = XLNetTokenizer.from_pretrained(
            model_shortcut_name,
            cache_dir=os.path.join(pretrain_path, model_shortcut_name)
        )
        self.max_length = max_length
    
    def forward(self, tokens, mask):
        """XLNet encoder forward.

        Args:
            tokens: torch.Tensor, [-1, max_length]
            mask: torch.Tensor, [-1, max_length]
            
        Returns:
            sentence_embedding: torch.Tensor, [-1, hidden_size]"""
        # last_hidden_state: [-1, max_length, hidden_size]
        last_hidden_state = self.xlnet(tokens, attention_mask=mask)
        return last_hidden_state[0][:, -1, :]  # The last hidden-state of <CLS>

    def tokenize(self, text):
        ids = self.xlnet_tokenizer.encode(text, add_special_tokens=True,
                                          max_length=self.max_length)
        # attention mask: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        mask = [1] * len(ids)
        # Padding
        while len(ids) < self.max_length:
            ids.append(0)
            mask.append(0)
        # truncation
        ids = ids[:self.max_length]
        mask = mask[:self.max_length]
        return ids, mask


class ALBERTEncoder(nn.Module):
    """Encoder indices of sentences in ALBERT last hidden states."""
    def __init__(self, model_shortcut_name, pretrain_path, max_length):
        super(ALBERTEncoder, self).__init__()
        self.albert = AlbertModel.from_pretrained(
            model_shortcut_name,
            cache_dir=os.path.join(pretrain_path, model_shortcut_name)
        )
        self.albert_tokenizer = AlbertTokenizer.from_pretrained(
            model_shortcut_name,
            cache_dir=os.path.join(pretrain_path, model_shortcut_name)
        )
        self.max_length = max_length
    
    def forward(self, tokens, mask):
        """ALBERT encoder forward.

        Args:
            tokens: torch.Tensor, [-1, max_length]
            mask: torch.Tensor, [-1, max_length]
            
        Returns:
            sentence_embedding: torch.Tensor, [-1, hidden_size]"""
        # last_hidden_state: [-1, max_length, hidden_size]
        last_hidden_state = self.albert(tokens, attention_mask=mask)
        return last_hidden_state[0][:, 0, :]  # The last hidden-state of <CLS>

    def tokenize(self, text):
        ids = self.albert_tokenizer.encode(text, add_special_tokens=True,
                                           max_length=self.max_length)
        # attention mask: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        mask = [1] * len(ids)
        # Padding
        while len(ids) < self.max_length:
            ids.append(0)
            mask.append(0)
        # truncation
        ids = ids[:self.max_length]
        mask = mask[:self.max_length]
        return ids, mask


class RoBERTaEncoder(nn.Module):
    """Encoder indices of sentences in RoBERTa last hidden states."""
    def __init__(self, model_shortcut_name, pretrain_path, max_length):
        super(RoBERTaEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            model_shortcut_name,
            cache_dir=os.path.join(pretrain_path, model_shortcut_name)
        )
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(
            model_shortcut_name,
            cache_dir=os.path.join(pretrain_path, model_shortcut_name)
        )
        self.max_length = max_length
    
    def forward(self, tokens, mask):
        """RoBERTa encoder forward.

        Args:
            inputs: dict. {"tokens": [-1, max_length], "mask": [-1, max_length]}
            
        Returns:
            sentence_embedding: torch.Tensor. [-1, hidden_size]"""
        # last_hidden_state: [-1, max_length, hidden_size]
        last_hidden_state = self.roberta(tokens, attention_mask=mask)
        return last_hidden_state[0][:, 0, :]  # The last hidden-state of <CLS>

    def tokenize(self, text):
        ids = self.roberta_tokenizer.encode(text, add_special_tokens=True,
                                            max_length=self.max_length)
        # attention mask: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        mask = [1] * len(ids)
        # Padding
        while len(ids) < self.max_length:
            ids.append(0)
            mask.append(0)
        # truncation
        ids = ids[:self.max_length]
        mask = mask[:self.max_length]
        return ids, mask
