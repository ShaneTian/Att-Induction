import os
import json
import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer


class AttBiLSTMEncoder(nn.Module):
    """Encoder indices of sentences by Self-Attention Bi-LSTM."""
    def __init__(self, pretrain_path, max_length, hidden_size, att_dim, is_zh=False):
        super(AttBiLSTMEncoder, self).__init__()
        self.is_zh = is_zh
        if not self.is_zh:
            # English
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
        else:
            # Chinese
            self.token2idx = json.load(open(os.path.join(
                pretrain_path,
                "att-bi-lstm-zh",
                "token2idx.json"
            ), "r"))
            word_vec = torch.from_numpy(np.load(os.path.join(
                pretrain_path,
                "att-bi-lstm-zh",
                "word_vec.npy"
            )))
            import pkuseg
            self.seg = pkuseg.pkuseg()
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
        if not self.is_zh:
            # English
            for char in text.strip().split():
                char = char.lower()
                char_idx = self.token2idx[char] if char in self.token2idx else self.unk_idx
                ids.append(char_idx)
        else:
            # Chinese
            for token in self.seg.cut(text.strip()):
                token_idx = self.token2idx[token] if token in self.token2idx else self.unk_idx
                ids.append(token_idx)

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
