import os
import json
import logging
import random
import math

import torch
import torch.utils.data as data


class ARSCTrainDataset(data.Dataset):
    """
    Returns:
        support: torch.Tensor, [N, K, max_length]
        support_mask: torch.Tensor, [N, K, max_length]
        query: torch.Tensor, [totalQ, max_length]
        query_mask: torch.Tensor, [totalQ, max_length]
        label: torch.Tensor, [totalQ]"""
    def __init__(self, path, name, tokenizer, N, K, Q):
        file_path = os.path.join(path, name + ".json")
        if not os.path.exists(file_path):
            raise Exception("File {} does not exist.".format(file_path))
        self.data = json.load(open(file_path, "r"))
        self.classes = list(self.data.keys())
        self.N, self.K, self.Q = N, K, Q
        assert self.N == 2
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        target_class = random.choice(self.classes)  # Sample 1 class name
        target_data = self.data[target_class]

        support, support_mask = [], []
        query, query_mask = [], []
        query_label = []

        # Min number of samples in target_data.
        data_count = min(list(map(lambda x: len(x), target_data.values())))

        for class_idx, class_name in enumerate(target_data.keys()):
            # Add [] for each class
            support.append([])
            support_mask.append([])

            if self.K + self.Q > data_count:
                # Split total data to support and query when data is less than requirement.
                current_Q = data_count - self.K
                assert current_Q > 0, "Query set must not be empty!"
                logging.warning("Sampling is out of range: {ca} of {tc}. (Number of total samples: {dc:d})."
                                " Final Q: {q:d}"
                                .format(ca=class_name, tc=target_class, dc=data_count,
                                        q=current_Q))
            else:
                current_Q = self.Q
            samples = random.sample(target_data[class_name], self.K + current_Q)
            
            for idx, sample in enumerate(samples):
                # Tokenize. Senquences to indices.
                indices, mask = self.tokenizer(sample)

                if idx < self.K:
                    support[class_idx].append(indices)
                    support_mask[class_idx].append(mask)
                else:
                    query.append(indices)
                    query_mask.append(mask)
            query_label += [class_idx] * current_Q
        # print("S:", support)
        # print("SM:", support_mask)
        # print("Q:", query)
        # print("QM:", query_mask)
        # print("L:", query_label)
        return (torch.tensor(support, dtype=torch.long),
                torch.tensor(support_mask, dtype=torch.long),
                torch.tensor(query, dtype=torch.long),
                torch.tensor(query_mask, dtype=torch.long),
                torch.tensor(query_label, dtype=torch.long))

    def __len__(self):
        return 100000000


class ARSCValDataset(data.Dataset):
    """Fixed support set.
    Returns:
        support: torch.Tensor, [N, K, max_length]
        support_mask: torch.Tensor, [N, K, max_length]
        query: torch.Tensor, [totalQ, max_length]
        query_mask: torch.Tensor, [totalQ, max_length]
        label: torch.Tensor, [totalQ]"""
    def __init__(self, path, name, tokenizer, N, K, Q, is_test=False):
        support_file_path = os.path.join(path, name + ".support.json")
        if not os.path.exists(support_file_path):
            raise Exception("Support file {} does not exist".format(support_file_path))
        if not is_test:
            # Val data
            val_file_path = os.path.join(path, name + ".val.json")
        else:
            # Test data
            val_file_path = os.path.join(path, name + ".test.json")
        if not os.path.exists(val_file_path):
            raise Exception("Val/Test file {} does not exist.".format(val_file_path))
        self.support_data = json.load(open(support_file_path, "r"))
        self.classes = self.support_data.keys()  # Fixed class label ("-1", "1").
        self.N, self.K, self.Q = N, K, Q
        assert self.N == 2, "Must be 2-way!"
        self.tokenizer = tokenizer
        self.support, self.support_mask = self.__get_support()  # Get fixed support set.

        # Read query data and label from json.
        query_data = json.load(open(val_file_path, "r"))
        self.query_data = []
        self.label_data = []
        for class_idx, class_name in enumerate(self.classes):
            self.query_data += query_data[class_name]
            self.label_data += [class_idx] * len(query_data[class_name])

    def __get_support(self):
        support, support_mask = [], []
        for class_idx, class_name in enumerate(self.classes):
            support.append([])
            support_mask.append([])
            for sample in self.support_data[class_name][:self.K]:
                indices, mask = self.tokenizer(sample)
                support[class_idx].append(indices)
                support_mask[class_idx].append(mask)
        return torch.tensor(support, dtype=torch.long), torch.tensor(support_mask, dtype=torch.long)

    def __getitem__(self, index):
        query, query_mask = [], []

        samples = self.query_data[index * self.Q:(index + 1) * self.Q]
        query_label = self.label_data[index * self.Q:(index + 1) * self.Q]
        for sample in samples:
            # Tokenize. Senquences to indices.
            indices, mask = self.tokenizer(sample)
            query.append(indices)
            query_mask.append(mask)
        return (self.support, self.support_mask,
                torch.tensor(query, dtype=torch.long),
                torch.tensor(query_mask, dtype=torch.long),
                torch.tensor(query_label, dtype=torch.long))

    def __len__(self):
        return math.ceil(len(self.query_data) / self.Q)


def get_ARSC_data_loader(path, name, tokenizer, N, K, Q, batch_size,
                         data_type="train", num_workers=4, sampler=False):
    if data_type == "train":
        dataset = ARSCTrainDataset(path, name, tokenizer, N, K, Q)
    elif data_type == "val":
        dataset = ARSCValDataset(path, name, tokenizer, N, K, Q, is_test=False)
    elif data_type == "test":
        dataset = ARSCValDataset(path, name, tokenizer, N, K, Q, is_test=True)
    else:
        raise NotImplementedError("Unknown data_type: %s" % data_type)
    
    if sampler:
        sampler = data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # collate_fn=collate_fn,
        sampler=sampler
    )
    return iter(data_loader)


class GeneralDataset(data.Dataset):
    """
    Returns:
        support: torch.Tensor, [N, K, max_length]
        support_mask: torch.Tensor, [N, K, max_length]
        query: torch.Tensor, [totalQ, max_length]
        query_mask: torch.Tensor, [totalQ, max_length]
        label: torch.Tensor, [totalQ]"""
    def __init__(self, path, name, tokenizer, N, K, Q):
        file_path = os.path.join(path, name + ".json")
        if not os.path.exists(file_path):
            raise Exception("File {} does not exist.".format(file_path))
        self.data = json.load(open(file_path, "r"))
        self.classes = list(self.data.keys())
        self.N, self.K, self.Q = N, K, Q
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        support, support_mask = [], []
        query, query_mask = [], []
        query_label = []

        target_classes = random.sample(self.classes, self.N)  # Sample N class name

        for class_idx, class_name in enumerate(target_classes):
            # Add [] for each class
            support.append([])
            support_mask.append([])

            samples = random.sample(self.data[class_name], self.K + self.Q)
            for idx, sample in enumerate(samples):
                # Tokenize. Senquences to indices.
                indices, mask = self.tokenizer(sample)

                if idx < self.K:
                    support[class_idx].append(indices)
                    support_mask[class_idx].append(mask)
                else:
                    query.append(indices)
                    query_mask.append(mask)
            query_label += [class_idx] * self.Q
        return (torch.tensor(support, dtype=torch.long),
                torch.tensor(support_mask, dtype=torch.long),
                torch.tensor(query, dtype=torch.long),
                torch.tensor(query_mask, dtype=torch.long),
                torch.tensor(query_label, dtype=torch.long))

    def __len__(self):
        return 100000000


def get_general_data_loader(path, name, tokenizer, N, K, Q, batch_size,
                            num_workers=4, sampler=False):
    dataset = GeneralDataset(path, name, tokenizer, N, K, Q)
    if sampler:
        sampler = data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler
    )
    return iter(data_loader)
