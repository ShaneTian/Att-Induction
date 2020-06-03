import os
import json

import numpy as np
import pprint
from matplotlib import pyplot as plt
from transformers import BertTokenizer


def count_data(path, is_ARSC=False):
    vocabs = set()
    tokens_len = {}  # {len1: count1, len2: count2, ...}
    examples_per_cls = {}  # {class1: count1, class2: count2, ...}

    # Tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        cache_dir=os.path.join("../resource/pretrain/", "bert-base-uncased")
    )

    print("Number of examples each class:")
    for root, _, files in os.walk(path):
        for file in files:
            if file[-5:] != ".json" or "support" in file:
                continue
            data = json.load(open(os.path.join(root, file), "r"))

            if is_ARSC:
                if "train" in file:
                    for each_class in data:
                        each_data = data[each_class]
                        examples_per_cls_cur = {}  # Current
                        # 1. Count examples each class
                        for class_label in each_data:
                            examples_per_cls_cur[each_class + "." + class_label] = len(each_data[class_label])
                            # 2. Count the number of tokens
                            for text in each_data[class_label]:
                                text = bert_tokenizer.tokenize(text)
                                tokens_len[len(text)] = tokens_len.setdefault(len(text), 0) + 1
                                # 3. Count vocabulary
                                for char in text:
                                    vocabs.add(char)
                        print(each_class)
                        pprint.pprint(examples_per_cls_cur, indent=4)
                        examples_per_cls.update(examples_per_cls_cur)
                else:
                    examples_per_cls_cur = {}  # Current file
                    # 1. Count examples each class
                    for class_label in data:
                        examples_per_cls_cur[file.replace("json", class_label)] = len(data[class_label])
                        # 2. Count the number of tokens
                        for text in data[class_label]:
                            text = bert_tokenizer.tokenize(text)
                            tokens_len[len(text)] = tokens_len.setdefault(len(text), 0) + 1
                            # 3. Count vocabulary
                            for char in text:
                                vocabs.add(char)
                    print(file)
                    pprint.pprint(examples_per_cls_cur, indent=4)
                    examples_per_cls.update(examples_per_cls_cur)
            
            else:
                examples_per_cls_cur = {}  # Current file
                # 1. Count examples each class
                for class_label in data:
                    examples_per_cls_cur[class_label] = len(data[class_label])
                    # 2. Count the number of tokens
                    for text in data[class_label]:
                        text = bert_tokenizer.tokenize(text)
                        tokens_len[len(text)] = tokens_len.setdefault(len(text), 0) + 1
                        # 3. Count vocabulary
                        for char in text:
                            vocabs.add(char)
                print(file)
                pprint.pprint(examples_per_cls_cur, indent=4)
                examples_per_cls.update(examples_per_cls_cur)
    print("\nNumber of tokens length each example:")
    pprint.pprint(tokens_len)

    print("=" * 20)
    print("# tokens/example: {:.2f}".format(sum([i * j for i, j in tokens_len.items()]) / sum(tokens_len.values())))
    print("# examples/class: {:.2f}".format(sum(examples_per_cls.values()) / len(examples_per_cls)))
    print("Vocabulary size:", len(vocabs))
    print("=" * 20)

    # 4. Plot the number of tokens length each example
    length_min, length_max = min(tokens_len.keys()), max(tokens_len.keys())
    length_range = [_len for _len in range(length_min, length_max + 1)]
    tokens_len_sum = sum(tokens_len.values())  # Number of examples
    tokens_len_accumulation = [tokens_len.get(_len, 0) / tokens_len_sum for _len in length_range]
    for i in range(1, len(tokens_len_accumulation)):
        tokens_len_accumulation[i] += tokens_len_accumulation[i - 1]  # Accumulation %
    
    plt.figure(figsize=(15, 10))
    x_axis_max = 80
    x_axis_period = 10
    # Background grid
    for i in range(1, 10):
        plt.hlines(i / 10, 0, x_axis_max, color="gray", linestyle="dashed", linewidths=0.1)
    for i in range(0, x_axis_max + 1, x_axis_period):
        plt.vlines(i, 0, 1, color="gray", linestyle="dashed", linewidths=0.1)
    # Bar
    plt.bar(length_range, tokens_len_accumulation, width=1.0)
    # Axis setup
    plt.xticks(range(0, x_axis_max + 1, x_axis_period))
    plt.xlim((0, x_axis_max))
    plt.ylim((0, 1))
    plt.xlabel("Number of tokens per example")
    plt.ylabel("Percentage")
    plt.title("Accumulated the number of tokens per example")
    plt.savefig("../log/HuffPost_accumulated_tokens_per_example.pdf")


count_data("../temp/")