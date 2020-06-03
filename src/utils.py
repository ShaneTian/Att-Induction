import os
import json
import numpy as np
import re
import pandas as pd
import random


def ARSC_data_preprocessing(original_path, target_path):
    all_classes = list(map(lambda x: x.strip(), open(os.path.join(original_path, "workspace.filtered.list"), "r").readlines()))
    test_classes = list(map(lambda x: x.strip(), open(os.path.join(original_path, "workspace.target.list"), "r").readlines()))
    train_data = dict()

    # 4*3 val/test data
    for test_class in test_classes:
        for level in ("t2", "t4", "t5"):
            class_name = test_class + "." + level  # 'books.t2'
            for data_type in ("train", "dev", "test"):
                test_data = dict()
                file_name = class_name + "." + data_type  # 'books.t2.train'
                for line in open(os.path.join(original_path, file_name), "r"):
                    line = line.strip()
                    line = line.split("\t")
                    content, label = line[0], line[1]
                    if label == "1" or label == "-1":
                        test_data.setdefault(label, []).append(content)
                    else:
                        raise ValueError
                if data_type == "train":
                    test_target_file = os.path.join(target_path, "{}.support.json".format(class_name))
                elif data_type == "dev":
                    test_target_file = os.path.join(target_path, "{}.val.json".format(class_name))
                elif data_type == "test":
                    test_target_file = os.path.join(target_path, "{}.test.json".format(class_name))
                json.dump(test_data, open(test_target_file, "w"))
    
    # 19*3 train data
    for train_class in [i for i in all_classes if i not in test_classes]:
        for level in ("t2", "t4", "t5"):
            for data_type in ("train", "dev", "test"):
                class_name = train_class + "." + level  # 'apparel.t2'
                if class_name not in train_data:
                    train_data[class_name] = dict()
                file_name = class_name + "." + data_type  # 'apparel.t2.train'
                for line in open(os.path.join(original_path, file_name), "r"):
                    line = line.strip()
                    line = line.split("\t")
                    content, label = line[0], line[1]
                    if label == "1" or label == "-1":
                        train_data[class_name].setdefault(label, []).append(content)
                    else:
                        raise ValueError

    json.dump(train_data, open(os.path.join(target_path, "ARSC_train.json"), "w"))


# ARSC_data_preprocessing("../others/DiverseFewShot_Amazon/Amazon_few_shot/", "../data/ARSC/")


def news20_data_preprocessing(original_file, target_path):
    # Get train/val/test class label list from https://github.com/YujiaBao/Distributional-Signatures/blob/master/src/dataset/loader.py
    label_dict = {
            'talk.politics.mideast': 0,
            'sci.space': 1,
            'misc.forsale': 2,
            'talk.politics.misc': 3,
            'comp.graphics': 4,
            'sci.crypt': 5,
            'comp.windows.x': 6,
            'comp.os.ms-windows.misc': 7,
            'talk.politics.guns': 8,
            'talk.religion.misc': 9,
            'rec.autos': 10,
            'sci.med': 11,
            'comp.sys.mac.hardware': 12,
            'sci.electronics': 13,
            'rec.sport.hockey': 14,
            'alt.atheism': 15,
            'rec.motorcycles': 16,
            'comp.sys.ibm.pc.hardware': 17,
            'rec.sport.baseball': 18,
            'soc.religion.christian': 19,
        }

    # Split
    train_classes, val_classes, test_classes = [], [], []
    for key in label_dict.keys():
        if key in [
            "comp.sys.ibm.pc.hardware",
            "rec.sport.baseball",
            "sci.space",
            "misc.forsale",
            "talk.politics.mideast",
            "soc.religion.christian"
        ]:
            test_classes.append(label_dict[key])
        elif key in [
            "comp.sys.mac.hardware",
            "rec.sport.hockey",
            "sci.med",
            "talk.politics.guns",
            "alt.atheism"
        ]:
            val_classes.append(label_dict[key])
        else:
            train_classes.append(label_dict[key])
    
    print(train_classes, val_classes, test_classes)
    # [3, 4, 5, 6, 7, 9, 10, 13, 16] [8, 11, 12, 14, 15] [0, 1, 2, 17, 18, 19]

    train_data, val_data, test_data = dict(), dict(), dict()
    with open(original_file, "r") as raw_file:
        line = raw_file.readline()
        while line:
            raw_line = json.loads(line)
            cur_text, cur_label = raw_line["raw"], raw_line["label"]
            # Remove "From: ... Subject:"
            cur_text = re.sub("^From(.+?)Subject:", "", cur_text)
            cur_text = cur_text.strip()
            if cur_label in train_classes:
                train_data.setdefault(cur_label, []).append(cur_text)
            elif cur_label in val_classes:
                val_data.setdefault(cur_label, []).append(cur_text)
            elif cur_label in test_classes:
                test_data.setdefault(cur_label, []).append(cur_text)
            else:
                raise ValueError
            line = raw_file.readline()
    
    json.dump(train_data, open(os.path.join(target_path, "20news_train_new.json"), "w"))
    json.dump(val_data, open(os.path.join(target_path, "20news_val_new.json"), "w"))
    json.dump(test_data, open(os.path.join(target_path, "20news_test_new.json"), "w"))


# news20_data_preprocessing("../data/data/20news.json", "../data/20news/")


def HuffPost_data_preprocessing(original_file, target_path):
    # Split
    split_dict = {"train": [[0, "POLITICS"], [1, "WELLNESS"], [2, "ENTERTAINMENT"], [3, "TRAVEL"], [4, "STYLE & BEAUTY"], [5, "PARENTING"], [7, "QUEER VOICES"], [9, "BUSINESS"], [11, "SPORTS"], [12, "BLACK VOICES"], [13, "HOME & LIVING"], [14, "PARENTS"], [15, "THE WORLDPOST"], [16, "WEDDINGS"], [17, "WOMEN"], [18, "IMPACT"], [20, "CRIME"], [21, "MEDIA"], [22, "WEIRD NEWS"], [23, "GREEN"], [25, "RELIGION"], [27, "SCIENCE"], [33, "FIFTY"], [34, "GOOD NEWS"], [35, "ARTS & CULTURE"], [37, "COLLEGE"], [38, "LATINO VOICES"]], "val": [[8, "FOOD & DRINK"], [10, "COMEDY"], [28, "WORLD NEWS"], [29, "TASTE"], [30, "TECH"], [32, "ARTS"]], "test": [[6, "HEALTHY LIVING"], [19, "DIVORCE"], [24, "WORLDPOST"], [26, "STYLE"], [31, "MONEY"], [36, "ENVIRONMENT"], [39, "CULTURE & ARTS"], [40, "EDUCATION"]]}
    train_classes = [idx_name[0] for idx_name in split_dict["train"]]
    val_classes = [idx_name[0] for idx_name in split_dict["val"]]
    test_classes = [idx_name[0] for idx_name in split_dict["test"]]

    train_data, val_data, test_data = dict(), dict(), dict()
    with open(original_file, "r") as raw_file:
        line = raw_file.readline()
        while line:
            data = json.loads(line.strip())
            data_sentence = " ".join(data["text"])
            data_label = data["label"]
            if data_label in train_classes:
                train_data.setdefault(data_label, []).append(data_sentence)
            elif data_label in val_classes:
                val_data.setdefault(data_label, []).append(data_sentence)
            elif data_label in test_classes:
                test_data.setdefault(data_label, []).append(data_sentence)
            else:
                raise ValueError
            line = raw_file.readline()
    json.dump(train_data, open(os.path.join(target_path, "HuffPost_train_new.json"), "w"))
    json.dump(val_data, open(os.path.join(target_path, "HuffPost_val_new.json"), "w"))
    json.dump(test_data, open(os.path.join(target_path, "HuffPost_test_new.json"), "w"))


# HuffPost_data_preprocessing("../data/data/huffpost.json", "../data/HuffPost/")


def glove_preprocessing(word_vec_file, output_path):
    """Transforming English word embedding txt into .npy embedding matrix and JSON index file."""
    token2idx = {}
    word_vec = []
    with open(word_vec_file, "r") as f:
        line = f.readline()
        index = 0
        while line:
            line = line.strip().split()
            token, vec = line[0], line[1:]
            vec = list(map(float, vec))
            if token in token2idx:
                raise ValueError("{} is existed!".format(token))
            else:
                token2idx[token] = index
            word_vec.append(vec)
            index += 1
            line = f.readline()
    word_vec = np.array(word_vec)
    assert len(token2idx) == np.shape(word_vec)[0], "Length is not same!"
    json.dump(token2idx, open(os.path.join(output_path, "token2idx.json"), "w"))
    np.save(os.path.join(output_path, "word_vec.npy"), word_vec)


# glove_preprocessing("../data/glove.6B/glove.6B.300d.txt", "../resource/pretrain/att-bi-lstm/")


def chinese_word_vec_preprocessing(word_vec_file, output_path):
    """Transforming Chinese word embedding txt into .npy embedding matrix and JSON index file."""
    token2idx = {}
    word_vec = []
    with open(word_vec_file, "r") as f:
        line = f.readline()  # 1292607 300
        line = f.readline()
        index = 0
        while line:
            line = line.rstrip().split(" ")
            # print(line)
            token, vec = line[0], line[1:]
            vec = list(map(float, vec))
            if token in token2idx:
                print("{} is existed!".format(token))
                line = f.readline()
                continue
            else:
                token2idx[token] = index
            word_vec.append(vec)
            index += 1
            line = f.readline()
            if index % 100000 == 0:
                print("{:d} done!".format(index))
    word_vec = np.array(word_vec)
    assert len(token2idx) == np.shape(word_vec)[0], "Length is not same!"
    json.dump(token2idx, open(os.path.join(output_path, "token2idx.json"), "w"))
    np.save(os.path.join(output_path, "word_vec.npy"), word_vec)


# chinese_word_vec_preprocessing("../data/chinese_word_vec/sgns.merge.word", "../resource/pretrain/att-bi-lstm-zh/")


"""
# real-world Controversial Issues dataset
def controversial_issues_data_preprocessing(original_file, target_path):
    data = pd.read_excel(original_file, usecols="A,C,E,F")
    data = data[(data["焦点类别"] == "G4") & (data["焦点组id"] != "delete") & (data["焦点组id"] != "other")]
    data = data[["焦点组id", "焦点内容"]]
    # print(data)
    length = data.shape[0]
    texts, labels = [], []
    for i in range(length):
        label, text = data.iloc[i][0], data.iloc[i][1]
        texts.append(text)
        labels.append(label)
    
    # Filter label
    label2idx = {}
    count_each_label = {}  # Number of samples each label
    for each_label in labels:
        count_each_label[each_label] = count_each_label.setdefault(each_label, 0) + 1
    texts_new, labels_new = [], []  # After filter
    for each_text, each_label in zip(texts, labels):
        if count_each_label[each_label] > 6:  # Threshold
            texts_new.append(each_text)
            labels_new.append(each_label)
            if each_label not in label2idx:
                label2idx[each_label] = len(label2idx)
    # print(label2idx)

    label_unique = list(set(labels_new))
    random.shuffle(label_unique)
    # print(label_unique, len(label_unique))
    # M6.17: Split 111 classes to 75 train, 16 val, 20 test
    # M9.30.349: Split 61 classes to 36 train, 11 val, 14 test
    train_classes, val_classes, test_classes = label_unique[:36], label_unique[36:47], label_unique[47:]

    train_data, val_data, test_data = dict(), dict(), dict()
    for each_text, each_label in zip(texts_new, labels_new):
        if each_label in train_classes:
            train_data.setdefault(label2idx[each_label], []).append(each_text)
        elif each_label in val_classes:
            val_data.setdefault(label2idx[each_label], []).append(each_text)
        elif each_label in test_classes:
            test_data.setdefault(label2idx[each_label], []).append(each_text)
        else:
            raise ValueError("Invalid label {}".format(each_label))
    
    json.dump(train_data, open(os.path.join(target_path, "M9.30.349_train.json"), "w"), ensure_ascii=False)
    json.dump(val_data, open(os.path.join(target_path, "M9.30.349_val.json"), "w"), ensure_ascii=False)
    json.dump(test_data, open(os.path.join(target_path, "M9.30.349_test.json"), "w"), ensure_ascii=False)


controversial_issues_data_preprocessing("../data/M9.30.349.xlsx", "../data/controversial_issues/")
"""