import os
import json
import numpy as np
import re


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

    """
    train_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['sci', 'rec']:
            train_classes.append(label_dict[key])

    val_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] in ['comp']:
            val_classes.append(label_dict[key])

    test_classes = []
    for key in label_dict.keys():
        if key[:key.find('.')] not in ['comp', 'sci', 'rec']:
            test_classes.append(label_dict[key])
    """
    # New split
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
    # [1, 5, 10, 11, 13, 14, 16, 18] [4, 6, 7, 12, 17] [0, 2, 3, 8, 9, 15, 19]
    # New: [3, 4, 5, 6, 7, 9, 10, 13, 16] [8, 11, 12, 14, 15] [0, 1, 2, 17, 18, 19]

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


news20_data_preprocessing("../data/data/20news.json", "../data/20news/")


def glove_preprocessing(word_vec_file, output_path):
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


def HuffPost_data_preprocessing(original_file, target_path):
    # Old split method!
    # train_classes = list(range(27))  # 0-26
    # val_classes = list(range(27, 33))  # 27-32
    # test_classes = list(range(33, 41))  # 33-40

    # New split method!
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