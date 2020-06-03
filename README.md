English | [简体中文](./README_zh.md)

## Att-Induction: Attention-based Induction Networks for Few-Shot Text Classification

[![issues-open](https://img.shields.io/github/issues/ShaneTian/Att-Induction?color=success)](https://github.com/ShaneTian/Att-Induction/issues) [![issues-closed](https://img.shields.io/github/issues-closed/ShaneTian/Att-Induction?color=critical)](https://github.com/ShaneTian/Att-Induction/issues?q=is%3Aissue+is%3Aclosed) [![license](https://img.shields.io/github/license/ShaneTian/Att-Induction)](https://github.com/ShaneTian/Att-Induction/blob/master/LICENSE)

Code for paper [Attention-based Induction Networks for Few-Shot Text Classification]().

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Usage](#usage)
    - [Requirements](#requirements)
    - [Training](#training)
    - [Test](#test)
- [Maintainers](#maintainers)
- [Citation](#citation)
- [License](#license)


## Introduction
Attention-based Induction Networks is a model for few-shot text classification, which continues the work of [Induction Networks](https://www.aclweb.org/anthology/D19-1403/).

Attention-based Induction Networks can learn different class representations for diverse queries by the multi-head self-attention, in which induction module pays more attention to effective instances and feature dimensions for current query. In addition, we use the pre-trained model instead of training an encoder from scratch, which can capture more semantic information in the few-shot learning scenarios. Experiment results show that, on three public datasets and a real-world dataset, this model signiﬁcantly outperforms the existing state-of-the-art approaches.

## Datasets

- `ARSC`: Amazon Review Sentiment Classiﬁcation. This dataset is proposed by Yu in the NAACL 2018 paper [Diverse few-shot text classiﬁcation with multiple metrics](https://www.aclweb.org/anthology/N18-1109/). The dataset is downloaded from [DiverseFewShot_Amazon](https://github.com/Gorov/DiverseFewShot_Amazon). We use the same settings as [Geng](https://www.aclweb.org/anthology/D19-1403/).
- `HuffPost Headlines`: This dataset is published in kaggle -- [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset). We use a subset of the entire dataset following [Bao et al](https://github.com/YujiaBao/Distributional-Signatures). We split it in the `./src/utils.py`.
- `20 Newsgroups`: This dataset was originally collected by [Lang](https://www.sciencedirect.com/science/article/pii/B9781558603776500487). The dataset is downloaded from [Distributional-Signatures](https://github.com/YujiaBao/Distributional-Signatures). We split it in the `./src/utils.py`.
- `Controversial Issues`: This dataset consists of controversial issues during the trial. It is a real-world dataset. We create this dataset by choosing Labour Disputes (Disp-L) and Product Liability Disputes (Disp-PL).

## Usage
### Requirements
You can use `pip install -r requirements.txt` to install the following dependent packages:

- ![python-version](https://img.shields.io/badge/python-v3.7.5-blue)
- ![pytorch-version](https://img.shields.io/badge/pytorch-v1.3.1-blue)
- ![transformers-version](https://img.shields.io/badge/transformers-v2.6.0-blue)
- ![numpy-version](https://img.shields.io/badge/numpy-v1.17.4-blue)
- ![pandas-version](https://img.shields.io/badge/pandas-v0.25.3-blue)
- ![matplotlib-version](https://img.shields.io/badge/matplotlib-v3.1.3-blue)

### Training

Training scripts are placed in `./scripts/`. You only need to modify some training parameters in a shell file, and then run it on the terminal. For example:
```bash
bash ./scripts/run_train_HuffPost.sh
```

You can use `python3 train.py -h` to see all available parameters.

### Test

In fact, if the `--test_data` is given in the training, the test task will be always performed after training. Of course, you can perform a separate test task by specifying `--load_checkpoint` and `--only_test` in the training script.

## Maintainers

[@ShaneTian](https://github.com/ShaneTian).

## Citation

```
```

## License

[Apache License 2.0](LICENSE) © ShaneTian