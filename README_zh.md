简体中文 | [English](./README.md)

## Att-Induction: Attention-based Induction Networks for Few-Shot Text Classification

[![issues-open](https://img.shields.io/github/issues/ShaneTian/Att-Induction?color=success)](https://github.com/ShaneTian/Att-Induction/issues) [![issues-closed](https://img.shields.io/github/issues-closed/ShaneTian/Att-Induction?color=critical)](https://github.com/ShaneTian/Att-Induction/issues?q=is%3Aissue+is%3Aclosed) [![license](https://img.shields.io/github/license/ShaneTian/Att-Induction)](https://github.com/ShaneTian/Att-Induction/blob/master/LICENSE)

[Attention-based Induction Networks for Few-Shot Text Classification]() 的代码库。

## 目录

- [介绍](#介绍)
- [数据集](#数据集)
- [使用说明](#使用说明)
    - [依赖包](#依赖包)
    - [训练](#训练)
    - [测试](#测试)
- [维护者](#维护者)
- [引用](#引用)
- [使用许可](#使用许可)


## 介绍
基于注意力的归纳网络改进于[归纳网络](https://www.aclweb.org/anthology/D19-1403/)，是一个用于小样本文本分类的模型。

基于注意的归纳网络可以通过多头自注意机制对于不同查询学习得到不同的类别表示，其中归纳模块关注对于当前查询更加有效的样本和特征维度。此外，我们使用预训练模型代替从头训练一个编码器，使得在小样本场景下可以捕获更多的语义信息。实验结果表明，在三个公共数据集和一个真实数据集上，该模型取得了最好的分类效果。

## 数据集

- `ARSC`: 亚马逊评论情感分类。该数据集由 Yu 在 [Diverse few-shot text classiﬁcation with multiple metrics](https://www.aclweb.org/anthology/N18-1109/) 中提出。实验数据下载于 [DiverseFewShot_Amazon](https://github.com/Gorov/DiverseFewShot_Amazon)。 我们保持与 [Geng](https://www.aclweb.org/anthology/D19-1403/) 相同的设置。
- `HuffPost Headlines`: 该数据集发布于 kaggle -- [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset)。我们使用与 [Bao](https://github.com/YujiaBao/Distributional-Signatures) 相同的数据集子集，并在 `./src/utils.py` 中进行切分。
- `20 Newsgroups`: 该数据集最早由 [Lang](https://www.sciencedirect.com/science/article/pii/B9781558603776500487) 收集。实验数据下载于 [Distributional-Signatures](https://github.com/YujiaBao/Distributional-Signatures)。我们在 `./src/utils.py` 中进行切分。
- `Controversial Issues`: 该数据集由庭审过程中的争议焦点组成，是一个真实数据集。我们选择劳动争议 (Disp-L) 和产品责任纠纷 (Disp-PL) 案由构建该数据集。

## 使用说明
### 依赖包
使用 `pip install -r requirements.txt` 来安装以下依赖包：

- ![python-version](https://img.shields.io/badge/python-v3.7.5-blue)
- ![pytorch-version](https://img.shields.io/badge/pytorch-v1.3.1-blue)
- ![transformers-version](https://img.shields.io/badge/transformers-v2.6.0-blue)
- ![numpy-version](https://img.shields.io/badge/numpy-v1.17.4-blue)
- ![pandas-version](https://img.shields.io/badge/pandas-v0.25.3-blue)
- ![matplotlib-version](https://img.shields.io/badge/matplotlib-v3.1.3-blue)

### 训练

训练脚本位于 `./scripts/`。你只需修改某个 shell 文件中的一些参数，并在终端运行。例如：
```bash
bash ./scripts/run_train_HuffPost.sh
```

可以使用 `python3 train.py -h` 来查看所有可用的参数。

### 测试

事实上，如果你在训练时指定了 `--test_data` 参数，那么在训练完成后总是会执行一次测试。当然，你可以使用训练脚本指定 `--load_checkpoint` 和 `--only_test` 来执行一次独立的测试任务。

## 维护者

[@ShaneTian](https://github.com/ShaneTian).

## 引用

```
```

## 使用许可

[Apache License 2.0](LICENSE) © ShaneTian