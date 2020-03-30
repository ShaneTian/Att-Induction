import argparse
import datetime
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import ray
from ray import tune
from matplotlib import pyplot as plt

from data_loader import get_ARSC_data_loader, get_general_data_loader
from encoder_module import AttBiLSTMEncoder, BERTEncoder, XLNetEncoder, ALBERTEncoder, RoBERTaEncoder
from relation_module import NeuralTensorNetwork, BiLinear, Linear, L2Distance, CosineDistance
from att_induction import AttentionInductionNetwork
from induction import InductionNetwork
from relation import RelationNetwork
from prototype import PrototypeNetwork
from matching import MatchingNetwork


def main():
    args = args_parse()

    ray.init()
    tune_schedual = tune.schedulers.MedianStoppingRule(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        grace_period=10,  # Comparing period
        min_samples_required=5  # At least 5 trials for comparing
    )
    analysis = tune.run(
        TrainTune,
        name="{}-MSR".format(args.train_data),
        stop={
            "mean_accuracy": 0.8,
            "training_iteration": int(args.train_episodes / args.val_steps)
        },  # Stop after real train episodes
        scheduler=tune_schedual,
        config={
            "args": args,
            "lr": tune.sample_from(lambda spec: 1e-5 * np.random.uniform(1, 7)),
            "relation_size": tune.sample_from(lambda spec: np.random.choice([110, 130, 150])),
            "induction_iters": tune.sample_from(lambda spec: np.random.choice(range(2, 6))),
            "n_heads": tune.sample_from(lambda spec: np.random.choice([1, 2, 4, 8])),
            "dropout": tune.sample_from(lambda spec: np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5]))
        },
        num_samples=50,
        # num_samples=2,
        resources_per_trial={"cpu": 8, "gpu": 1},
        checkpoint_at_end=True,
        local_dir=args.output_path
    )

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

    # try:
    #     dfs = analysis.trial_dataframes
    #     # Plot by epoch
    #     ax = None  # This plots everything on the same plot
    #     for d in dfs.values():
    #         ax = d.mean_accuracy.plot(ax=ax, legend=False)
    #     plt.savefig(os.path.join(args.output_path, "MeanAcc.pdf"))
    # except:
    #     raise ValueError("Plot error.")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default=None, type=str,
                        help="File name of training data.")
    parser.add_argument("--val_data", default=None, type=str,
                        help="File name of validation data.")
    parser.add_argument("--test_data", default=None, type=str,
                        help="File names of testing data.")
    parser.add_argument("-N", default=2, type=int, help="N way")
    parser.add_argument("-K", default=5, type=int, help="K shot.")
    parser.add_argument("-Q", default=5, type=int,
                        help="Number of query instances per class.")
    parser.add_argument("--encoder", default="bert-base", type=str,
                        choices=("att-bi-lstm", "bert-base", "albert-base", "albert-large", "albert-xlarge", "xlnet-base", "roberta-base"),
                        help="Encoder: 'att-bi-lstm', 'bert-base', 'albert-base', 'albert-large', 'albert-xlarge', 'xlnet-base' or 'roberta-base'.")
    parser.add_argument("--model", default="att-induction", type=str,
                        choices=("att-induction", "induction", "matching", "prototype", "relation"),
                        help="Models: 'att-induction', 'induction', 'matching', 'prototype' or 'relation'.")
    parser.add_argument("--optim", default="adamw", type=str,
                        choices=("sgd", "adam", "adamw"),
                        help="Optimizer: 'sgd', 'adam' or 'adamw'.")
    parser.add_argument("--train_episodes", default=50000, type=int,
                        help="Number of training episodes. (train_episodes*=batch_size)")
    parser.add_argument("--val_episodes", default=1000, type=int,
                        help="Number of validation episodes. (val_episodes*=batch_size)")
    parser.add_argument("--val_steps", default=1000, type=int,
                        help="Validate after x train_episodes.")
    parser.add_argument("--test_episodes", default=1000, type=int,
                        help="Number of testing episodes. test_episodes*=batch_size")
    parser.add_argument("--max_length", default=512, type=int,
                        help="Maximum length of sentences.")
    parser.add_argument("--hidden_size", default=768, type=int, help="Hidden size.")
    parser.add_argument("--att_dim", default=None, type=int,
                        help="Attention dimension of Self-Attention Bi-LSTM encoder.")
    parser.add_argument("--induction_iters", default=None, type=int,
                        help="Number of iterations in capsule network.")
    parser.add_argument("--n_heads", default=None, type=int,
                        help="Number of heads in self-attention.")
    parser.add_argument("--dropout", default=None, type=float, help="Dropout rate.")
    parser.add_argument("-H", "--relation_size", default=100, type=int,
                        help="Size of neural tensor network.")
    parser.add_argument("-B", "--batch_size", default=1, type=int, help="Batch size.")
    parser.add_argument("--grad_steps", default=32, type=int,
                        help="Accumulate gradient update every x iterations.")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate.")
    parser.add_argument("--warmup", default=0.06, type=float, help="Warmup ratio.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
    parser.add_argument("--pretrain_path", default="../resource/pretrain/",
                        type=str, help="Path to pretraind models.")
    parser.add_argument("--output_path", default="../log_tune/", type=str, help="Save log and results.")
    # parser.add_argument("--load_checkpoint", default=None, type=str, help="Path of checkpoint file.")
    return parser.parse_args()


class TrainTune(tune.Trainable):
    def _setup(self, config):
        args = config.pop("args")
        vars(args).update(config)
        print(args)
        self.args = args

        # 1. Device steup
        if torch.cuda.is_available():
            self.current_cuda = True
            self.current_device = torch.device("cuda")
        else:
            self.current_cuda = False
            self.current_device = torch.device("cpu")
        
        # 2. Encoder setup
        if args.encoder == "att-bi-lstm":
            encoder = AttBiLSTMEncoder(args.pretrain_path, args.max_length, args.hidden_size, args.att_dim)
            args.hidden_size *= 2
        elif args.encoder == "bert-base":
            encoder = BERTEncoder("bert-base-uncased", args.pretrain_path, args.max_length)
        elif args.encoder == "albert-base":
            encoder = ALBERTEncoder("albert-base-v2", args.pretrain_path, args.max_length)
        elif args.encoder == "albert-large":
            encoder = ALBERTEncoder("albert-large-v2", args.pretrain_path, args.max_length)
        elif args.encoder == "albert-xlarge":
            encoder = ALBERTEncoder("albert-xlarge-v2", args.pretrain_path, args.max_length)
        elif args.encoder == "xlnet-base":
            encoder = XLNetEncoder("xlnet-base-cased", args.pretrain_path, args.max_length)
        elif args.encoder == "roberta-base":
            encoder = RoBERTaEncoder("roberta-base", args.pretrain_path, args.max_length)
        else:
            raise NotImplementedError
        tokenizer = encoder.tokenize

        # 3. Model steup
        if args.model == "att-induction":
            relation_module = NeuralTensorNetwork(args.hidden_size, args.relation_size)
            self.model = AttentionInductionNetwork(
                encoder,
                relation_module,
                args.hidden_size,
                args.max_length,
                args.induction_iters,
                args.n_heads,
                args.dropout,
                current_device=self.current_device
            )
        elif args.model == "induction":
            relation_module = BiLinear(args.hidden_size, args.relation_size)
            self.model = InductionNetwork(
                encoder,
                relation_module,
                args.hidden_size,
                args.max_length,
                args.induction_iters,
                current_device=self.current_device
            )
        elif args.model == "matching":
            relation_module = CosineDistance()
            self.model = MatchingNetwork(
                encoder,
                relation_module,
                args.hidden_size,
                args.max_length,
                current_device=self.current_device
            )
        elif args.model == "prototype":
            relation_module = L2Distance()
            self.model = PrototypeNetwork(
                encoder,
                relation_module,
                args.hidden_size,
                args.max_length,
                current_device=self.current_device
            )
        elif args.model == "relation":
            relation_module = Linear(args.hidden_size, args.relation_size)
            self.model = RelationNetwork(
                encoder,
                relation_module,
                args.hidden_size,
                args.max_length,
                current_device=self.current_device
            )
        else:
            raise NotImplementedError

        # 4. Optimizer setup                
        parameter_list = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # Do not use weight decay.
        parameter_optim = [
            {
                "params": [param for name, param in parameter_list
                if not any(nd in name for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [param for name, param in parameter_list
                if any(nd in name for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        if args.optim == "adamw":
            self.optimizer = AdamW(parameter_optim, lr=args.lr, correct_bias=False)
        elif args.optim == "adam":
            self.optimizer = torch.optim.Adam(parameter_optim, lr=args.lr)
        elif args.optim == "sgd":
            self.optimizer = torch.optim.SGD(parameter_optim, lr=args.lr)
        else:
            raise NotImplementedError

        # A schedule with a learning rate that decreases linearly after linearly
        # increasing during a warmup period.
        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(args.warmup * args.train_episodes),
            num_training_steps=args.train_episodes
        )
        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=int(warmup * train_episodes),
        #     num_training_steps=train_episodes,
        #     num_cycles=1.0
        # )

        # 6. Dataloader setup
        if "20news" in args.train_data:
            # 20 News. dataset
            self.train_data_loader = get_general_data_loader(
                "/home/tx/att_induction/data/20news/", args.train_data, tokenizer,
                args.N, args.K, args.Q, args.batch_size
            )
            self.test_data_loader = get_general_data_loader(
                "/home/tx/att_induction/data/20news/", args.test_data, tokenizer,
                args.N, args.K, args.Q, 1
            )
        elif "HuffPost" in args.train_data:
            # HuffPost dataset
            self.train_data_loader = get_general_data_loader(
                "/home/tx/att_induction/data/HuffPost/", args.train_data, tokenizer,
                args.N, args.K, args.Q, args.batch_size
            )
            self.test_data_loader = get_general_data_loader(
                "/home/tx/att_induction/data/HuffPost/", args.test_data, tokenizer,
                args.N, args.K, args.Q, 1
            )
        else:
            raise NotImplementedError

        if self.current_cuda:
            self.model.cuda()

    def _train(self):
        self.model.train()
        for episode in range(self.args.val_steps):
            support, support_mask, query, query_mask, label = next(self.train_data_loader)
            if self.current_cuda:
                support = support.cuda()
                support_mask = support_mask.cuda()
                query = query.cuda()
                query_mask = query_mask.cuda()
                label = label.cuda()

            relation_score, predict_label = self.model(support, support_mask, query, query_mask)
            loss = self.model.loss(relation_score, label) / self.args.grad_steps
            # acc_mean = self.model.mean_accuracy(predict_label, label)
            loss.backward()

            if episode % self.args.grad_steps == 0:
                # Update params
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        return self._test()


    def _test(self):
        self.model.eval()
        total_loss = 0.0
        total_acc_mean = 0.0
        with torch.no_grad():
            for _ in range(1, self.args.test_episodes + 1):
                support, support_mask, query, query_mask, label = next(self.test_data_loader)
                if self.current_cuda:
                    support = support.cuda()
                    support_mask = support_mask.cuda()
                    query = query.cuda()
                    query_mask = query_mask.cuda()
                    label = label.cuda()

                relation_score, predict_label = self.model(support, support_mask, query, query_mask)
                loss = self.model.loss(relation_score, label)
                acc_mean = self.model.mean_accuracy(predict_label, label)
                total_loss += loss.item()
                total_acc_mean += acc_mean.item()

        loss_mean, acc_mean = total_loss / self.args.test_episodes, total_acc_mean / self.args.test_episodes
        return {"mean_loss": loss_mean, "mean_accuracy": acc_mean}

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


if __name__ == "__main__":
    main()
