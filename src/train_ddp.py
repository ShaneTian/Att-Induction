import argparse
import datetime
import logging
import os
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from data_loader import get_data_loader
from encoder_module import BERTEncoder, RoBERTaEncoder
from relation_module import NeuralTensorNetwork, BiLinear, Linear, PrototypeNetwork
from att_capsule import AttentionCapsuleNetwork
from capsule import CapsuleNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default=None, type=str,
                        help="File name of training data.")
    parser.add_argument("--val_data", default=None, type=str,
                        help="File name of validation data.")
    parser.add_argument("--test_data", default=None, type=str,
                        help="File name of testing data.")
    parser.add_argument("-N", default=2, type=int, help="N way")
    parser.add_argument("-K", default=5, type=int, help="K shot.")
    parser.add_argument("-Q", default=5, type=int,
                        help="Number of query instances per class.")
    parser.add_argument("--encoder", default="roberta-base", type=str,
                        choices=("bert-base", "roberta-base", "roberta-large", "xlnet"),
                        help="Ecoder: 'bert-base', 'roberta-base', 'roberta-large' or 'xlnet'.")
    parser.add_argument("--induction", default="att-capsule", type=str,
                        choices=("att-capsule", "capsule"),
                        help="Induction: 'att-capsule' or 'capsule'.")
    parser.add_argument("--relation", default="ntn", type=str,
                        choices=("ntn", "bilinear", "linear", "proto"),
                        help="Relation: 'ntn', 'bilinear', 'linear' or 'proto'.")
    parser.add_argument("--optim", default="adamw", type=str,
                        choices=("sgd", "adam", "adamw"),
                        help="Optimizer: 'sgd', 'adam' or 'adamw'.")
    parser.add_argument("--train_episodes", default=10000, type=int,
                        help="Number of training episodes. (train_episodes*=batch_size)")
    parser.add_argument("--val_episodes", default=1000, type=int,
                        help="Number of validation episodes. (val_episodes*=batch_size)")
    parser.add_argument("--val_steps", default=100, type=int,
                        help="Validate after x train_episodes.")
    parser.add_argument("--test_episodes", default=1000, type=int,
                        help="Number of testing episodes. test_episodes*=batch_size")
    parser.add_argument("--max_length", default=256, type=int,
                        help="Maximum length of sentences.")
    parser.add_argument("--hidden_size", default=768, type=int, help="Hidden size.")
    parser.add_argument("--induction_iters", default=5, type=int,
                        help="Number of iterations in capsule network.")
    parser.add_argument("--n_heads", default=None, type=int,
                        help="Number of heads in self-attention.")
    parser.add_argument("--dropout", default=None, type=float, help="Dropout rate.")
    parser.add_argument("-H", "--relation_size", default=100, type=int,
                        help="Size of neural tensor network.")
    parser.add_argument("-B", "--batch_size", default=2, type=int, help="Batch size.")
    parser.add_argument("--grad_steps", default=10, type=int,
                        help="Accumulate gradient update every x iterations.")
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate.")
    parser.add_argument("--warmup", default=0.06, type=float, help="Warmup ratio.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
    parser.add_argument("--pretrain_path", default="../resource/pretrain/",
                        type=str, help="Path to pretraind models.")
    parser.add_argument("--output_path", default="../log/", type=str, help="Save log and results.")
    parser.add_argument("--local_rank", type=int, help="For distributed training.")
    args = parser.parse_args()

    # Prefix of this running
    if args.n_heads is not None and args.dropout is not None:
        log_path = os.path.join(args.output_path, args.train_data,
                                args.encoder + f"-maxlen{args.max_length}-hidden{args.hidden_size}+"
                                + args.induction + f"-iters{args.induction_iters}-nheads{args.n_heads}"
                                f"-dropout{args.dropout}+" + args.relation + f"-H{args.relation_size}")
    else:
        log_path = os.path.join(args.output_path, args.train_data,
                                args.encoder + f"-maxlen{args.max_length}-hidden{args.hidden_size}+"
                                + args.induction + f"-iters{args.induction_iters}+"
                                + args.relation + f"-H{args.relation_size}")
    prefix = "-".join(str(datetime.datetime.now())[:-10].split()) + (f"+episodes{args.train_episodes}"
        f"-{args.val_episodes}-{args.test_episodes}+B{args.batch_size}+lr{args.lr}"
        f"+warmup{args.warmup}+weightdecay{args.weight_decay}")
    output_path = os.path.join(log_path, prefix)
    time.sleep(random.random())
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(filename=os.path.join(output_path, "run.log"),
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info("All parameters: {}".format(vars(args)))

    writer = SummaryWriter(log_dir=os.path.join(output_path, "tensorboard/"))

    if torch.cuda.is_available():
        logging.info("CUDA is available!")
        current_cuda = True
        current_device = torch.device("cuda")
        # For DDP
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    else:
        logging.warning("CUDA is not available. Using CPU!")
        current_cuda = False
        current_device = torch.device("cpu")
    
    if args.encoder == "roberta-base":
        encoder = RoBERTaEncoder("roberta-base", args.pretrain_path, args.max_length)
    elif args.encoder == "bert-base":
        encoder = BERTEncoder("bert-base-uncased", args.pretrain_path, args.max_length)
    else:
        raise NotImplementedError
    tokenizer = encoder.tokenize

    if args.relation == "ntn":
        relation = NeuralTensorNetwork(args.hidden_size, args.relation_size)
    elif args.relation == "bilinear":
        relation = BiLinear(args.hidden_size, args.relation_size)
    elif args.relation == "linear":
        relation = Linear(args.hidden_size, args.relation_size)
    elif args.relation == "proto":
        relation = PrototypeNetwork()
    else:
        raise NotImplementedError

    if args.induction == "att-capsule":
        model = AttentionCapsuleNetwork(
            encoder,
            relation,
            args.hidden_size,
            args.max_length,
            args.induction_iters,
            args.n_heads,
            args.dropout,
            current_device=current_device
        )
    elif args.induction == "capsule":
        model = CapsuleNetwork(
            encoder,
            relation,
            args.hidden_size,
            args.max_length,
            args.induction_iters,
            current_device=current_device
        )
    else:
        raise NotImplementedError

    if args.optim == "adamw":
        optimizer = AdamW
    elif args.optim == "adam":
        optimizer = torch.optim.Adam
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD
    else:
        raise NotImplementedError

    best_val_acc = train(args.train_data, args.val_data, tokenizer, model, args.batch_size,
                         args.N, args.K, args.Q, optimizer, args.train_episodes,
                         args.val_episodes, args.val_steps, args.grad_steps, args.lr,
                         args.warmup, args.weight_decay, writer, cuda=current_cuda,
                         local_rank=args.local_rank)
    logging.info("Best val mean acc: {:2.4f}".format(100 * best_val_acc))
    
    if args.test_data is not None:
        test_data_loader = get_data_loader("../data/", args.test_data, tokenizer,
                                           args.N, args.K, args.Q, args.batch_size, sampler=True)
        _, test_acc = eval(test_data_loader, model, args.batch_size, args.N, args.K,
                           args.Q, args.test_episodes, cuda=current_cuda)
        logging.info("Test mean acc: {:2.4f}".format(100 * test_acc))
        # with open(os.path.join(args.output_path, prefix, "test_results.txt")
    writer.flush()
    writer.close()


def loss_fn(predict_proba, label_one_hot):
    N = predict_proba.size(-1)
    return F.mse_loss(predict_proba.view(-1, N),
                      label_one_hot.view(-1, N).type(torch.float),
                      reduction="sum")


def mean_accuracy_fn(predict_label, label):
    return torch.mean((predict_label.view(-1) == label.view(-1)).type(torch.FloatTensor))


def train(train_data, val_data, tokenizer, model, B, N, K, Q, optimizer,
          train_episodes, val_episodes, val_steps, grad_steps, lr,
          warmup, weight_decay, writer, cuda=False, local_rank=None):
    train_data_loader = get_data_loader("../data/", train_data, tokenizer, N, K, Q, B, sampler=True)
    if val_data is not None:
        # Single batch
        val_data_loader = get_data_loader("../data/", val_data, tokenizer, N, K, Q, 1, sampler=True)

    parameter_list = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # Do not use weight decay.
    optimizer = optimizer(
        [
            {
                "params": [param for name, param in parameter_list
                if not any(nd in name for nd in no_decay)],
                "weight_decay": weight_decay
            },
            {
                "params": [param for name, param in parameter_list
                if any(nd in name for nd in no_decay)],
                "weight_decay": 0.0
            }
        ], lr=lr, correct_bias=False
    )

    # A schedule with a learning rate that decreases linearly after linearly
    # increasing during a warmup period.
    # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup * train_episodes),
        num_training_steps=train_episodes
    )

    if cuda:
        model.cuda()
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank,
                                                    find_unused_parameters=True)
    model.train()  # Set model to train mode.

    # Add model graph to tensorboard.
    dummy_support, dummy_support_mask, dummy_query, dummy_query_mask, _ = next(train_data_loader)
    if cuda:
        dummy_support = dummy_support.cuda()
        dummy_support_mask = dummy_support_mask.cuda()
        dummy_query = dummy_query.cuda()
        dummy_query_mask = dummy_query_mask.cuda()
    writer.add_graph(model, input_to_model=(dummy_support,
                                            dummy_support_mask,
                                            dummy_query,
                                            dummy_query_mask))

    total_samples = 0  # Count 'total' samples
    total_loss = 0.0
    total_acc_mean = 0.0
    best_val_acc = 0.0

    for episode in range(train_episodes):
        support, support_mask, query, query_mask, label = next(train_data_loader)
        if cuda:
            support = support.cuda()
            support_mask = support_mask.cuda()
            query = query.cuda()
            query_mask = query_mask.cuda()
            label = label.cuda()

        relation_score, _ = model(support, support_mask, query, query_mask)
        label_one_hot = F.one_hot(label.type(torch.long))
        loss = loss_fn(relation_score, label_one_hot) / grad_steps
        predict_label = relation_score.argmax(dim=-1, keepdims=False)
        acc_mean = mean_accuracy_fn(predict_label, label)

        loss.backward()
        
        # Log param and its gradient to tensorboard
        if episode % grad_steps == 0:
            for name, param in model.named_parameters():
                if param is None:
                    logging.warning("None value: {}".format(name))
                    continue
                if param.grad is None:
                    logging.warning("None grad: {}".format(name))
                    continue
                if param.requires_grad is False:
                    logging.warning("{}.requires_grad is False".format(name))
                name = name.replace('.', '/')
                writer.add_histogram(name, param.data.cpu().numpy(), (episode + 1) // grad_steps)
                writer.add_histogram(name + "/grad", param.grad.data.cpu().numpy(), (episode + 1) // grad_steps)

        if (episode + 1) % grad_steps == 0:
            # Log train loss and acc.
            writer.add_scalar("Loss/train", total_loss / total_samples, episode + 1)
            writer.add_scalar("Accuracy/train", total_acc_mean / total_samples, episode + 1)
            # Update params
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        total_acc_mean += acc_mean.item()
        total_samples += 1
        logging.info("[Train episode: {:6d}/{:6d}] ==> Loss: {:2.4f} Mean acc: {:2.4f}"
                     .format(episode + 1, train_episodes,
                             100 * total_loss / total_samples,
                             100 * total_acc_mean / total_samples))
        
        if val_data is not None and (episode + 1) % val_steps == 0:
            eval_loss, eval_acc = eval(val_data_loader, model, 1, N, K, Q, val_episodes, cuda=cuda)
            writer.add_scalar("Loss/val", eval_loss, episode + 1)
            writer.add_scalar("Accuracy/val", eval_acc, episode + 1)
            model.train()  # Reset model to train mode.
            if eval_acc > best_val_acc:
                # Save model
                logging.info("Best val mean acc: {:2.4f}".format(100 * eval_acc))
                # TODO
                # torch.save()
                best_val_acc = eval_acc
            total_samples = 0
            total_loss = 0.0
            total_acc_mean = 0.0
    return best_val_acc


def eval(data_loader, model, B, N, K, Q, episodes, cuda=False):
    model.eval()  # Set model to eval mode.
    total_loss = 0.0
    total_acc_mean = 0.0
    with torch.no_grad():
        for episode in range(episodes):
            support, support_mask, query, query_mask, label = next(data_loader)
            if cuda:
                support = support.cuda()
                support_mask = support_mask.cuda()
                query = query.cuda()
                query_mask = query_mask.cuda()
                label = label.cuda()

            relation_score, _ = model(support, support_mask, query, query_mask)
            label_one_hot = F.one_hot(label.type(torch.long))
            loss = loss_fn(relation_score, label_one_hot)
            predict_label = relation_score.argmax(dim=-1, keepdims=False)
            acc_mean = mean_accuracy_fn(predict_label, label)
            total_loss += loss.item()
            total_acc_mean += acc_mean.item()

            logging.info("[Val episode: {:5d}/{:5d}] ==> Loss: {:2.4f} Mean acc: {:2.4f}"
                         .format(episode + 1, episodes,
                                 100 * total_loss / (episode + 1),
                                 100 * total_acc_mean / (episode + 1)))
    return total_loss / episodes, total_acc_mean / episodes


if __name__ == "__main__":
    main()
