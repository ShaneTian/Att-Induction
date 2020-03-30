import argparse
import datetime
import logging
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from data_loader import get_ARSC_data_loader, get_general_data_loader
from encoder_module import AttBiLSTMEncoder, BERTEncoder, XLNetEncoder, ALBERTEncoder, RoBERTaEncoder
from relation_module import NeuralTensorNetwork, BiLinear, Linear, L2Distance, CosineDistance
from att_induction import AttentionInductionNetwork
from induction import InductionNetwork
from matching import MatchingNetwork
from prototype import PrototypeNetwork
from relation import RelationNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default=None, type=str,
                        help="File name of training data.")
    parser.add_argument("--val_data", default=None, type=str,
                        help="File name of validation data.")
    parser.add_argument("--test_data", action="append", type=str,
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
    # parser.add_argument("--relation", default="ntn", type=str,
    #                     choices=("ntn", "bilinear", "linear", "proto"),
    #                     help="Relation: 'ntn', 'bilinear', 'linear' or 'proto'.")
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
    parser.add_argument("--output_path", default="../log/", type=str, help="Save log and results.")
    parser.add_argument("--load_checkpoint", default=None, type=str, help="Path of checkpoint file.")
    parser.add_argument("--only_test", action="store_true")
    args = parser.parse_args()

    if "ARSC" in args.train_data:
        data_name = "ARSC"
    elif "20news" in args.train_data:
        data_name = "20news"
    elif "HuffPost" in args.train_data:
        data_name = "HuffPost"
    else:
        raise ValueError

    # Log config of this running
    if not args.only_test:
        log_path = os.path.join(args.output_path, args.train_data,
                                args.encoder + f"-maxlen{args.max_length}-hidden{args.hidden_size}+"
                                + args.model + f"-iters{args.induction_iters}-nheads{args.n_heads}"
                                f"-dropout{args.dropout}+" + f"-H{args.relation_size}")
        prefix = "-".join(str(datetime.datetime.now())[:-10].split()) + (f"+episodes{args.train_episodes}"
            f"+B{args.batch_size}+grad{args.grad_steps}+lr{args.lr}"
            f"+warmup{args.warmup}+weightdecay{args.weight_decay}")
        output_path = os.path.join(log_path, prefix)
        save_checkpoint = os.path.join(output_path, "checkpoint.pt")
        writer = SummaryWriter(log_dir=os.path.join(output_path, "tensorboard/"))
    else:
        output_path = os.path.join(os.path.dirname(args.load_checkpoint), "test",
                                   "-".join(str(datetime.datetime.now())[:-7].split()))
        writer = None
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(filename=os.path.join(output_path, "run.log"),
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logging.info("All parameters: {}".format(vars(args)))

    if torch.cuda.is_available():
        logging.info("CUDA is available!")
        current_cuda = True
        current_device = torch.device("cuda")
    else:
        logging.warning("CUDA is not available. Using CPU!")
        current_cuda = False
        current_device = torch.device("cpu")
    
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

    # if args.relation == "ntn":
    #     relation = NeuralTensorNetwork(args.hidden_size, args.relation_size)
    # elif args.relation == "bilinear":
    #     relation = BiLinear(args.hidden_size, args.relation_size)
    # elif args.relation == "linear":
    #     relation = Linear(args.hidden_size, args.relation_size)
    # elif args.relation == "proto":
    #     relation = PrototypeNetwork()
    # else:
    #     raise NotImplementedError

    if args.model == "att-induction":
        relation_module = NeuralTensorNetwork(args.hidden_size, args.relation_size)
        model = AttentionInductionNetwork(
            encoder,
            relation_module,
            args.hidden_size,
            args.max_length,
            args.induction_iters,
            args.n_heads,
            args.dropout,
            current_device=current_device
        )
    elif args.model == "induction":
        relation_module = BiLinear(args.hidden_size, args.relation_size)
        model = InductionNetwork(
            encoder,
            relation_module,
            args.hidden_size,
            args.max_length,
            args.induction_iters,
            current_device=current_device
        )
    elif args.model == "matching":
        relation_module = CosineDistance()
        model = MatchingNetwork(
            encoder,
            relation_module,
            args.hidden_size,
            args.max_length,
            current_device=current_device
        )
    elif args.model == "prototype":
        relation_module = L2Distance()
        model = PrototypeNetwork(
            encoder,
            relation_module,
            args.hidden_size,
            args.max_length,
            current_device=current_device
        )
    elif args.model == "relation":
        relation_module = Linear(args.hidden_size, args.relation_size)
        model = RelationNetwork(
            encoder,
            relation_module,
            args.hidden_size,
            args.max_length,
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

    if data_name == "ARSC":
        val_data = args.test_data
        test_data = args.test_data
    else:
        val_data = args.val_data
        test_data = args.test_data[0]

    if not args.only_test:
        best_val_acc = train(data_name, args.train_data, val_data, tokenizer, model, args.batch_size,
            args.N, args.K, args.Q, optimizer, args.train_episodes, args.val_episodes,
            args.val_steps, args.grad_steps, args.lr, args.warmup, args.weight_decay,
            writer, save_checkpoint, cuda=current_cuda, fp16=False)
        logging.info("Best val mean acc: {:2.4f}".format(best_val_acc))
        logging.info("Best model: {}".format(save_checkpoint))
    
    if test_data is not None:
        current_checkpoint = args.load_checkpoint if args.only_test else save_checkpoint
        if data_name == "ARSC":
            _, test_acc = eval_ARSC(test_data, tokenizer, model,
                                    1, args.N, args.K,
                                    args.Q, current_checkpoint, is_test=True,
                                    cuda=current_cuda)
        else:
            _, test_acc = eval(data_name, test_data, tokenizer, model,
                               1, args.N, args.K,
                               args.Q, args.test_episodes, current_checkpoint,
                               is_test=True, cuda=current_cuda)
        logging.info("Test mean acc: {:2.4f}".format(test_acc))


def train(data_name, train_data, val_data, tokenizer, model, B, N, K, Q, optimizer,
          train_episodes, val_episodes, val_steps, grad_steps, lr,
          warmup, weight_decay, writer, save_checkpoint, cuda=False, fp16=False):
    if data_name == "ARSC":
        train_data_loader = get_ARSC_data_loader("../data/ARSC/", train_data, tokenizer, N, K, Q, B)
    elif data_name == "20news":
        train_data_loader = get_general_data_loader("../data/20news/", train_data, tokenizer, N, K, Q, B)
    elif data_name == "HuffPost":
        train_data_loader = get_general_data_loader("../data/HuffPost/", train_data, tokenizer, N, K, Q, B)
    else:
        raise NotImplementedError

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
    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=int(warmup * train_episodes),
    #     num_training_steps=train_episodes,
    #     num_cycles=1.0
    # )

    if cuda:
        model.cuda()
    if fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
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

    for episode in range(1, train_episodes + 1):
        support, support_mask, query, query_mask, label = next(train_data_loader)
        if cuda:
            support = support.cuda()
            support_mask = support_mask.cuda()
            query = query.cuda()
            query_mask = query_mask.cuda()
            label = label.cuda()

        relation_score, predict_label = model(support, support_mask, query, query_mask)
        loss = model.loss(relation_score, label) / grad_steps
        acc_mean = model.mean_accuracy(predict_label, label)

        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        """
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
        """

        if episode % grad_steps == 0:
            # Log train loss and acc.
            writer.add_scalar("Loss/train", total_loss / max(1, total_samples), episode)
            writer.add_scalar("Accuracy/train", total_acc_mean / max(1, total_samples), episode)
            # Update params
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        total_acc_mean += acc_mean.item()
        total_samples += 1
        logging.info("[Train episode: {:6d}/{:6d}] ==> Loss: {:2.4f} Mean acc: {:2.4f}"
                     .format(episode, train_episodes,
                             100 * total_loss / total_samples,
                             100 * total_acc_mean / total_samples))
        
        if val_data is not None and episode % val_steps == 0:
            if data_name == "ARSC":
                eval_loss, eval_acc = eval_ARSC(val_data, tokenizer, model, 1, N, K, Q, cuda=cuda)
            else:
                eval_loss, eval_acc = eval(data_name, val_data, tokenizer, model, 1, N, K, Q, val_episodes, cuda=cuda)
            writer.add_scalar("Loss/val", eval_loss, episode)
            writer.add_scalar("Accuracy/val", eval_acc, episode)
            model.train()  # Reset model to train mode.
            if eval_acc > best_val_acc:
                # Save model
                logging.info("Best val mean acc: {:2.4f}".format(eval_acc))
                torch.save(model.state_dict(), save_checkpoint)
                logging.info("Saved model to {}".format(save_checkpoint))
                best_val_acc = eval_acc
            total_samples = 0
            total_loss = 0.0
            total_acc_mean = 0.0
    writer.flush()
    writer.close()
    return best_val_acc


def eval_ARSC(val_data, tokenizer, model, B, N, K, Q, load_checkpoint=None, is_test=False, cuda=False):
    if load_checkpoint is not None:
        if cuda:
            model.load_state_dict(torch.load(load_checkpoint))
        else:
            model.load_state_dict(torch.load(load_checkpoint, map_location=torch.device('cpu')))
        logging.info("Loaded model from {}".format(load_checkpoint))
        if cuda:
            model.cuda()
    model.eval()  # Set model to eval mode.

    results = {}  # All results of each dataset
    final_loss = final_acc = 0  # Sum of each dataset
    for each_data in val_data:
        # For each dataset
        if not is_test:
            data_loader = get_ARSC_data_loader("../data/ARSC/", each_data, tokenizer,
                                            N, K, Q, B, data_type="val")
            out_mark = "Val"
        else:
            data_loader = get_ARSC_data_loader("../data/ARSC/", each_data, tokenizer,
                                            N, K, Q, B, data_type="test")
            out_mark = "Test"
        total_loss = 0.0
        total_acc_mean = 0.0
        total_samples = 0
        with torch.no_grad():
            for episode, (support, support_mask, query, query_mask, label) in enumerate(data_loader):
                if cuda:
                    support = support.cuda()
                    support_mask = support_mask.cuda()
                    query = query.cuda()
                    query_mask = query_mask.cuda()
                    label = label.cuda()

                relation_score, predict_label = model(support, support_mask, query, query_mask)
                loss = model.loss(relation_score, label)
                acc_mean = model.mean_accuracy(predict_label, label)
                total_loss += loss.item()
                total_acc_mean += acc_mean.item()

                logging.info("[{} episode: {:3d}] ==> Loss: {:2.4f} Mean acc: {:2.4f}"
                            .format(out_mark, episode + 1,
                                    100 * total_loss / (episode + 1),
                                    100 * total_acc_mean / (episode + 1)))
                total_samples += 1
        loss_mean, acc_mean = total_loss / total_samples, total_acc_mean / total_samples
        results[each_data] = {"loss": 100 * loss_mean, "acc": 100 * acc_mean}
        logging.info("{} results of '{}': ==> Loss: {:2.4f} Mean acc: {:2.4f}"
                     .format(out_mark, each_data, 100 * loss_mean, 100 * acc_mean))
        final_loss += 100 * loss_mean
        final_acc += 100 * acc_mean
    logging.info("Final {} results: {}".format(out_mark, results))
    return final_loss / len(val_data), final_acc / len(val_data)


def eval(data_name, val_data, tokenizer, model, B, N, K, Q, val_episodes, load_checkpoint=None, is_test=False, cuda=False):
    if load_checkpoint is not None:
        if cuda:
            model.load_state_dict(torch.load(load_checkpoint))
        else:
            model.load_state_dict(torch.load(load_checkpoint, map_location=torch.device('cpu')))
        logging.info("Loaded model from {}".format(load_checkpoint))
        if cuda:
            model.cuda()
    model.eval()  # Set model to eval mode.

    if data_name == "20news":
        data_loader = get_general_data_loader("../data/20news/", val_data, tokenizer, N, K, Q, B)
    elif data_name == "HuffPost":
        data_loader = get_general_data_loader("../data/HuffPost/", val_data, tokenizer, N, K, Q, B)
    else:
        raise NotImplementedError

    out_mark = "Test" if is_test else "Val"

    total_loss = 0.0
    total_acc_mean = 0.0
    with torch.no_grad():
        for episode in range(1, val_episodes + 1):
            support, support_mask, query, query_mask, label = next(data_loader)
            if cuda:
                support = support.cuda()
                support_mask = support_mask.cuda()
                query = query.cuda()
                query_mask = query_mask.cuda()
                label = label.cuda()

            relation_score, predict_label = model(support, support_mask, query, query_mask)
            loss = model.loss(relation_score, label)
            acc_mean = model.mean_accuracy(predict_label, label)
            total_loss += loss.item()
            total_acc_mean += acc_mean.item()

            logging.info("[{} episode: {:5d}/{:5d}] ==> Loss: {:2.4f} Mean acc: {:2.4f}"
                        .format(out_mark, episode, val_episodes,
                                100 * total_loss / episode,
                                100 * total_acc_mean / episode))
    loss_mean, acc_mean = 100 * total_loss / val_episodes, 100 * total_acc_mean / val_episodes
    return loss_mean, acc_mean


if __name__ == "__main__":
    main()
