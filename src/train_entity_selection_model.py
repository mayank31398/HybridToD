import argparse
import json
import logging
import os
import random
from argparse import Namespace

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import MultiwozEntitySelectionDataset, MultiwozKnowledge
from model import MyModel, ParallelModelWrapper
from preprocess import MultiwozPreprocessor
from values import ALL_DOMAINS

logger = logging.getLogger(__name__)


def UpdateArgs(args: Namespace) -> Namespace:
    file_name = args.params_file
    if (args.continue_training):
        file_name = os.path.join(
            args.model_path, args.checkpoint, "params.json")

    with open(file_name, "r") as f:
        params = json.load(f)
        args = vars(args)
        args.update(params)
        args = Namespace(**args)
    args.params = params  # used for saving checkpoints
    for key in vars(args):
        logger.info(str(key) + " = " + str(vars(args)[key]))

    return args


def SetSeed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def SaveOptimizerAndScheduler(args: Namespace, optimizer, scheduler, model_name) -> None:
    optimizer_scheduler = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(optimizer_scheduler, os.path.join(
        args.model_path, model_name, "optimizer_scheduler"))


def LoadOptimizerAndScheduler(args: Namespace, optimizer, scheduler) -> None:
    optimizer_scheduler = torch.load(os.path.join(
        args.model_path, args.checkpoint, "optimizer_scheduler"))
    optimizer.load_state_dict(optimizer_scheduler["optimizer"])
    scheduler.load_state_dict(optimizer_scheduler["scheduler"])
    return optimizer, scheduler


def Train(args: Namespace,
          train_dataset: MultiwozEntitySelectionDataset,
          eval_dataset: MultiwozEntitySelectionDataset,
          model: MyModel):
    train_iterator = iter(train_dataset)
    t_total = args.num_train_steps // args.gradient_accumulation_steps

    optimizer = AdamW(model.GetParameters(),
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    first_step = 0
    if (args.continue_training and args.model_path and args.checkpoint):
        optimizer, scheduler = LoadOptimizerAndScheduler(
            args, optimizer, scheduler)
        first_step = int(args.checkpoint.split("-")[1]) + 1

    best_loss = np.inf
    num_times_best_acc = 0
    best_found = False
    model.zero_grad()
    total_loss = 0
    iterator = trange(first_step, args.num_train_steps, desc="step")
    for step in iterator:
        model.train()
        batch = next(train_iterator)
        batch_ids = model.CollateFunction(batch)
        loss = model(batch_ids).sum() / batch_ids[0].shape[0]

        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        total_loss += loss.item()

        if ((step + 1) % args.gradient_accumulation_steps == 0):
            torch.nn.utils.clip_grad_norm_(
                model.GetParameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            iterator.set_postfix(Loss=total_loss / (step + 1))

        if (step != 0 and step % args.save_every == 0):
            model_name = "checkpoint-" + str(step)
            model.SaveModel(args, model_name)
            SaveOptimizerAndScheduler(
                args, optimizer, scheduler, model_name)

            eval_loss = Evaluate(eval_dataset, model)
            logger.info("eval loss = " + str(eval_loss))

            if (eval_loss < best_loss):
                num_times_best_acc = 0
                best_loss = eval_loss
                best_found = True

                model_name = "best"
                model.SaveModel(args, model_name)
                SaveOptimizerAndScheduler(
                    args, optimizer, scheduler, model_name)
            else:
                num_times_best_acc += 1
                if (num_times_best_acc == args.stopping_criteria):
                    break

    if (not best_found):
        model_name = "best"
        model.SaveModel(args, model_name)
        SaveOptimizerAndScheduler(args, optimizer, scheduler, model_name)


def Evaluate(eval_dataset: MultiwozEntitySelectionDataset, model: MyModel) -> dict:
    with torch.no_grad():
        model.eval()

        total_loss = 0
        for batch in tqdm(eval_dataset):
            batch_ids = model.CollateFunction(batch)
            loss = model(batch_ids).sum() / batch_ids[0].shape[0]
            total_loss += loss.item()

    return total_loss / len(eval_dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str,
                        help="JSON configuration file")
    parser.add_argument("--raw_data_path", type=str, help="Path to dataset.")
    parser.add_argument("--preprocessed_data_path", type=str,
                        help="Path to preprocessed dataset.")
    parser.add_argument("--modified_DB_path", type=str,
                        help="Path to modified DB.")
    parser.add_argument("--raw_DB_path", type=str,
                        help="Path to raw DB.")
    parser.add_argument("--model_path", type=str,
                        help="Name of the experiment, checkpoints will be stored here")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Only do preprocessing")
    parser.add_argument("--shuffle_KB", action="store_true",
                        help="shuffle KB docs")
    parser.add_argument("--continue_training", action="store_true",
                        help="Continue training from specified checkpoint")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="Number of GPUs for training")
    args = parser.parse_args()

    args.checkpoint = None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    args = UpdateArgs(args)
    SetSeed(args.seed)

    MultiwozPreprocessor().main(args)
    if (args.preprocess_only):
        exit()

    my_model = MyModel(args).cuda()
    if (args.n_gpus > 1):
        my_model = ParallelModelWrapper(my_model)
    multiwoz_knowledge = MultiwozKnowledge(args, my_model.tokenizer)

    train_dataset = MultiwozEntitySelectionDataset(
        args,
        my_model.tokenizer,
        multiwoz_knowledge,
        os.path.join(args.preprocessed_data_path, "data_train.json"),
        True,
        args.num_positive_examples_in_batch,
        args.num_negative_examples_in_batch
    )
    batch_size = args.num_positive_examples_in_batch + \
        args.num_negative_examples_in_batch
    val_dataset = MultiwozEntitySelectionDataset(
        args,
        my_model.tokenizer,
        multiwoz_knowledge,
        os.path.join(args.preprocessed_data_path, "data_val.json"),
        False,
        batch_size,
        batch_size
    )
    Train(args, train_dataset, val_dataset, my_model)


if (__name__ == "__main__"):
    main()
