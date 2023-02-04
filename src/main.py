import argparse
import json
import logging
import os
import random
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

from dataset1 import (DB, KB, MultiwozDataset, Pad, PrepareInput)
from model import MyModel, ParallelModelWrapper
from preprocess import Preprocess
from values1 import ALLOWED_DOMAINS, SPECIAL_TOKENS

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def GetIterator(l: list, batch_size: int) -> list:
    num_batches = len(l) // batch_size
    for i in range(num_batches):
        yield l[i * batch_size: (i + 1) * batch_size]

    if ((i + 1) * batch_size < len(l)):
        yield l[(i + 1) * batch_size:]


def Collate(inputs: list, tokenizer: AutoTokenizer):
    inp = [i[0] for i in inputs]
    domains = [i[1] for i in inputs]
    entity_names = [i[2] for i in inputs]

    padding = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["pad"])
    inp, input_attention_mask = Pad(inp, padding)

    inp = torch.tensor(inp)
    input_attention_mask = torch.tensor(input_attention_mask)

    return inp, input_attention_mask, domains, entity_names


def UpdateArgs(args: Namespace) -> Namespace:
    file_name = args.params_file
    if (args.eval_file or args.continue_training):
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


def UpdateTurnWithEntitySelectionExample(turn: dict,
                                         domains: list[str],
                                         entity_names: list[str],
                                         confidences: list[float],
                                         predicted_labels: list[int]):
    if ("entity_selection" not in turn):
        turn["entity_selection"] = {}

    z = zip(domains, entity_names, confidences, predicted_labels)
    for domain, entity_name, confidence, predicted_label in z:
        if (domain not in turn["entity_selection"]):
            turn["entity_selection"][domain] = {}

        turn["entity_selection"][domain][entity_name] = {
            "confidence": confidence,
            "predicted_label": predicted_label
        }

    return turn


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


def GetBestEntity(turn: dict, use_no_entity: bool = False) -> str:
    max_confidence = -1
    max_domain = None
    max_entity_name = None
    if (use_no_entity):
        max_domain = "no_entity"
        max_entity_name = "no_entity"

    for domain in turn["entity_selection"]:
        for entity_name, ent in turn["entity_selection"][domain].items():
            if ((use_no_entity and ent["confidence"] > max_confidence and ent["predicted_label"] == 1)
                    or (not use_no_entity and ent["confidence"] > max_confidence)):
                max_confidence = ent["confidence"]
                max_domain = domain
                max_entity_name = entity_name
    return max_domain, max_entity_name


def GetWorstEntity(turn: dict) -> str:
    min_confidence = np.inf
    min_domain = None
    min_entity_name = None
    for domain in turn["entity_selection"]:
        for entity_name, ent in turn["entity_selection"][domain].items():
            if (ent["confidence"] < min_confidence):
                min_confidence = ent["confidence"]
                min_domain = domain
                min_entity_name = entity_name
    return min_domain, min_entity_name


def Train(args: Namespace,
          train_dataset,
          eval_dataset,
          model: MyModel):
    train_iterator = iter(train_dataset)

    optimizer = AdamW(model.GetParameters(),
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_steps
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
    train_loss = 0
    iterator = trange(first_step, args.num_train_steps, desc="step")
    for step in iterator:
        model.train()
        batch = next(train_iterator)
        batch_ids = model.CollateFunction(batch)
        loss = model(batch_ids).sum() / batch_ids[0].shape[0]

        loss.backward()
        train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(
            model.GetParameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        iterator.set_postfix(loss=loss.item())

        if (step != 0 and step % args.save_every == 0):
            model_name = "checkpoint-" + str(step)
            model.SaveModel(args, model_name)
            SaveOptimizerAndScheduler(
                args, optimizer, scheduler, model_name)

            eval_loss = Evaluate(eval_dataset, model)

            logger.info("train loss = " + str(train_loss / args.save_every))
            logger.info("eval loss = " + str(eval_loss))
            train_loss = 0

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


def Evaluate(eval_dataset, model: MyModel) -> dict:
    with torch.no_grad():
        model.eval()

        total_loss = 0
        for batch in tqdm(eval_dataset):
            batch_ids = model.CollateFunction(batch)
            loss = model(batch_ids).sum() / batch_ids[0].shape[0]
            total_loss += loss.item()

    return total_loss / len(eval_dataset)


def Test(args: Namespace,
         data: dict,
         db: DB,
         kb: KB,
         model: MyModel) -> Tuple:
    SetSeed(args.seed)
    db.SetNoEntity(args.use_no_entity)
    kb.SetNoEntity(args.use_no_entity)

    batch_size = args.num_positive_examples_in_batch + \
        args.num_negative_examples_in_batch + \
        args.num_response_generation_examples_in_batch

    with torch.no_grad():
        model.eval()
        task_entity_selection_token_id = model.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["task_entity_selection"])
        task_response_generation_token_id = model.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["task_response_generation"])

        for dialog in tqdm(data["dialogues"]):
            for turn in dialog["items"]:
                if (turn["speaker"] != "system"):
                    continue

                if (not (turn["active_domain"] in ALLOWED_DOMAINS or turn["active_domain"] == None)):
                    continue

                inputs = []
                for domain, entity_name, _ in db:
                    inp = PrepareInput(
                        dialog["items"],
                        model.tokenizer,
                        db,
                        kb,
                        task_entity_selection_token_id,
                        turn["turn_number"],
                        args.num_turns,
                        domain,
                        entity_name,
                        model.tokenizer.max_model_input_sizes[args.model_name]
                    )
                    inputs.append((inp, domain, entity_name))
                inputs = GetIterator(inputs, batch_size)

                for inp in inputs:
                    inp, attn_mask, domains, entity_names = Collate(
                        inp, model.tokenizer)
                    predicted_labels, confidences = model.Classify(
                        inp, attn_mask)

                    UpdateTurnWithEntitySelectionExample(
                        turn,
                        domains,
                        entity_names,
                        confidences,
                        predicted_labels
                    )

                if (not args.select_only):
                    if (args.generate_using_worst):
                        domain, entity_name = GetWorstEntity(turn)
                    else:
                        domain, entity_name = GetBestEntity(
                            turn, args.use_no_entity)

                    inp = PrepareInput(
                        dialog["items"],
                        model.tokenizer,
                        db,
                        kb,
                        task_response_generation_token_id,
                        turn["turn_number"],
                        args.num_turns,
                        domain,
                        entity_name,
                        model.tokenizer.max_model_input_sizes[args.model_name]
                    )
                    inp = torch.tensor([inp]).cuda()

                    generated_response, _ = model.Generate(
                        args, inp, True)
                    turn["response_generation"] = {
                        "generated_response": generated_response,
                        "domain": domain,
                        "entity_name": entity_name
                    }

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str,
                        help="JSON configuration file")
    # eval_only = bool(eval_file)
    parser.add_argument("--eval_file", type=str,
                        help="JSON file to evaluate")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Saved checkpoint directory")
    parser.add_argument("--raw_data_path", type=str, help="Path to dataset.")
    parser.add_argument("--new_data_path", type=str, help="Path to dataset.")
    parser.add_argument("--preprocessed_data_path", type=str,
                        help="Path to preprocessed dataset.")
    parser.add_argument("--output_file", type=str, default="",
                        help="Predictions will be written to this file.")
    parser.add_argument("--model_path", type=str,
                        help="Name of the experiment, checkpoints will be stored here")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Only do preprocessing")
    parser.add_argument("--shuffle_KB", action="store_true",
                        help="shuffle KB docs")
    parser.add_argument("--select_only", action="store_true",
                        help="only run entity selection")
    parser.add_argument("--continue_training", action="store_true",
                        help="Continue training from specified checkpoint")
    parser.add_argument("--generate_using_worst", action="store_true",
                        help="Generate using worst entity")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="Number of GPUs for training")
    args = parser.parse_args()

    args = UpdateArgs(args)
    SetSeed(args.seed)

    Preprocess(args)
    if (args.preprocess_only):
        exit()

    my_model = MyModel(args).cuda()
    if (args.n_gpus > 1):
        my_model = ParallelModelWrapper(my_model)

    db = DB(args.preprocessed_data_path)
    kb = KB(args.preprocessed_data_path)

    if (args.eval_file):
        output_path = os.path.dirname(args.eval_file)
        os.makedirs(output_path, exist_ok=True)
        test_data = json.load(open(args.eval_file, "r"))

        test_data = Test(args, test_data, db, kb, my_model)
        json.dump(test_data, open(args.output_file, "w"), indent=4)
    else:
        train_dataset = MultiwozDataset(
            args,
            my_model.tokenizer,
            db,
            kb,
            os.path.join(args.preprocessed_data_path, "train.json"),
            True,
            args.num_positive_examples_in_batch,
            args.num_negative_examples_in_batch,
            args.num_response_generation_examples_in_batch
        )
        batch_size = args.num_positive_examples_in_batch + \
            args.num_negative_examples_in_batch + \
            args.num_response_generation_examples_in_batch
        val_dataset = MultiwozDataset(
            args,
            my_model.tokenizer,
            db,
            kb,
            os.path.join(args.preprocessed_data_path, "val.json"),
            False,
            batch_size,
            batch_size,
            batch_size
        )
        Train(args, train_dataset, val_dataset, my_model)


if (__name__ == "__main__"):
    main()
