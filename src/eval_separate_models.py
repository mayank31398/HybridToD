import argparse
import json
import logging
import os
import random
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset1 import DB, KB, Pad, PrepareInput
from model import MyModel
from values1 import ALLOWED_DOMAINS, SPECIAL_TOKENS

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


def SetSeed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def UpdateDataWithEntitySelectionExample(data: dict,
                                         example: dict,
                                         confidence: float,
                                         predicted_label: str,
                                         generated_explanation: str):
    turn = data[example["fname"]]["log"][example["turn_number"]]
    domain = example["domain"]
    entity_name = example["entity_name"]
    for ent in turn["entity_selection"]:
        if (ent["domain"] == domain and
                ent["entity_name"] == entity_name):
            if (confidence != None):
                ent["confidence"] = confidence
            ent["ground_truth_label"] = example["label"]
            ent["ground_truth_explanation"] = example["explanation"]
            if (predicted_label):
                ent["predicted_label"] = predicted_label
            if (generated_explanation):
                ent["generated_explanation"] = generated_explanation
            break


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


def Test(args: Namespace,
         data: dict,
         db: DB,
         kb: KB,
         entity_selection_model: MyModel,
         response_generation_model: MyModel) -> Tuple:
    SetSeed(args.seed)
    db.SetNoEntity(args.use_no_entity)
    kb.SetNoEntity(args.use_no_entity)

    batch_size = args.num_positive_examples_in_batch + \
        args.num_negative_examples_in_batch + \
        args.num_response_generation_examples_in_batch

    with torch.no_grad():
        entity_selection_model.eval()
        response_generation_model.eval()

        task_entity_selection_token_id = entity_selection_model.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["task_entity_selection"])
        task_response_generation_token_id = response_generation_model.tokenizer.convert_tokens_to_ids(
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
                        entity_selection_model.tokenizer,
                        db,
                        kb,
                        task_entity_selection_token_id,
                        turn["turn_number"],
                        args.num_turns,
                        domain,
                        entity_name,
                        entity_selection_model.tokenizer.max_model_input_sizes[args.model_name]
                    )
                    inputs.append((inp, domain, entity_name))
                inputs = GetIterator(inputs, batch_size)

                for inp in inputs:
                    inp, attn_mask, domains, entity_names = Collate(
                        inp, entity_selection_model.tokenizer)
                    predicted_labels, confidences = entity_selection_model.Classify(
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
                        response_generation_model.tokenizer,
                        db,
                        kb,
                        task_response_generation_token_id,
                        turn["turn_number"],
                        args.num_turns,
                        domain,
                        entity_name,
                        response_generation_model.tokenizer.max_model_input_sizes[args.model_name]
                    )
                    inp = torch.tensor([inp]).cuda()

                    generated_response, _ = response_generation_model.Generate(
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
    parser.add_argument("--eval_file", type=str,
                        help="JSON file to evaluate")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Saved checkpoint directory")
    parser.add_argument("--raw_data_path", type=str, help="Path to dataset.")
    parser.add_argument("--preprocessed_data_path", type=str,
                        help="Path to preprocessed dataset.")
    parser.add_argument("--modified_DB_path", type=str,
                        help="Path to modified DB.")
    parser.add_argument("--raw_DB_path", type=str,
                        help="Path to raw DB.")
    parser.add_argument("--output_file", type=str, default="",
                        help="Predictions will be written to this file.")
    parser.add_argument("--entity_selection_model_path", type=str,
                        help="Name of the experiment, checkpoints will be stored here")
    parser.add_argument("--response_generation_model_path", type=str,
                        help="Name of the experiment, checkpoints will be stored here")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Only do preprocessing")
    parser.add_argument("--shuffle_KB", action="store_true",
                        help="shuffle KB docs")
    parser.add_argument("--generate_using_worst", action="store_true",
                        help="Generate using worst entity")
    parser.add_argument("--select_only", action="store_true",
                        help="select only")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="Number of GPUs for training")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    args.model_path = args.entity_selection_model_path
    args = UpdateArgs(args)
    SetSeed(args.seed)

    entity_selection_model = MyModel(args).cuda()
    args.model_path = args.response_generation_model_path
    response_generation_model = MyModel(args).cuda()

    output_path = os.path.dirname(args.eval_file)
    os.makedirs(output_path, exist_ok=True)

    db = DB(args.preprocessed_data_path)
    kb = KB(args.preprocessed_data_path)

    test_data = json.load(open(args.eval_file, "r"))
    test_data = Test(args, test_data, db, kb, entity_selection_model, response_generation_model)
    json.dump(test_data, open(args.output_file, "w"), indent=4)


if (__name__ == "__main__"):
    main()
