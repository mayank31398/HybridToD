import argparse
from distutils.log import error
import json
import logging
import os
import re
import sys
import zipfile
from typing import List, Tuple

from fuzzywuzzy import fuzz
from nltk.tokenize import word_tokenize as tknz
from tqdm import tqdm

from scorer1 import (ConvertDSVtoSV, MakeFractionalString,
                    Metrics)
from values1 import (ALLOWED_DOMAINS, BOOLEAN_STRINGS, NAME_MAP,
                    REQUESTABLE_SLOTS, SORTED_KEYS, UNIQUE_SLOTS)
from dataset1 import DB, DSV


sys.path.append("SeKnow-PLM/scripts")

from preprocess_convlab_multiwoz import Database, clear_whitespaces

DSEV_PARSED = {}

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def HackReplacements(replacement: list) -> list:
    if (replacement[0] == "price range"):
        replacement = ["pricerange", replacement[1]]
    if (replacement == ["address", "Huntingdon Marriott Hotel"]):
        replacement = ["name", "Huntingdon Marriott Hotel"]
    elif (replacement == ["address", "01223332400"]):
        replacement = ["phone", "01223332400"]
    elif (replacement == ["postcode", "5 Greens Road"]):
        replacement = ["address", "5 Greens Road"]
    elif (replacement == ["name", "cambridge college"]):
        replacement = None
    elif (replacement == ['address', 'cafe jello gallery']):
        replacement = ['name', 'cafe jello gallery']
    elif (replacement == ['stars', 'Two']):
        replacement = ['stars', '2']
    elif (replacement == ['stars', 'two']):
        replacement = ['stars', '2']
    elif (replacement == ['stars', 'three']):
        replacement = ['stars', '3']
    elif (replacement == ['stars', 'four']):
        replacement = ['stars', '4']
    elif (replacement == ['stars', 'four - star']):
        replacement = ['stars', '4']
    return replacement


def Canonicalize(text: str) -> str:
    text = clear_whitespaces(text).lower()
    return NAME_MAP.get(text, text)


def Normalize(text: str) -> str:
    text = text.lower()
    text = re.sub("guesthouse", "guest house", text, flags=re.I)
    text = re.sub("guest\shouse", "guest house", text, flags=re.I)
    text = text.replace("moderately", "moderate")
    text = text.replace("cheaply", "cheap")
    text = text.replace("expensively", "expensive")
    text = " ".join(tknz(text))
    return text


def GetEntitySelectionScores(data: dict) -> dict:
    def GetSuccessNumber(entities: list, oracle_entities: dict, k: int = 5) -> int:
        for ent in entities[:k]:
            domain = ent[0]
            entity_name = ent[1]
            if (domain not in oracle_entities):
                continue
            for ent_name in oracle_entities[domain]:
                if (ent_name.lower() == entity_name.lower()):
                    return 1
        return 0

    def GetErrorNumber(entities: list) -> int:
        error = 0
        for ent in entities:
            confidence = ent[2]
            label = ent[3]
            if ((confidence < 0.5 and label == 1) or (confidence >= 0.5 and label == 0)):
                error += 1
        return error

    def Flatten(entities: dict, filter: bool = False) -> list:
        l = []
        for domain in entities:
            for entity_name in entities[domain]:
                l.append(
                    [
                        domain,
                        entity_name,
                        entities[domain][entity_name]["confidence"],
                        entities[domain][entity_name]["predicted_label"]
                    ]
                )
        if (filter):
            q = []
            for i in l:
                if (i[3] == 1):
                    q.append(i)
            l = q
        l.sort(key=lambda x: x[2], reverse=True)
        return l

    total_context_response_pairs = 0
    total_best_entity_examples = 0
    success1 = 0
    successk = 0
    recall1 = 0
    recallk = 0
    label_error = 0
    cr_labels_examples = 0
    for dialog in data["dialogues"]:
        for turn in dialog["items"]:
            if ("entity_selection" in turn):
                entities_filtered = Flatten(turn["entity_selection"], True)
                total_context_response_pairs += 1
                s1 = GetSuccessNumber(
                    entities_filtered, turn["oracle_entities"], 1)
                sk = GetSuccessNumber(
                    entities_filtered, turn["oracle_entities"], 5)
                success1 += s1
                successk += sk
                turn["s1"] = s1
                turn["sk"] = sk
                if ("best_entity" in turn):
                    total_best_entity_examples += 1
                    r = {
                        turn["best_entity"]["domain"]: {
                            turn["best_entity"]["entity_name"]
                        }
                    }
                    r1 = GetSuccessNumber(entities_filtered, r, 1)
                    rk = GetSuccessNumber(entities_filtered, r, 5)
                    recall1 += r1
                    recallk += rk
                    turn["r1"] = r1
                    turn["rk"] = rk

                entities = Flatten(turn["entity_selection"])
                label_error += GetErrorNumber(entities)
                cr_labels_examples += len(entities)

    results = {
        "entity_selection": {}
    }
    if (total_context_response_pairs != 0):
        results["entity_selection"]["total_context_response_pairs"] = total_context_response_pairs
        _, results["entity_selection"]["success@1"] = MakeFractionalString(
            success1, total_context_response_pairs)
        _, results["entity_selection"]["success@k"] = MakeFractionalString(
            successk, total_context_response_pairs)
        _, results["entity_selection"]["label_error"] = MakeFractionalString(
            label_error, cr_labels_examples)
        _, results["entity_selection"]["recall@1"] = MakeFractionalString(
            recall1, total_best_entity_examples)
        _, results["entity_selection"]["recall@k"] = MakeFractionalString(
            recallk, total_best_entity_examples)
    return results


def GetSlotValuesFromText(text: str,
                          requestable_values: dict,
                          threshold: int = 90) -> list:
    text = Normalize(text)
    slot_values = []

    for slot in requestable_values:
        for domain, value in requestable_values[slot]:
            exists, text = CheckExistsInTextAndUpdate(
                text, Normalize(value), 100)
            if (exists):
                dsv = DSV(domain, slot, value)
                slot_values.append(dsv)

    if (threshold != 100):
        for slot in requestable_values:
            for domain, value in requestable_values[slot]:
                exists, text = CheckExistsInTextAndUpdate(
                    text, Normalize(value), threshold)
                if (exists):
                    dsv = DSV(domain, slot, value)
                    slot_values.append(dsv)

    slot_values = list(set(slot_values))
    slot_values.sort(key=lambda x: (x.domain, x.slot, x.value))
    return slot_values


def FilterSlotValues(slot_values: List[DSV],
                     use_unique_slots: bool = False,
                     no_name: bool = False) -> List[DSV]:
    results = []
    for slot_value in slot_values:
        if (use_unique_slots and slot_value.slot not in UNIQUE_SLOTS):
            continue

        if (no_name and slot_value.slot == "name"):
            continue

        if (slot_value.value in ["?"] + BOOLEAN_STRINGS):
            continue
        results.append(slot_value)
    return results


def UpdateTurnWithSlotValues(turn: dict,
                             gt_values: List[DSV],
                             ge_values: List[DSV]) -> None:
    turn["GT_slot_values"] = [x.GetRepresentation() for x in gt_values]
    turn["GE_slot_values"] = [x.GetRepresentation() for x in ge_values]


def GetResponseGenerationScores(data: dict,
                                db_raw: dict,
                                requestable_values: dict,
                                use_unique_slots: bool = False,
                                no_name: bool = False) -> dict:
    metrics = Metrics()

    num_examples = 0
    for dialog in tqdm(data["dialogues"]):
        for turn in dialog["items"]:
            if ("response_generation" in turn):
                num_examples += 1
                metrics.UpdateBleu(
                    turn["text"],
                    turn["response_generation"]["generated_response"]
                )

                gt_values = GetSlotValuesFromText(
                    turn["text"],
                    requestable_values
                )
                gt_values = FilterSlotValues(
                    gt_values,
                    use_unique_slots=use_unique_slots,
                    no_name=no_name
                )
                if (not gt_values):
                    continue

                ge_values = GetSlotValuesFromText(
                    turn["response_generation"]["generated_response"],
                    requestable_values
                )
                ge_values = FilterSlotValues(
                    ge_values,
                    use_unique_slots=use_unique_slots,
                    no_name=no_name
                )

                metrics.UpdateTP_FP_FN_Hallucination(
                    gt_values,
                    ge_values,
                    turn["response_generation"]["domain"],
                    turn["response_generation"]["entity_name"],
                    db_raw
                )

                metrics.UpdateCopyScores(ge_values,
                                         turn["response_generation"]["domain"],
                                         turn["response_generation"]["entity_name"],
                                         db_raw)

                UpdateTurnWithSlotValues(turn, gt_values, ge_values)

                print("GT text =", turn["text"])
                print("GT =", ConvertDSVtoSV(gt_values))
                print("GE text =", turn["response_generation"]["generated_response"])
                print("GE =", ConvertDSVtoSV(ge_values))
                print("used_domain =", turn["response_generation"]["domain"])
                print("used_entity_name =", turn["response_generation"]["entity_name"])
                print()
                print("==================================================")
                print()

    results = {}
    results["response_generation"] = metrics.Scores()
    print("num_examples =", num_examples)
    return results


def GetSlotEntityMapping(db_raw: DB) -> dict:
    r = {}
    for ent in db_raw:
        domain, entity_name, db_entity = ent
        for slot_value in db_entity:
            slot = slot_value.slot
            value = slot_value.value
            value = value.lower()
            if (value not in r):
                r[value] = [(domain, entity_name, slot)]
            else:
                r[value].append((domain, entity_name, slot))
    return r


def GetOntology(db_raw: DB) -> dict:
    r = {}
    for ent in db_raw:
        domain, _, db_entity = ent
        if (domain not in ALLOWED_DOMAINS):
            continue

        for slot_value in db_entity:
            slot = slot_value.slot
            value = slot_value.value
            value = value.lower()
            if (domain + "-" + slot not in r):
                r[domain + "-" + slot] = []
            r[domain + "-" + slot].append(value)

    # sort values
    for ds in r:
        r[ds] = list(set(r[ds]))
        r[ds].sort(key=lambda x: (-len(x), x))

    # sort keys
    l = {}
    for ds in SORTED_KEYS:
        l[ds] = r[ds]
    for ds in r:
        if (ds not in SORTED_KEYS):
            l[ds] = r[ds]
    return l


def GetRequestableValues(db_ontology: dict) -> dict:
    r = {}
    for ds in db_ontology:
        domain, slot = ds.split("-")
        if (slot in REQUESTABLE_SLOTS[domain]):
            if (slot not in r):
                r[slot] = []
            for value in db_ontology[ds]:
                r[slot].append((domain, value))
    for slot in r:
        r[slot] = list(set(r[slot]))
        r[slot].sort(key=lambda x: (-len(x[1]), x[1], x[0]))
    return r


def CheckExistsInTextAndUpdate(haystack: str,
                               needle: str,
                               threshold: int = 90) -> Tuple[bool, str]:
    haystack_ = haystack.split()

    if (len(needle) == 1):
        l = []
        found = False
        for i in haystack_:
            if (i == needle):
                found = True
                l.append("<:::slot_value:::>")
            else:
                l.append(i)
        return found, " ".join(l)
    else:
        needle_length = len(needle.split())

        for i in range(len(haystack_) - needle_length + 1):
            substring = " ".join(haystack_[i: i + needle_length])

            if ((threshold == 100 and needle == substring)
                    or fuzz.ratio(needle, substring) > threshold):
                return True, " ".join(haystack_[: i] + ["<:::slot_value:::>"] + haystack_[i + needle_length:])

        return False, haystack


def GetMatchScore(x: str, y: str) -> bool:
    x = x.lower()
    y = y.lower()
    score = fuzz.ratio(x, y)
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str,
                        default="", help="file to evaluate")
    parser.add_argument("--score_file", type=str,
                        default="", help="file to dump")
    parser.add_argument("--slot_value_file", type=str,
                        default="", help="file to dump")
    parser.add_argument("--raw_db_path", type=str, help="db path")
    parser.add_argument("--db_path", type=str, help="db path")
    parser.add_argument("--use_unique_slots", action="store_true")
    parser.add_argument("--no_name", action="store_true")
    args = parser.parse_args()

    data = json.load(open(args.output_file, "r"))
    db_raw = DB(args.raw_db_path)

    db_ontology = GetOntology(db_raw)
    json.dump(db_ontology, open("db_ontology.json", "w"), indent=4)

    requestable_values = GetRequestableValues(db_ontology)
    json.dump(requestable_values, open(
        "requestable_values.json", "w"), indent=4)

    scores = {}
    entity_selection_scores = GetEntitySelectionScores(data)

    response_generation_scores = GetResponseGenerationScores(
        data,
        db_raw,
        requestable_values,
        use_unique_slots=args.use_unique_slots,
        no_name=args.no_name
    )
    scores = {
        "entity_selection": entity_selection_scores,
        "response_generation": response_generation_scores
    }
    json.dump(scores, open(args.score_file, "w"), indent=4)
    if (args.slot_value_file):
        json.dump(data, open(args.slot_value_file, "w"), indent=4)


if (__name__ == "__main__"):
    main()
