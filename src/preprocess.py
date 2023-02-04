import copy
import json
import logging
import os
import random
import re
import shutil
import sys
import zipfile
from argparse import Namespace
from fuzzywuzzy import fuzz
import nltk
from nltk.tokenize import word_tokenize as tknz
from tqdm import tqdm

from dataset1 import DB, KB

logger = logging.getLogger(__name__)

sys.path.append("SeKnow-PLM/scripts")

from preprocess_convlab_multiwoz import Database

from values1 import ALLOWED_DOMAINS, BOOLEAN_STRINGS


def AddBestEntities(dialog: dict,
                    fuzzy_threshold: int = 80):
    def PreprocessKEKW(l: str):
        l = re.sub("&", " and ", l)
        l = re.sub("'", "", l)
        l = re.sub("guesthouse", "guest house", l, flags=re.I)
        l = re.sub("guest\shouse", "guest house", l, flags=re.I)
        l = " ".join(l.split())
        l = l.lower()
        return l

    def SetMatching(chunk, entity_name: str):
        for node in chunk:
            x = ""
            if (type(node) == nltk.Tree):
                x = " ".join([x[0] for x in node.leaves()])
            elif (node[1][:2] == "NN"):
                x = node[0]

            if (x == ""):
                continue

            x = PreprocessKEKW(x)
            x = set(x.split())
            if (len(x) == 1):
                continue

            # already preprocessed
            entity_words = set(entity_name.split())
            if (len(x.intersection(entity_words)) / len(entity_words) > 0.7):
                return True
        return False

    chunks = []
    for turn in dialog["items"]:
        chunks.append(nltk.ne_chunk(nltk.pos_tag(
            tknz(PreprocessKEKW(turn["text"])))))

    for response_turn in dialog["items"]:
        if (response_turn["speaker"] != "system"):
            continue

        if (response_turn.get("belief")):
            found = False
            for domain in response_turn["belief"]:
                if (domain not in ALLOWED_DOMAINS):
                    continue

                if ("ruk" in response_turn["belief"][domain] or
                        "name" in response_turn["belief"][domain]):
                    if ("oracle_entities" in response_turn and
                            domain in response_turn["oracle_entities"] and
                            len(response_turn["oracle_entities"][domain]) == 1):
                        dialog["items"][response_turn["turn_number"]]["best_entity"] = {
                            "domain": domain,
                            "entity_name": response_turn["oracle_entities"][domain][0].lower()
                        }
                    found = True
                    break
            if (found):
                continue

        strict_matches = {}
        fuzzy_matches = {}
        for domain in response_turn["oracle_entities"]:
            if (domain not in ALLOWED_DOMAINS):
                continue

            strict_matches[domain] = {}
            fuzzy_matches[domain] = {}
            for ent in response_turn["oracle_entities"][domain]:
                strict_matches[domain][ent] = 0
                fuzzy_matches[domain][ent] = []
                for turn in dialog["items"][:response_turn["turn_number"] + 1]:
                    turn_ = PreprocessKEKW(turn["text"])
                    ent_ = PreprocessKEKW(ent)

                    if (ent_ in turn_):
                        strict_matches[domain][ent] += 1

                    set_matching = SetMatching(
                        chunks[turn["turn_number"]], ent_)
                    if (set_matching):
                        fuzzy_matches[domain][ent].append(100)
                    else:
                        turn_s = turn_.split()
                        n = len(ent_.split())
                        for i in range(len(turn_s) - n + 1):
                            x = " ".join(turn_s[i: i + n])
                            val = fuzz.ratio(x, ent_)

                            if (val >= fuzzy_threshold):
                                fuzzy_matches[domain][ent].append(val)

        max_score = 0
        max_ent = None
        max_domain = None
        for domain in strict_matches:
            for ent in strict_matches[domain]:
                score = strict_matches[domain][ent] * \
                    100 + sum(fuzzy_matches[domain][ent])
                if (score > max_score):
                    max_score = score
                    max_ent = ent
                    max_domain = domain
        if (max_ent):
            dialog["items"][response_turn["turn_number"]]["best_entity"] = {
                "domain": max_domain,
                "entity_name": max_ent.lower()
            }

    return dialog


def QueryKB(constraints: dict, kb: dict) -> dict:
    def MyTokenizer(l: str):
        if (l[-1] == "?" or l[-1] == "."):
            l = l[:-1]
        l = l.split()
        return l

    """
    Returns oracle knowledge (HyKnow knowledge)
    """
    kb_result = {}
    for domain, constraint in constraints.items():
        if (domain not in ["hotel", "restaurant", "attraction"]):
            continue

        if (domain not in kb):
            continue

        if ("ruk" in constraint):
            kb_result[domain] = [constraint["ruk"].lower()]
            continue
        elif ("name" in constraint):
            ent_found_by_name = False
            cons = constraint["name"]
            for kbn in kb[domain]:
                if (kbn.endswith(cons) or kbn.startswith(cons) or
                        cons.endswith(kbn) or cons.startswith(kbn)):
                    kb_result[domain] = [kbn.lower()]
                    ent_found_by_name = True
                    break
            if (ent_found_by_name):
                continue

        for slot, value in constraint.items():
            if (slot in ["ruk", "name", "people", "stay", "type"] or
                (domain == "hotel" and slot == "day") or
                    (domain == "restaurant" and slot in ["day", "time"])):
                continue
            elif (domain in kb):
                for ent_name, docs in kb[domain].items():
                    for _, doc in docs.items():
                        title = MyTokenizer(doc["title"])
                        body = MyTokenizer(doc["body"])

                        found = False
                        if (value in BOOLEAN_STRINGS):
                            if (slot in title or slot in body):
                                found = True
                        elif (type(value) == int or type(value) == float):
                            if (slot in title or slot in body):
                                found = True
                        else:
                            if (value in title or value in body):
                                found = True

                        if (found):
                            if (domain in kb_result):
                                kb_result[domain].append(ent_name.lower())
                            else:
                                kb_result[domain] = [ent_name.lower()]

    return ConvertToSet(kb_result)


def GetUniverseSize(s: dict) -> int:
    x = 0
    for domain in s:
        x += len(s[domain])
    return x


def ExtendReducedUniverse(reduced_universe: dict, l: dict) -> dict:
    for domain in l:
        if (domain not in reduced_universe):
            reduced_universe[domain] = []
        reduced_universe[domain].extend(l[domain])
    return ConvertToSet(reduced_universe)


def ConvertToSet(d: dict) -> dict:
    for domain in d:
        d[domain] = list(set(d[domain]))
    return d


def InflateReducedUniverse(reduced_universe: dict, k: int, all_entities: set) -> dict:
    all_entities_ = copy.deepcopy(all_entities)
    for domain in reduced_universe:
        for entity_name in reduced_universe[domain]:
            all_entities_.remove(domain + ":::" + entity_name)

    l = random.sample(all_entities_, k)
    for de in l:
        domain, entity_name = de.split(":::")
        if (domain in reduced_universe):
            reduced_universe[domain].append(entity_name)
        else:
            reduced_universe[domain] = [entity_name]

    return ConvertToSet(reduced_universe)


def Preprocess(args: Namespace) -> None:
    if (os.path.isdir(args.preprocessed_data_path)):
        logger.info("preprocessed data already present at " +
                    args.preprocessed_data_path)
        return

    with zipfile.ZipFile(os.path.join(args.raw_data_path, 'database.zip')) as dbzipf:
        database_old = Database(dbzipf)

    with zipfile.ZipFile(os.path.join(args.new_data_path, 'database.zip')) as dbzipf:
        database_new = Database(dbzipf)

    db = DB(args.raw_data_path)
    all_entities = set()
    for domain, entity_name, _ in db:
        all_entities.add(domain + ":::" + entity_name)
    if (args.use_no_entity):
        no_entity_domain, no_entity_name, _ = db.GetNoEntity()

    kb = json.load(
        open(os.path.join(args.new_data_path, "document_base.json"), "r"))

    logger.info("preprocessing data and saving in " +
                args.preprocessed_data_path)
    os.makedirs(args.preprocessed_data_path, exist_ok=True)

    for split in ["train", "val", "test"]:
        data = json.load(
            open(os.path.join(args.new_data_path, split + ".json"), "r"))
        data_ds = json.load(
            open(os.path.join(args.raw_data_path, split + ".json"), "r"))

        reduced_universe_size_avg = 0
        oracle_entities_size_avg = 0
        total = 0
        for dialog, dialog_ds in tqdm(zip(data["dialogues"], data_ds["dialogues"])):
            for turn, turn_ds in zip(dialog["items"], dialog_ds["items"]):
                del turn["dialogue_act"]

                if (turn_ds["speaker"] == "system"):
                    belief = turn_ds["belief"]
                    turn["belief"] = belief

                    oracle_entities = database_old(belief, return_results=True)
                    turn["oracle_entities"] = {}
                    for domain in oracle_entities:
                        if (oracle_entities[domain][0] > 0):
                            ents = oracle_entities[domain][1]
                            turn["oracle_entities"][domain] = [
                                i["name"].lower() for i in ents]

                    db_entities = database_new(belief, return_results=True)
                    turn["DB_entities"] = {}
                    for domain in db_entities:
                        if (db_entities[domain][0] > 0):
                            ents = db_entities[domain][1]
                            turn["DB_entities"][domain] = [i["name"].lower()
                                                           for i in ents]

                    turn["KB_entities"] = QueryKB(belief, kb)

                    turn["reduced_universe"] = copy.deepcopy(
                        turn["oracle_entities"])
                    turn["reduced_universe"] = ExtendReducedUniverse(
                        turn["reduced_universe"], turn["DB_entities"])
                    turn["reduced_universe"] = ExtendReducedUniverse(
                        turn["reduced_universe"], turn["KB_entities"])

                    turn["reduced_universe"] = ConvertToSet(
                        turn["reduced_universe"])

                    n = GetUniverseSize(turn["reduced_universe"])
                    if (n < args.min_reduced_universe_size):
                        turn["reduced_universe"] = InflateReducedUniverse(
                            turn["reduced_universe"],
                            min(
                                args.min_reduced_universe_size - n,
                                len(all_entities) - n
                            ),
                            all_entities
                        )

                    r = GetUniverseSize(turn["oracle_entities"])
                    n = GetUniverseSize(turn["reduced_universe"])
                    if (n < 2 * r):
                        turn["reduced_universe"] = InflateReducedUniverse(
                            turn["reduced_universe"],
                            min(
                                2 * r - n,
                                len(all_entities) - n
                            ),
                            all_entities
                        )

                    for domain in turn["reduced_universe"]:
                        turn["reduced_universe"][domain] = list(
                            set(turn["reduced_universe"][domain]))

                    del turn["DB_entities"]
                    del turn["KB_entities"]

                    if ("best_entities" in turn):
                        del turn["best_entities"]

                    if (args.use_no_entity):
                        if (GetUniverseSize(turn["oracle_entities"]) == 0):
                            turn["oracle_entities"][no_entity_domain] = [
                                no_entity_name]
                        turn["reduced_universe"][no_entity_domain] = [
                            no_entity_name]

                    del turn["uk_based"]
                    del turn["document"]
                    del turn["booked_domains"]
                    del turn["database"]
                    del turn["delexicalised_text"]

                    reduced_universe_size_avg += GetUniverseSize(
                        turn["reduced_universe"])
                    oracle_entities_size_avg += GetUniverseSize(
                        turn["oracle_entities"])
                    total += 1

            dialog = AddBestEntities(dialog)

        logger.info("reduced universe average size = " +
                    str(reduced_universe_size_avg / total))
        logger.info("oracle entities average size = " +
                    str(oracle_entities_size_avg / total))

        json.dump(data, open(os.path.join(
            args.preprocessed_data_path, split + '.json'), "w"), indent=4)

    shutil.copyfile(os.path.join(args.new_data_path, 'database.zip'),
                    os.path.join(args.preprocessed_data_path, 'database.zip'))
    shutil.copyfile(os.path.join(args.new_data_path, 'document_base.json'),
                    os.path.join(args.preprocessed_data_path, 'document_base.json'))
    shutil.copyfile(os.path.join(args.new_data_path, 'train-blacklist.txt'),
                    os.path.join(args.preprocessed_data_path, 'train-blacklist.txt'))
    shutil.copyfile(os.path.join(args.new_data_path, 'val-blacklist.txt'),
                    os.path.join(args.preprocessed_data_path, 'val-blacklist.txt'))
    shutil.copyfile(os.path.join(args.new_data_path, 'test-blacklist.txt'),
                    os.path.join(args.preprocessed_data_path, 'test-blacklist.txt'))
