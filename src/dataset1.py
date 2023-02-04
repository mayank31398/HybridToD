import copy
import json
import logging
import os
import random
import zipfile
from argparse import Namespace
from typing import Tuple, Union

from tqdm import tqdm
from transformers import AutoTokenizer

from values1 import ALLOWED_DOMAINS, ALLOWED_SLOTS, SKIP_CASE, SPECIAL_TOKENS

logger = logging.getLogger(__name__)


class DSV:
    def __init__(self,
                 domain: str,
                 slot: str,
                 value: str) -> None:
        self.domain = domain
        self.slot = slot
        self.value = value

    def __repr__(self) -> str:
        x = (self.domain, self.slot, self.value)
        if (self.domain == None):
            x = (self.slot, self.value)
        return str(x)

    def __eq__(self, __o: object) -> bool:
        if (type(__o) != DSV):
            return False
        return self.domain == __o.domain and self.slot == __o.slot and self.value == __o.value

    def __hash__(self) -> int:
        return hash(str(self))

    def GetRepresentation(self) -> tuple:
        return (self.slot, self.value)


class SlotValue:
    def __init__(self,
                 slot: str,
                 value: Union[str, int, list, dict],
                 add_space: bool = False) -> None:
        self.slot = slot
        self.value = value

        space = ""
        if (add_space):
            space = " "

        self.slot_value_representation = SPECIAL_TOKENS["slot"] + \
            space + slot + space + SPECIAL_TOKENS["value"] + space

        if (type(value) == list):
            for v in value:
                self.slot_value_representation += SPECIAL_TOKENS["list_value"] + space + str(
                    v) + space
        elif (type(value) == dict):
            for s, v in value.items():
                self.slot_value_representation += SPECIAL_TOKENS["dict_slot"] + space + str(
                    s) + space + SPECIAL_TOKENS["dict_value"] + space + str(v) + space
        else:
            self.slot_value_representation += str(value)

        self.slot_value_representation = self.slot_value_representation.strip()
        self.slot_value_representation = " ".join(
            self.slot_value_representation.split())

    def __repr__(self) -> str:
        return self.slot_value_representation

    def GetValue(self) -> Union[str, int, list, dict]:
        return self.value


class DB_Entity:
    def __init__(self,
                 domain: str,
                 entity_name: str,
                 entity: dict,
                 add_space: bool = False) -> None:
        self.domain = domain
        self.entity_name = entity_name

        space = ""
        if (add_space):
            space = " "

        self.entity = {}
        self.entity_representation = SPECIAL_TOKENS["db"] + space
        for slot, value in entity.items():
            if (slot in ALLOWED_SLOTS):
                slot_value = SlotValue(slot, value, add_space)
                self.entity[slot] = slot_value
                self.entity_representation += str(slot_value) + space
        self.entity_representation = self.entity_representation.strip()
        self.entity_representation = " ".join(
            self.entity_representation.split())

    def __repr__(self) -> str:
        return self.entity_representation

    def __iter__(self) -> SlotValue:
        for _, slot_value in self.entity.items():
            yield slot_value

    def HasSlot(self, slot: str) -> bool:
        return slot in self.entity

    def GetSlotValue(self, slot: str) -> SlotValue:
        return self.entity[slot]


class DB:
    def __init__(self,
                 data_path: str,
                 add_space: bool = False,
                 use_no_entity: bool = False) -> None:
        self.db = {}
        self.add_space = add_space

        self.use_no_entity = use_no_entity
        self.has_no_entity = False
        if (self.use_no_entity):
            self.no_entity = DB_Entity("no_entity", "no_entity", {}, add_space)

        with zipfile.ZipFile(os.path.join(data_path, "database.zip"), "r") as file:
            filenames = file.namelist()
            for domain in ALLOWED_DOMAINS:
                db_filename = os.path.join("db", domain + "_db.json")
                if (db_filename not in filenames):
                    continue

                db = json.load(file.open(db_filename, "r"))

                self.db[domain] = {}
                for entity in db:
                    entity_name = entity["name"]
                    # NOTE lowercase entity name to match with KB
                    self.db[domain][entity_name.lower()] = DB_Entity(
                        domain,
                        entity_name,
                        entity,
                        add_space
                    )

    def GetNoEntity(self) -> Tuple[str, str, DB_Entity]:
        if (self.use_no_entity and self.has_no_entity):
            return self.no_entity.domain, self.no_entity.entity_name, self.no_entity
        raise ZeroDivisionError

    def GetEntity(self, domain: str, entity_name: str) -> DB_Entity:
        if (self.use_no_entity and self.has_no_entity):
            if (not self.db.get(domain) or not self.db[domain].get(entity_name)):
                return self.no_entity
        return self.db[domain][entity_name]

    def HasEntity(self, domain: str, entity_name: str) -> bool:
        return domain in self.db and entity_name in self.db[domain]

    def IsNoEntity(self, entity: DB_Entity) -> bool:
        if (self.use_no_entity and self.has_no_entity):
            return entity.domain == "no_entity" and entity.entity_name == "no_entity"
        raise ZeroDivisionError

    def SetNoEntity(self, x: bool) -> None:
        if (self.use_no_entity):
            self.has_no_entity = x

    def __iter__(self) -> Tuple[str, str, DB_Entity]:
        for domain in self.db:
            for entity_name in self.db[domain]:
                yield domain, entity_name, self.db[domain][entity_name]

        if (self.use_no_entity and self.has_no_entity):
            yield self.no_entity.domain, self.no_entity.entity_name, self.no_entity


class Document:
    def __init__(self,
                 topic: str,
                 title: str,
                 body: str,
                 use_topic: bool = False,
                 use_title: bool = False,
                 add_space: bool = False) -> None:
        self.topic = topic
        self.title = title
        self.body = body

        space = ""
        if (add_space):
            space = " "

        self.document_representation = SPECIAL_TOKENS["doc"] + space
        if (use_topic and use_title):
            self.document_representation += SPECIAL_TOKENS["doc_topic"] + \
                space + topic + space + SPECIAL_TOKENS["doc_title"] + \
                space + title + space + SPECIAL_TOKENS["doc_body"] + space
        elif (use_topic):
            self.document_representation += SPECIAL_TOKENS["doc_topic"] + \
                space + topic + space + SPECIAL_TOKENS["doc_body"] + space
        elif (use_title):
            self.document_representation += SPECIAL_TOKENS["doc_title"] + \
                space + title + space + SPECIAL_TOKENS["doc_body"] + space
        self.document_representation += body
        self.document_representation = self.document_representation.strip()
        self.document_representation = " ".join(
            self.document_representation.split())

    def __repr__(self) -> str:
        return self.document_representation


class KB_Entity:
    def __init__(self,
                 domain: str,
                 entity_name: str,
                 entity: dict,
                 use_topic: bool = False,
                 use_title: bool = False,
                 add_space: bool = False) -> None:
        self.domain = domain
        self.entity_name = entity_name

        space = ""
        if (add_space):
            space = " "

        self.entity = {}
        self.entity_representation = SPECIAL_TOKENS["kb"] + space
        for topic, doc in entity.items():
            document = Document(
                topic,
                doc["title"],
                doc["body"],
                use_topic,
                use_title,
                add_space
            )
            self.entity[topic] = document
            self.entity_representation += str(document) + space
        self.entity_representation = self.entity_representation.strip()
        self.entity_representation = " ".join(
            self.entity_representation.split())

    def __repr__(self) -> str:
        return self.entity_representation

    def __iter__(self) -> Document:
        for _, document in self.entity:
            yield document


class KB:
    def __init__(self,
                 data_path: str,
                 use_topic: bool = False,
                 use_title: bool = False,
                 add_space: bool = False,
                 use_no_entity: bool = False) -> None:
        self.kb = {}
        self.add_space = add_space

        kb = json.load(
            open(os.path.join(data_path, "document_base.json"), "r"))
        for domain in ALLOWED_DOMAINS:
            if (domain not in kb):
                continue

            self.kb[domain] = {}
            for entity_name in kb[domain]:
                self.kb[domain][entity_name] = KB_Entity(
                    domain,
                    entity_name,
                    kb[domain][entity_name],
                    use_topic,
                    use_title,
                    add_space
                )

        self.use_topic = use_topic
        self.use_title = use_title

        self.use_no_entity = use_no_entity
        self.has_no_entity = False
        if (self.use_no_entity):
            self.no_entity = KB_Entity(
                "no_entity",
                "no_entity",
                {},
                self.use_topic,
                self.use_title,
                self.add_space
            )

    def GetNoEntity(self) -> KB_Entity:
        if (self.has_no_entity and self.has_no_entity):
            return self.no_entity
        raise ZeroDivisionError

    def GetEntity(self, domain: str, entity_name: str) -> KB_Entity:
        if (self.use_no_entity and self.has_no_entity):
            if (not self.kb.get(domain) or not self.kb[domain].get(entity_name)):
                return self.no_entity
        if (domain == "attraction" and domain not in self.kb):
            return KB_Entity(
                domain,
                entity_name,
                {},
                self.use_topic,
                self.use_title,
                self.add_space
            )
        return self.kb[domain][entity_name]

    def SetNoEntity(self, x: bool) -> None:
        if (self.use_no_entity):
            self.has_no_entity = x

    def __iter__(self) -> Tuple[str, str, DB_Entity]:
        for domain in self.kb:
            for entity_name in self.kb[domain]:
                yield domain, entity_name, self.kb[domain][entity_name]

        if (self.use_no_entity and self.has_no_entity):
            yield self.no_entity.domain, self.no_entity.entity_name, self.no_entity


class MultiwozDataset:
    def __init__(self,
                 args: Namespace,
                 tokenizer: AutoTokenizer,
                 db: DB,
                 kb: KB,
                 data_path: str,
                 repeat: bool = True,
                 pos_entity_selection_batch_size: int = 1,
                 neg_entity_selection_batch_size: int = 1,
                 response_generation_batch_size: int = 1) -> None:
        self.tokenizer = tokenizer
        self.repeat = repeat
        self.has_entity_selection_enabled = (
            args.num_positive_examples_in_batch + args.num_negative_examples_in_batch) != 0
        self.has_response_generation_enabled = args.num_response_generation_examples_in_batch != 0

        self.num_examples = 0
        if (self.has_entity_selection_enabled):
            self.multiwoz_entity_selection_dataset = MultiwozEntitySelectionDataset(
                args,
                tokenizer,
                db,
                kb,
                data_path,
                repeat,
                pos_entity_selection_batch_size,
                neg_entity_selection_batch_size
            )
            self.num_examples += self.multiwoz_entity_selection_dataset.num_examples
        if (self.has_response_generation_enabled):
            self.multiwoz_response_generation_dataset = MultiwozResponseGenerationDataset(
                args,
                tokenizer,
                db,
                kb,
                data_path,
                repeat,
                response_generation_batch_size
            )
            self.num_examples += self.multiwoz_response_generation_dataset.num_examples

    def __iter__(self):
        if (self.repeat):
            if (self.has_entity_selection_enabled):
                es_iterator = iter(self.multiwoz_entity_selection_dataset)
            if (self.has_response_generation_enabled):
                rg_iterator = iter(self.multiwoz_response_generation_dataset)

            while (True):
                x = []
                if (self.has_entity_selection_enabled):
                    es_batch = next(es_iterator)
                    x += es_batch
                if (self.has_response_generation_enabled):
                    rg_batch = next(rg_iterator)
                    x += rg_batch
                random.shuffle(x)
                yield x
        else:
            if (self.has_entity_selection_enabled):
                for x in self.multiwoz_entity_selection_dataset:
                    yield x
            if (self.has_response_generation_enabled):
                for x in self.multiwoz_response_generation_dataset:
                    yield x

    def __len__(self):
        if (not self.repeat):
            n = 0
            if (self.has_entity_selection_enabled):
                num_es_batches = len(self.multiwoz_entity_selection_dataset)
                n += num_es_batches
            if (self.has_response_generation_enabled):
                num_rg_batches = len(self.multiwoz_response_generation_dataset)
                n += num_rg_batches
            return n


class MultiwozEntitySelectionDataset:
    def __init__(self,
                 args: Namespace,
                 tokenizer: AutoTokenizer,
                 db: DB,
                 kb: KB,
                 data_path: str,
                 repeat: bool = True,
                 pos_batch_size: int = 1,
                 neg_batch_size: int = 1) -> None:
        self.tokenizer = tokenizer
        self.num_turns = args.num_turns
        self.max_length = tokenizer.max_model_input_sizes[args.model_name]
        self.repeat = repeat
        self.use_explanations = args.use_explanations

        self.db = db
        self.kb = kb

        self.pos_batch_size = pos_batch_size
        self.neg_batch_size = neg_batch_size

        logger.info("creating entity selection examples")
        data = json.load(open(data_path))
        self.positive_examples, self.negative_examples = self.CreateEntitySelectionExamples(
            data)

        self.num_examples = len(self.positive_examples) + \
            len(self.negative_examples)
        logger.info("num entity selection examples = " +
                    str(self.num_examples))
        logger.info("num positive entity selection examples = " +
                    str(len(self.positive_examples)))
        logger.info("num negative entity selection examples = " +
                    str(len(self.negative_examples)))

    def CreateEntitySelectionExamples(self, data: dict):
        def GetNegativeUniverses(positive_entities: dict,
                                 reduced_universe: dict):
            negative_universe = copy.deepcopy(reduced_universe)

            for domain in positive_entities:
                if (domain in negative_universe):
                    negative_universe[domain] = list(
                        set(negative_universe[domain]).difference(positive_entities[domain]))

            return negative_universe

        positive_examples = []
        negative_examples = []
        for dialog in tqdm(data["dialogues"]):
            fname = dialog["name"]
            for turn in dialog["items"]:
                if (turn["speaker"] != "system"):
                    continue

                if (not (turn["active_domain"] in ALLOWED_DOMAINS or turn["active_domain"] == None)):
                    continue

                positive_entities = turn["oracle_entities"]
                if (turn.get("best_entity")):
                    positive_entities = {
                        turn["best_entity"]["domain"]: {
                            turn["best_entity"]["entity_name"]
                        }
                    }

                negative_universe = GetNegativeUniverses(
                    positive_entities,
                    turn["reduced_universe"]
                )

                for domain in positive_entities:
                    for entity_name in positive_entities[domain]:
                        explanation = ""
                        if (self.use_explanations):
                            explanation = GetExplanation(
                                turn["belief"],
                                domain,
                                entity_name,
                                self.db,
                                "yes"
                            )

                        ex = MakeEntitySelectionExample(
                            dialog["items"],
                            "yes",
                            self.tokenizer,
                            self.db,
                            self.kb,
                            explanation,
                            domain,
                            entity_name,
                            turn["turn_number"],
                            self.num_turns,
                            self.max_length,
                            self.use_explanations
                        )
                        ex["fname"] = fname
                        positive_examples.append(ex)

                for domain in negative_universe:
                    for entity_name in negative_universe[domain]:
                        explanation = ""
                        if (self.use_explanations):
                            explanation = GetExplanation(
                                turn["belief"],
                                domain,
                                entity_name,
                                self.db,
                                "no"
                            )

                        ex = MakeEntitySelectionExample(
                            dialog["items"],
                            "no",
                            self.tokenizer,
                            self.db,
                            self.kb,
                            explanation,
                            domain,
                            entity_name,
                            turn["turn_number"],
                            self.num_turns,
                            self.max_length,
                            self.use_explanations
                        )
                        ex["fname"] = fname
                        negative_examples.append(ex)

        return positive_examples, negative_examples

    def GetExamplesIterator(self, examples: list, batch_size: int):
        # skips extra examples outside batch
        num_batches = len(examples) // batch_size
        if (self.repeat):
            while (True):
                random.shuffle(examples)
                for i in range(num_batches):
                    start_index = i * batch_size
                    end_index = (i + 1) * batch_size
                    yield examples[start_index: end_index]
        else:
            for i in range(num_batches):
                start_index = i * batch_size
                end_index = (i + 1) * batch_size
                yield examples[start_index: end_index]

            if (end_index < len(examples)):
                yield examples[end_index:]

    def __iter__(self):
        pos_iterator = self.GetExamplesIterator(
            self.positive_examples, self.pos_batch_size)
        neg_iterator = self.GetExamplesIterator(
            self.negative_examples, self.neg_batch_size)

        if (self.repeat):
            while (True):
                pos_example = next(pos_iterator)
                neg_example = next(neg_iterator)
                x = pos_example + neg_example
                yield x
        else:
            for x in pos_iterator:
                yield x
            for x in neg_iterator:
                yield x

    def __len__(self):
        if (not self.repeat):
            num_pos_batches = len(
                self.positive_examples) // self.pos_batch_size
            num_neg_batches = len(
                self.negative_examples) // self.neg_batch_size
            return num_pos_batches + num_neg_batches


class MultiwozResponseGenerationDataset:
    def __init__(self,
                 args: Namespace,
                 tokenizer: AutoTokenizer,
                 db: DB,
                 kb: KB,
                 data_path: str,
                 repeat: bool = True,
                 batch_size: int = 1) -> None:
        self.db = db
        self.kb = kb
        self.tokenizer = tokenizer
        self.num_turns = args.num_turns
        self.max_length = tokenizer.max_model_input_sizes[args.model_name]
        self.repeat = repeat
        self.batch_size = batch_size
        self.use_no_entity = args.use_no_entity

        logger.info("creating response generation examples")
        data = json.load(open(data_path, "r"))
        self.response_generation_examples = self.CreateResponseGenerationExamples(
            data)
        self.num_examples = len(self.response_generation_examples)
        logger.info("num response generation examples = " +
                    str(self.num_examples))

    def CreateResponseGenerationExamples(self, data: dict):
        response_generation_examples = []
        count_no_best_ent = 0
        for dialog in tqdm(data["dialogues"]):
            fname = dialog["name"]
            for turn in dialog["items"]:
                if (turn["speaker"] != "system"):
                    continue

                if (not (turn["active_domain"] in ALLOWED_DOMAINS or turn["active_domain"] == None)):
                    continue

                domain = None
                entity_name = None
                if (self.use_no_entity):
                    domain = "no_entity"
                    entity_name = "no_entity"
                if (turn.get("best_entity")):
                    domain = turn["best_entity"]["domain"]
                    entity_name = turn["best_entity"]["entity_name"]
                elif (turn.get("oracle_entities")):
                    l = []
                    for d, ens in turn["oracle_entities"].items():
                        for en in ens:
                            l.append((d, en))
                    ex = random.sample(l, 1)[0]
                    domain = ex[0]
                    entity_name = ex[1]

                if ((self.use_no_entity and entity_name == "no_entity")
                        or (not self.use_no_entity and not entity_name)):
                    count_no_best_ent += 1

                ex = MakeResponseGenerationExample(
                    dialog["items"],
                    self.tokenizer,
                    self.db,
                    self.kb,
                    domain,
                    entity_name,
                    turn["turn_number"],
                    self.num_turns,
                    self.max_length
                )
                ex["fname"] = fname
                response_generation_examples.append(ex)

        logger.info("best entity not found in " +
                    str(count_no_best_ent) + " response generation examples")

        return response_generation_examples

    def __iter__(self):
        # skips extra examples outside batch
        num_batches = self.num_examples // self.batch_size
        if (self.repeat):
            while (True):
                random.shuffle(self.response_generation_examples)
                for i in range(num_batches):
                    start_index = i * self.batch_size
                    end_index = (i + 1) * self.batch_size
                    yield self.response_generation_examples[start_index: end_index]
        else:
            for i in range(num_batches):
                start_index = i * self.batch_size
                end_index = (i + 1) * self.batch_size
                yield self.response_generation_examples[start_index: end_index]

            if (end_index < len(self.response_generation_examples)):
                yield self.response_generation_examples[end_index:]

    def __len__(self):
        return self.num_examples // self.batch_size


def Pad(arrays: list,
        padding: int,
        max_length: int = -1):
    if (max_length < 0):
        max_length = max(list(map(len, arrays)))

    inputs = [array + [padding] * (max_length - len(array))
              for array in arrays]
    masks = [[1] * len(array) + [0] * (max_length - len(array))
             for array in arrays]

    return inputs, masks


def PrepareInput(dialog_turns: list,
                 tokenizer: AutoTokenizer,
                 db: DB,
                 kb: KB,
                 task_token_id: int,
                 turn_number: int,
                 num_turns: int,
                 domain: str,
                 entity_name: str,
                 max_length: int):
    user_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["user"])
    agent_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["system"])
    bos_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["bos"])
    eos_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos"])

    inp = []
    for j in range(max(0, turn_number - num_turns), turn_number):
        if (dialog_turns[j]["speaker"] == "user"):
            inp += [user_token_id]
        elif (dialog_turns[j]["speaker"] == "system"):
            inp += [agent_token_id]
        inp += Tokenize(tokenizer, dialog_turns[j]["text"])

    if (domain in ALLOWED_DOMAINS):
        db_ent = db.GetEntity(domain, entity_name)
        kb_ent = kb.GetEntity(domain, entity_name)
    elif (domain == None and entity_name == None):
        db_ent = ""
        kb_ent = ""
    else:
        db_ent = db.GetNoEntity()
        kb_ent = kb.GetNoEntity()

    info = str(db_ent) + str(kb_ent)
    info = Tokenize(tokenizer, info)

    # truncate context (this is a very extreme measure, rarely occurs)
    n = len(inp) + len(info) + 3 - max_length
    if (n > 0):
        inp = inp[:-n]
    inp = [bos_token_id] + [task_token_id] + inp + info + [eos_token_id]

    return inp


def MakeEntitySelectionExample(dialog_turns: list,
                               label: str,
                               tokenizer: AutoTokenizer,
                               db: DB,
                               kb: KB,
                               explanation: str,
                               domain: str,
                               entity_name: str,
                               turn_number: int,
                               num_turns: int,
                               max_length: int,
                               use_explanations: bool = True):
    bos_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["bos"])
    eos_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos"])
    task_entity_selection_token_id = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS["task_entity_selection"])
    entity_label_token_id = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS["entity_label"])
    entity_explanation_token_id = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS["entity_explanation"])

    inp = PrepareInput(
        dialog_turns,
        tokenizer,
        db,
        kb,
        task_entity_selection_token_id,
        turn_number,
        num_turns,
        domain,
        entity_name,
        max_length
    )

    out = [bos_token_id, entity_label_token_id] + Tokenize(tokenizer, label)
    if (use_explanations):
        out += [entity_explanation_token_id] + Tokenize(tokenizer, explanation)
    out += [eos_token_id]

    e = {
        "task": "entity_selection",
        "label": label,
        "explanation": explanation,
        "domain": domain,
        "entity_name": entity_name,
        "turn_number": turn_number,
        "input": inp,
        "output": out,
        "belief": dialog_turns[turn_number]["belief"]
    }
    return e


def MakeResponseGenerationExample(dialog_turns: list,
                                  tokenizer: AutoTokenizer,
                                  db: DB,
                                  kb: KB,
                                  domain: str,
                                  entity_name: str,
                                  turn_number: int,
                                  num_turns: int,
                                  max_length: int):
    bos_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["bos"])
    eos_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["eos"])
    task_response_generation_token_id = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS["task_response_generation"])

    inp = PrepareInput(
        dialog_turns,
        tokenizer,
        db,
        kb,
        task_response_generation_token_id,
        turn_number,
        num_turns,
        domain,
        entity_name,
        max_length
    )

    response = dialog_turns[turn_number]["text"]
    out = [bos_token_id] + Tokenize(tokenizer, response) + [eos_token_id]

    e = {
        "task": "response_generation",
        "response": response,
        "domain": domain,
        "entity_name": entity_name,
        "turn_number": turn_number,
        "input": inp,
        "output": out
    }
    return e


def Tokenize(tokenizer: AutoTokenizer, x: str):
    x = tokenizer(x)["input_ids"]
    # remove bos
    x = x[1:]
    # remove eos
    x = x[:-1]
    return x


def GetExplanation(dialog_state: dict,
                   domain: str,
                   entity_name: str,
                   db: DB,
                   label: str):
    if ("ruk" in dialog_state[domain] or
            "name" in dialog_state[domain]):
        if (label == "yes"):
            return "name matched"
        else:
            return "name didn't match"

    db_ent = db.GetEntity(domain, entity_name)
    if (db.IsNoEntity(db_ent)):
        return ""

    matched = []
    not_matched = []
    match = True
    for s, v in dialog_state[domain].items():
        if (s in ["people", "stay"] or
            (domain == "hotel" and s == "day") or
                (domain == "restaurant" and s in ["day", "time"])):
            continue

        if (v in SKIP_CASE):
            continue

        v = "yes" if v == "free" else v

        if (db_ent.HasSlot(s) and db_ent.GetSlotValue(s).GetValue() != v):
            not_matched.append(s)
            match = False
        elif (db_ent.HasSlot(s) and db_ent.GetSlotValue(s).GetValue() == v):
            matched.append(s)

    if (match):
        response = ", ".join(matched[:-1])
        if (len(matched) > 1):
            response += " and " + matched[-1] + " matched"
        else:
            response = matched[0] + " matched"
    else:
        response = ", ".join(not_matched[:-1])
        if (len(not_matched) > 1):
            response += " and " + not_matched[-1] + " didn't match"
        else:
            response = not_matched[0] + " didn't match"
    return response
