import copy
import logging
import re
from typing import List

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from dataset1 import DB, DSV

from values1 import ALLOWED_DOMAINS

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

logger = logging.getLogger(__name__)


def MakeFractionalString(numerator: int, denominator: int) -> str:
    if (numerator == 0 and denominator == 0):
        fraction = np.nan
    else:
        fraction = numerator / denominator
    text = str(fraction) + " (" + str(numerator) + \
        " / " + str(denominator) + ")"
    return fraction, text


def NormalizeText(text: str):
    result = text.lower()
    result = RE_PUNC.sub(' ', result)
    result = RE_ART.sub(' ', result)
    result = ' '.join(result.split())
    return result


def Bleu(ref_tokens: List[str], hyp_tokens: List[str], n: int = 4):
    if (len(hyp_tokens) == 0):
        return 0

    weights = [1 / n] * n
    score = sentence_bleu([ref_tokens], hyp_tokens, weights)

    return score


def ComputePRF(tps: int, fps: int, fns: int):
    micro_recall, micro_recall_text = MakeFractionalString(tps, tps + fns)
    micro_precision, micro_precision_text = MakeFractionalString(
        tps, tps + fps)
    micro_f1 = 2 * micro_precision * micro_recall / \
        (micro_precision + micro_recall + 1e-10)

    return {
        "precision": micro_precision_text,
        "recall": micro_recall_text,
        "f1": micro_f1
    }


def ComputePRFFromDict(tp: dict, fp: dict, fn: dict):
    result = {}
    for domain in tp:
        result[domain] = ComputePRF(
            tp[domain],
            fp[domain],
            fn[domain]
        )
    return result


def ComputeHallucinationRecallFromDict(hallucination: dict,
                                       tp: dict,
                                       fn: dict):
    result = {}
    for domain in tp:
        _, result[domain] = MakeFractionalString(
            hallucination[domain], tp[domain] + fn[domain])
    return result


def ConvertDSVtoSV(dsv_list: List[DSV]) -> List[DSV]:
    l = []
    for dsv in dsv_list:
        x = DSV(None, dsv.slot, dsv.value)
        l.append(x)
    l = list(set(l))
    l.sort(key=lambda x: (x.slot, x.value))
    return l


def GetTP_FP_FN_Hallucination(gt_values: List[DSV],
                              ge_values: List[DSV],
                              used_domain: str,
                              used_entity_name: str,
                              db_raw: DB,
                              domain: str) -> dict:
    if (domain != "global"):
        gt_values_ = []
        for dsv in gt_values:
            d = dsv.domain
            if (d == domain):
                gt_values_.append(dsv)

        ge_values_ = []
        for dsv in ge_values:
            d = dsv.domain
            if (d == domain):
                ge_values_.append(dsv)
    else:
        gt_values_ = ConvertDSVtoSV(gt_values)
        ge_values_ = ConvertDSVtoSV(ge_values)

    gt_values_ = set(gt_values_)
    ge_values_ = set(ge_values_)

    if (not gt_values_):
        None

    hallucination_numerator = 0
    if (used_domain != None and used_entity_name != None):
        for dsv in ge_values_:
            slot = dsv.slot
            value = dsv.value
            # print("BRUH")
            # print(dsv)
            # print(db_raw.GetEntity(used_domain, used_entity_name).HasSlot(slot))
            # if (db_raw.GetEntity(used_domain, used_entity_name).HasSlot(slot)):
            #     print(db_raw.GetEntity(used_domain, used_entity_name).GetSlotValue(slot).GetValue())
            if (dsv in gt_values_
                    and db_raw.GetEntity(used_domain, used_entity_name).HasSlot(slot)
                    and db_raw.GetEntity(used_domain, used_entity_name).GetSlotValue(slot).GetValue() == value):
                hallucination_numerator += 1

    return {
        "hallucination_numerator": hallucination_numerator,
        "tp": len(gt_values_.intersection(ge_values_)),
        "fp": len(ge_values_.difference(gt_values_)),
        "fn": len(gt_values_.difference(ge_values_))
    }


class Metrics():
    def __init__(self) -> None:
        self.bleu = {
            "bleu1": 0,
            "bleu2": 0,
            "bleu3": 0,
            "bleu4": 0,
            "num_examples": 0
        }

        self.tp = {
            "global": 0
        }
        allowed_domains = list(ALLOWED_DOMAINS)
        allowed_domains.sort()
        for domain in allowed_domains:
            self.tp[domain] = 0

        self.fp = copy.deepcopy(self.tp)
        self.fn = copy.deepcopy(self.tp)
        self.copy = {
            "num_slot_matches": 0,
            "num_slot_mismatches": 0,
            "num_examples": 0
        }
        self.hallucination_numerator = copy.deepcopy(self.tp)

    def UpdateBleu(self, ref_text: str, hyp_text: str):
        ref_tokens = NormalizeText(ref_text).split()
        hyp_tokens = NormalizeText(hyp_text).split()
        for n in range(1, 5):
            self.bleu["bleu" + str(n)] += Bleu(ref_tokens, hyp_tokens, n=n)

        self.bleu["num_examples"] += 1

    def UpdateTP_FP_FN_Hallucination(self,
                                     gt_values: List[DSV],
                                     ge_values: List[DSV],
                                     used_domain: str,
                                     used_entity_name: str,
                                     db_raw: DB):
        for domain in self.tp:
            numbers = GetTP_FP_FN_Hallucination(
                gt_values,
                ge_values,
                used_domain,
                used_entity_name,
                db_raw,
                domain
            )
            self.tp[domain] += numbers["tp"]
            self.fp[domain] += numbers["fp"]
            self.fn[domain] += numbers["fn"]
            self.hallucination_numerator[domain] += numbers["hallucination_numerator"]

    def UpdateCopyScores(self,
                         ge_values: List[DSV],
                         used_domain: str,
                         used_entity_name: str,
                         db_raw: DB):
        if (used_domain != None):
            for dsv in ge_values:
                if (db_raw.GetEntity(used_domain, used_entity_name).HasSlot(dsv.slot)
                        and db_raw.GetEntity(used_domain, used_entity_name).GetSlotValue(dsv.slot).GetValue() == dsv.value):
                    self.copy["num_slot_matches"] += 1
                else:
                    self.copy["num_slot_mismatches"] += 1
            self.copy["num_examples"] += 1

    def Scores(self):
        results = {}

        # bleu
        if (self.bleu["num_examples"] > 0):
            for n in range(1, 5):
                self.bleu["bleu" + str(n)] /= self.bleu["num_examples"]
            results["bleu"] = self.bleu

        # prf
        results["prf"] = ComputePRFFromDict(
            self.tp,
            self.fp,
            self.fn
        )

        # hallucination
        results["hallucination_recall"] = ComputeHallucinationRecallFromDict(
            self.hallucination_numerator,
            self.tp,
            self.fn
        )

        # copy
        _, self.copy["precision"] = MakeFractionalString(
            self.copy["num_slot_matches"],
            self.copy["num_slot_matches"] + self.copy["num_slot_mismatches"]
        )
        _, self.copy["avg_slot_matches"] = MakeFractionalString(
            self.copy["num_slot_matches"],
            self.copy["num_examples"]
        )
        _, self.copy["avg_slot_mismatches"] = MakeFractionalString(
            self.copy["num_slot_mismatches"],
            self.copy["num_examples"]
        )
        results["copy"] = self.copy

        return results
