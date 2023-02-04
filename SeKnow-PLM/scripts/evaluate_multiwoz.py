#!/bin/env python
import argparse
import logging
import os
from typing import List

import torch
import transformers

from data import load_dataset  # noqa: E402
from data.evaluation.multiwoz import (MultiWozEvaluator,
                                      compute_bmr_remove_reference,
                                      compute_delexicalized_bmr)
from data.utils import (BeliefParser, DatabaseParser,  # noqa: E402
                        DialogDataset, wrap_dataset_with_cache)
from generate import generate_predictions  # noqa:E402
from utils import pull_model, setup_logging  # noqa:E402

# from evaluation_utils import compute_delexicalized_bleu  # noqa:E402


def CustomDelexer(evaluator: MultiWozEvaluator,
                  dataset: DialogDataset,
                  responses: List[str]):
    tmp = len(responses) * [None]
    dialogues = evaluator.pack_dialogues(
        dataset,
        tmp,
        tmp,
        tmp,
        responses
    )
    delexed_responses = []
    for items, _, _, _, _, responses, _ in dialogues:
        for item, response in zip(items, responses):
            # response = " ".join(nltk.word_tokenize(response))
            for delex_text, text in item.replacements:
                # delex_text = " ".join(nltk.word_tokenize(delex_text))
                delex_text = "[" + delex_text + "]"
                response = response.replace(text, delex_text)
            delexed_responses.append(response)
    return delexed_responses


def parse_predictions(filename):
    gts, bfs, ds, rs, docs = [], [], [], [], []
    delexrs, delexgts = [], []
    bf_parser = BeliefParser()
    d_paser = DatabaseParser()
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('GT:'):
                gts.append(line[len('GT:'):])
            elif line.startswith('GTD:'):
                delexgts.append(line[len('GTD:'):])
            elif line.startswith('BF:'):
                bf = line[len('BF:'):]
                bf = bf_parser(bf)
                assert bf is not None
                bfs.append(bf)
            elif line.startswith('RD:'):
                delexrs.append(line[len('RD:'):])
            elif line.startswith('R:'):
                r = line[len('R:'):]
                rs.append(r)
            elif line.startswith('D:'):
                d = line[len('D:'):]
                if (d == ""):
                    d = "restaurant no match"
                d = d_paser(d)
                assert d is not None
                ds.append(d)
            elif line.startswith("DOC:"):
                doc = [line[len("DOC:"):]]
                docs.append(doc)
    return rs, bfs, ds, gts, delexrs, delexgts, docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--file', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dataset', default='multiwoz-2.1-test')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--delex', action='store_true')
    parser.add_argument('--use_blacklist', action='store_true')
    args = parser.parse_args()
    if args.resume is not None and args.model is None:
        args.model = f'wandb:{args.resume}'
    assert args.model is not None or args.file is not None

    # Update punkt
    # nltk.download('punkt')

    setup_logging()
    logger = logging.getLogger()
    if args.resume:
        import wandb

        # Resume run and fill metrics
        os.environ.pop('WANDB_NAME', None)
        wandb.init(resume=args.resume)
    elif args.wandb:
        import wandb

        # It is an artifact
        # Start a new evaluate run
        wandb.init(job_type='evaluation')
    else:
        wandb = None

    dataset = load_dataset(args.dataset, use_goal=True, use_blacklist=args.use_blacklist)
    dataset = wrap_dataset_with_cache(dataset)

    if args.file is None or not os.path.exists(args.file):
        args.model = pull_model(args.model)

    if args.file is not None:
        path = args.file
        if not os.path.exists(path):
            path = os.path.join(args.model, args.file)
        responses, beliefs, databases, gold_responses, delex_responses, delex_gold_responses, documents = \
            parse_predictions(path)
    else:
        logger.info('generating responses')
        pipeline = transformers.pipeline(
            'augpt-conversational', args.model, device=0 if torch.cuda.is_available() else -1)
        beliefs, databases, documents, responses, gold_responses, delex_responses, delex_gold_responses = \
            generate_predictions(pipeline, dataset, os.path.join(
                wandb.run.dir if wandb and wandb.run else '.', 'test-predictions.txt'))

    evaluator = MultiWozEvaluator(
        dataset, is_multiwoz_eval=True, logger=logger)

    if (args.delex):
        delex_responses = CustomDelexer(evaluator, dataset, delex_responses)
        delex_gold_responses = CustomDelexer(
            evaluator, dataset, delex_gold_responses)

    joint_goal, db_correct, mmr5, r1, r5, matches, success, domain_results, dp, dr, df = \
        evaluator.evaluate(beliefs, databases, documents,
                           delex_responses, progressbar=True, pre_num=1)

    logger.info(f'joint goal: {joint_goal:.4f}, db acc: {db_correct:.4f}')
    logger.info(f'match: {matches:.4f}, success: {success:.4f}')
    logger.info(f'joint goal: {joint_goal:.4f}, db acc: {db_correct:.4f}')
    logger.info(f'detect_p: {dp:.4f}, detect_r: {dr:.4f}, detect_f: {df:.4f}')
    logger.info(f'MRR@5: {mmr5:.4f}, R@1: {r1:.4f}, R@5: {r5:.4f}')
    logger.info(f'inform: {matches:.4f}, success: {success:.4f}')

    if wandb and wandb.run:
        wandb.run.summary.update(dict(
            test_joint_goal=joint_goal,
            test_db_acc=db_correct,
        ))
        wandb.run.summary.update(dict(
            test_inform=matches,
            test_success=success,
        ))

    if dataset.lexicalizer is not None:
        bleu, meteor, rouge = compute_bmr_remove_reference(
            responses, gold_responses)
        logger.info(
            f'bleu: {bleu:.4f}, meteor: {meteor:.4f}, rouge: {rouge:.4f}')

        if wandb and wandb.run:
            wandb.run.summary.update(
                dict(test_bleu=bleu, test_meteor=meteor, test_rouge=rouge))

    delex_bleu, delex_meteor, delex_rouge = compute_delexicalized_bmr(
        delex_responses, delex_gold_responses)
    logger.info(
        f'delex_bleu: {delex_bleu:.4f}, delex_meteor: {delex_meteor:.4f}, delex_rouge: {delex_rouge:.4f}')

    if wandb and wandb.run:
        wandb.run.summary.update(dict(
            test_delex_bleu=delex_bleu, test_delex_meteor=delex_meteor, test_delex_rouge=delex_rouge))
