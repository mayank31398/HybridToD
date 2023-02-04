#!/bin/env python
import logging
from data import load_dataset
import torch
from torch.nn.parallel import DistributedDataParallel
import argparse
import dataclasses
import itertools
from collections import OrderedDict
from data.utils import DialogDatasetItem
from data.utils import BeliefParser, InsertLabelsTransformation, format_belief, format_database
from utils import pull_model, setup_logging
from pipelines import AuGPTConversation, get_context_from_conversation
import transformers
from tqdm import tqdm
from model import AuGPTModel, add_custom_tokens, AuGPTConfig, AuGPTTokenizer
from pipelines import AuGPTConversationalPipeline

def conversation_to_sample(conversation: AuGPTConversation):
    user = conversation.past_user_inputs
    if conversation.new_user_input is not None:
        user = user + [conversation.new_user_input]
    sys = conversation.generated_responses[:-1]
    context = get_context_from_conversation(user, sys)
    context = [x for x in itertools.chain(*itertools.zip_longest(user, sys)) if x is not None]
    database = OrderedDict((k, v[0]) for k, v in conversation.database_results.items())
    return DialogDatasetItem(context=context, belief=conversation.generated_belief,
                             raw_response=conversation.generated_responses[-1],
                             response=conversation.raw_response,
                             database=database)


def sample_to_conversation(sample):
    conversation = AuGPTConversation()
    conversation.new_user_input = sample.context[-1]
    arr, other = conversation.generated_responses, conversation.past_user_inputs
    for utt in reversed(sample.context[:-1]):
        arr.append(utt)
        arr, other = other, arr
    arr.reverse()  # past turns' responses
    other.reverse()  # past turns' user utterances
    return conversation


def format_samples(samples):
    add_labels = InsertLabelsTransformation()
    formatted = []
    for i, sample in enumerate(samples):
        sample = dataclasses.replace(sample, context=[])
        sample = add_labels(sample)
        formatted.append('=>' + sample.belief + '<|eob|>' + sample.database +
                         '<|eokb|>' + sample.response + '<|endoftext|>')
    return formatted


def generate_predictions(pipeline, dataset, output_file='predictions.txt'):
    belief_parser = BeliefParser()
    add_labels = InsertLabelsTransformation('U:', 'S:', 'D:', 'BF:', 'DOC:')
    gold_responses = []
    responses = []
    delex_gold_responses = []
    delex_responses = []
    beliefs = []
    databases = []
    documents = []
    with open(output_file, 'w+') as fout:
        d = 0
        for i, sample in enumerate(tqdm(dataset, desc='generating predictions')):
            if len(sample.context) == 1:
                d += 1
                print(f'======== dialogue {d} ========', file=fout)

            print(f'U:{sample.context[-1]}', file=fout)
            print(f'GT:{sample.raw_response}', file=fout)
            print(f'GTD:{sample.response}', file=fout)
            print(f'GBF:{format_belief(sample.raw_belief)}', file=fout)
            print(f'GDB:{format_database(sample.database)}', file=fout)
            print(f'GDOC:{sample.document}', file=fout)

            gold_responses.append(sample.raw_response)
            delex_gold_responses.append(sample.response)

            conversation = sample_to_conversation(sample)
            conversation = pipeline(conversation)

            belief = conversation.generated_belief
            raw_belief = belief_parser(belief)
            beliefs.append(raw_belief)
            database = OrderedDict((d, v[0]) for d, v in conversation.database_results.items())
            databases.append(database)
            documents.append(conversation.top_document_list)

            sample = add_labels((sample.context, belief, database, conversation.generated_responses[-1], 1,
                                 sample.raw_belief, sample.raw_response, conversation.top_document_list))
            print(sample.belief, file=fout)
            print(sample.database, file=fout)
            print(sample.document, file=fout)

            if pipeline.lexicalizer:
                print(f'R:{sample.response}', file=fout)
            else:
                print('R:', file=fout)
            print(f'RD:{conversation.raw_response}', file=fout)
            responses.append(conversation.generated_responses[-1])
            delex_responses.append(conversation.raw_response)

    return beliefs, databases, documents, responses, gold_responses, delex_responses, delex_gold_responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='wandb/offline-run-20220329_102758-2um3p4u9/files')
    parser.add_argument('--file', default='predictions.txt')
    parser.add_argument('--dataset', default='multiwoz-2.1-test')
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger()

    model_name = args.model
    dataset = load_dataset(args.dataset)

    config = AuGPTConfig.from_pretrained(model_name)
    model = AuGPTModel.from_pretrained(model_name, config=config)
    tokenizer = AuGPTTokenizer.from_pretrained(model_name)
    pipeline = AuGPTConversationalPipeline(
        model=model.module if isinstance(model, DistributedDataParallel) else model,
        tokenizer=tokenizer,
        lexicalizer=dataset.lexicalizer,
        database=dataset.database,
        docbase=dataset.docbase
    )

    # Generate
    generate_predictions(pipeline, dataset, args.file)
