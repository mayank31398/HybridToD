import json
import logging
import os
from argparse import Namespace
from typing import Tuple

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoTokenizer,
                          BartForConditionalGeneration)

from dataset1 import Pad
from values1 import SPECIAL_TOKENS

logger = logging.getLogger(__name__)


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()

        self.special_tokens = {
            "bos_token": SPECIAL_TOKENS["bos"],
            "eos_token": SPECIAL_TOKENS["eos"],
            "pad_token": SPECIAL_TOKENS["pad"],
            "additional_special_tokens": [
                SPECIAL_TOKENS["domain"],
                SPECIAL_TOKENS["entity_name"],
                SPECIAL_TOKENS["user"],
                SPECIAL_TOKENS["system"],
                SPECIAL_TOKENS["kb"],
                SPECIAL_TOKENS["doc"],
                SPECIAL_TOKENS["doc_topic"],
                SPECIAL_TOKENS["doc_title"],
                SPECIAL_TOKENS["doc_body"],
                SPECIAL_TOKENS["db"],
                SPECIAL_TOKENS["slot"],
                SPECIAL_TOKENS["value"],
                SPECIAL_TOKENS["dict_slot"],
                SPECIAL_TOKENS["dict_value"],
                SPECIAL_TOKENS["list_value"],
                SPECIAL_TOKENS["entity_label"],
                SPECIAL_TOKENS["entity_explanation"],
                SPECIAL_TOKENS["task_entity_selection"],
                SPECIAL_TOKENS["task_response_generation"]
            ]
        }

        if (args.model_path and args.checkpoint and
                (args.eval_file or args.continue_training)):
            path = os.path.join(args.model_path, args.checkpoint)
            logger.info("Loading model from %s", path)

            self.config = AutoConfig.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = BartForConditionalGeneration.from_pretrained(
                path, config=self.config)
        else:
            logger.info("Loading model from %s", args.model_name)

            self.config = AutoConfig.from_pretrained(args.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(
                args.model_name, config=self.config)

            self.tokenizer.add_special_tokens(self.special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch):
        inp, input_attn_mask, out = batch
        inp = inp.cuda()
        input_attn_mask = input_attn_mask.cuda()
        out = out.cuda()

        model_outputs = self.model(
            input_ids=inp,
            attention_mask=input_attn_mask,
            labels=out
        )

        # return total loss (for multi-GPU)
        batch_size = inp.shape[0]
        return model_outputs.loss * batch_size

    def Classify(self, input_ids, attention_mask):
        bos_token_id = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["bos"])
        eos_token_id = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["eos"])
        pad_token_id = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["pad"])

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

        # run with batch_size = 1
        # sentence length
        output_length = 7
        generated = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            min_length=output_length,
            max_length=output_length,
            return_dict_in_generate=True,
            num_beams=1,
            output_scores=True,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
        vocab = generated.scores[-2]

        # "yes" = select entity
        class_labels = ["no", "yes"]

        yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        no_token_id = self.tokenizer.convert_tokens_to_ids("no")

        no = vocab[:, no_token_id].unsqueeze(1)
        yes = vocab[:, yes_token_id].unsqueeze(1)
        logits = torch.cat([no, yes], dim=-1)
        yes_prob = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().tolist()
        result = torch.argmax(logits, dim=-1).cpu().tolist()

        return result, yes_prob

    def Generate(self,
                 args: Namespace,
                 input_ids: torch.Tensor,
                 skip_special_tokens: bool) -> Tuple:
        bos_token_id = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["bos"])
        eos_token_id = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["eos"])
        pad_token_id = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["pad"])

        input_ids = input_ids.cuda()
        output = self.model.generate(
            input_ids=input_ids,
            max_length=args.generation_args["max_length"],
            min_length=args.generation_args["min_length"],
            top_k=args.generation_args["top_k"],
            top_p=args.generation_args["top_p"],
            temperature=args.generation_args["temperature"],
            num_beams=args.generation_args["num_beams"],
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )

        output_tokens = output[0]
        output_text = self.tokenizer.decode(
            output_tokens, skip_special_tokens=skip_special_tokens)

        return output_text, output_tokens.tolist()

    def GetParameters(self) -> list:
        return list(self.model.parameters())

    def SaveModel(self, args: Namespace, model_name: str):
        output_dir = os.path.join(args.model_path, model_name)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to " + output_dir)
        self.config.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
            json.dump(args.params, jsonfile, indent=4,
                      default=lambda x: str(x))

    def CollateFunction(self, batch: list):
        inp = [i["input"] for i in batch]
        out = [i["output"] for i in batch]

        padding = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["pad"])
        inp, input_attention_mask = Pad(inp, padding)
        out, _ = Pad(out, -100)

        inp = torch.tensor(inp)
        out = torch.tensor(out)
        input_attention_mask = torch.tensor(input_attention_mask)

        return inp, input_attention_mask, out


class ParallelModelWrapper(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
