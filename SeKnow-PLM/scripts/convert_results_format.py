import argparse
import json
from evaluate_multiwoz import parse_predictions


def UpdateTurns(dataset: dict, responses: list):
    index = 0
    for dialog in dataset["dialogues"]:
        for turn in dialog["items"]:
            if ("response_generation" in turn):
                turn["response_generation"]["generated_response"] = responses[index]
                turn["response_generation"]["domain"] = None
                turn["response_generation"]["entity_name"] = None

                del turn["entity_selection"]
                del turn["oracle_entities"]
                del turn["reduced_universe"]
                index += 1


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_file')
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    responses, _, _, _, _, _, _ = parse_predictions(args.input_file)

    labels_file = json.load(open(args.labels_file, "r"))
    UpdateTurns(labels_file, responses)
    json.dump(labels_file, open(args.output_file, "w"), indent=4)
