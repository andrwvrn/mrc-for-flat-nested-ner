import argparse
import json
import os

import jsonlines


def convert_file(input_file: str, output_file: str, tag2query_file: str):
    """
    Converts files from jsonl to mrc format

    Input file format:
    
        {"text": "Ежедневно в мессенджере Telegram отправляется два миллиарда
                  сообщений, что вдвое больше, чем в конце прошлого года.
                  Первый клиент для Telegram на iOS был представлен в
                  августе 2013 года",
        "entities": [
                     {"label": "org_name", "start": 24, "end": 32},
                     {"label": "org_name", "start": 134, "end": 142},
                     {"label": "date", "start": 168, "end": 185}
                    ]},
        {...}
        
        
    Output file format:
    
        [
         {
            "context": "Ежедневно в мессенджере Telegram отправляется два миллиарда
                        сообщений, что вдвое больше, чем в конце прошлого года.
                        Первый клиент для Telegram на iOS был представлен в
                        августе 2013 года",
            "end_position": [32, 142],
            "entity_label": "org_name",
            "query": "org_name",
            "span_position": ["24;32", "134;142"],
            "start_position": [24, 134]
         },
         {
            "context": "Ежедневно в мессенджере Telegram отправляется два миллиарда
                        сообщений, что вдвое больше, чем в конце прошлого года.
                        Первый клиент для Telegram на iOS был представлен в
                        августе 2013 года",
            "end_position": [185],
            "entity_label": "date",
            "query": "date",
            "span_position": ["168;185"],
            "start_position": [168]
         },
         {...}
        ]

    Args:
        input_file: str
            source markup file path
        output_file: str
            output markup file path
        tag2query_file: str
           path to file that maps tag to its description
    """
    tag2query = json.load(open(tag2query_file))
    origin_count = 0
    new_count = 0
    output = []

    with jsonlines.open(input_file) as lines:
        for line in lines:
            origin_count += 1
            context, entities = line["text"], line["entities"]
            context_entities = {}

            for e in entities:
                label = e["label"]
                start_pos, end_pos = e["start"], e["end"]

                if label not in context_entities:
                    new_count += 1
                    context_entities[label] = {
                        "context": context,
                        "end_position": [end_pos],
                        "entity_label": tag2query.get(label, label),
                        "query": label,
                        "span_position": [f"{start_pos};{end_pos}"],
                        "start_position": [start_pos]
                    }
                else:
                    output_entity = context_entities[label]

                    output_entity["start_position"].append(start_pos)
                    output_entity["end_position"].append(end_pos)
                    output_entity["span_position"].append(f"{start_pos};{end_pos}")

            output.extend(list(context_entities.values()))

    json.dump(output, open(output_file, "w"), ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="convert jsonlines to mrc")
    parser.add_argument("--convert_from_dir", type=str, required=True)
    parser.add_argument("--convert_to_dir", type=str, required=True)
    parser.add_argument("--t2q_filename", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.convert_to_dir, exist_ok=True)

    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(args.convert_from_dir, f"{phase}.jsonl")
        new_file = os.path.join(args.convert_to_dir, f"mrc-ner.{phase}")
        try:
            convert_file(old_file, new_file, args.t2q_filename)
        except FileNotFoundError as e:
            print(e)
            continue


if __name__ == '__main__':
    main()
