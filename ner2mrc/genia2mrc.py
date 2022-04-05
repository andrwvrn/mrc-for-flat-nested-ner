#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: genia2mrc.py

import argparse
import os
import json


def convert_file(input_file: str, output_file: str, tag2query_file: str):
    """
    Converts GENIA data to MRC format
    """
    all_data = json.load(open(input_file))
    tag2query = json.load(open(tag2query_file))

    output = []
    origin_count = 0
    new_count = 0

    for data in all_data:
        origin_count += 1
        context = data["context"]
        label2positions = data["label"]
        for tag_idx, (tag, query) in enumerate(tag2query.items()):
            positions = label2positions.get(tag, [])
            mrc_sample = {
                "context": context,
                "query": query,
                "start_position": [int(x.split(";")[0]) for x in positions],
                "end_position": [int(x.split(";")[1]) for x in positions],
                "qas_id": f"{origin_count}.{tag_idx}"
            }
            output.append(mrc_sample)
            new_count += 1

    json.dump(output, open(output_file, "w"), ensure_ascii=False, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="convert jsonlines to mrc")
    parser.add_argument("--convert_from_dir", type=str, required=True)
    parser.add_argument("--convert_to_dir", type=str, required=True)
    parser.add_argument("--t2q_filename", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.convert_to_dir, exist_ok=True)

    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(args.convert_from_dir, f"{phase}.genia.json")
        new_file = os.path.join(args.convert_to_dir, f"mrc-ner.{phase}")
        try:
            convert_file(old_file, new_file, args.t2q_filename)
        except FileNotFoundError as e:
            print(e)
            continue


if __name__ == '__main__':
    main()
