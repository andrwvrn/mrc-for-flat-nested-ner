#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: msra2mrc.py

import argparse
import os
import json

from typing import Optional

from utils.bmes_decode import bmes_decode


def convert_file(input_file: str, output_file: str, tag2query_file: Optional[str] = None):
    """
    Converts MSRA raw data to MRC format
    """
    if tag2query_file:
        tag2query = json.load(open(tag2query_file))
    else:
        tag2query = {}

    origin_count = 0
    new_count = 0
    mrc_samples = []

    with open(input_file) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            origin_count += 1
            src, labels = line.split("\t")
            tags = bmes_decode(char_label_list=[(char, label) for char, label in zip(src.split(), labels.split())])
            for label, query in tag2query.items():
                mrc_samples.append(
                    {
                        "context": src,
                        "start_position": [tag.begin for tag in tags if tag.tag == label],
                        "end_position": [tag.end-1 for tag in tags if tag.tag == label],
                        "query": query
                    }
                )
                new_count += 1

    json.dump(mrc_samples, open(output_file, "w"), ensure_ascii=False, sort_keys=True, indent=2)
    print(f"Converted {origin_count} samples to {new_count} samples and saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="convert jsonlines to mrc")
    parser.add_argument("--convert_from_dir", type=str, required=True)
    parser.add_argument("--convert_to_dir", type=str, required=True)
    parser.add_argument("--t2q_filename", type=str, required=False, default=None)

    args = parser.parse_args()

    os.makedirs(args.convert_to_dir, exist_ok=True)

    for phase in ["train", "dev", "test"]:
        old_file = os.path.join(args.convert_from_dir, f"{phase}.tsv")
        new_file = os.path.join(args.convert_to_dir, f"mrc-ner.{phase}")
        try:
            convert_file(old_file, new_file, args.t2q_filename)
        except FileNotFoundError as e:
            print(e)
            continue


if __name__ == '__main__':
    main()
