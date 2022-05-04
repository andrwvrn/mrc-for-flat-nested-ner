#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: tagger_ner_dataset.py

import argparse

import torch

from typing import List, Tuple

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer



def get_labels(data_sign: str) -> List[str]:
    """
    Returns list of labels for a given dataset

    Args:
        data_sign: str
            name of the dataset

    Returns:
        List of labels
    """
    if data_sign == "zh_onto":
        return ["O", "S-GPE", "B-GPE", "M-GPE", "E-GPE",
                "S-LOC", "B-LOC", "M-LOC", "E-LOC",
                "S-PER", "B-PER", "M-PER", "E-PER",
                "S-ORG", "B-ORG", "M-ORG", "E-ORG",]
    elif data_sign == "zh_msra":
        return ["O", "S-NS", "B-NS", "M-NS", "E-NS",
                "S-NR", "B-NR", "M-NR", "E-NR",
                "S-NT", "B-NT", "M-NT", "E-NT"]
    elif data_sign == "en_onto":
        return ["O", "S-LAW", "B-LAW", "M-LAW", "E-LAW",
                "S-EVENT", "B-EVENT", "M-EVENT", "E-EVENT",
                "S-CARDINAL", "B-CARDINAL", "M-CARDINAL", "E-CARDINAL",
                "S-FAC", "B-FAC", "M-FAC", "E-FAC",
                "S-TIME", "B-TIME", "M-TIME", "E-TIME",
                "S-DATE", "B-DATE", "M-DATE", "E-DATE",
                "S-ORDINAL", "B-ORDINAL", "M-ORDINAL", "E-ORDINAL",
                "S-ORG", "B-ORG", "M-ORG", "E-ORG",
                "S-QUANTITY", "B-QUANTITY", "M-QUANTITY", "E-QUANTITY",
                "S-PERCENT", "B-PERCENT", "M-PERCENT", "E-PERCENT",
                "S-WORK_OF_ART", "B-WORK_OF_ART", "M-WORK_OF_ART", "E-WORK_OF_ART",
                "S-LOC", "B-LOC", "M-LOC", "E-LOC",
                "S-LANGUAGE", "B-LANGUAGE", "M-LANGUAGE", "E-LANGUAGE",
                "S-NORP", "B-NORP", "M-NORP", "E-NORP",
                "S-MONEY", "B-MONEY", "M-MONEY", "E-MONEY",
                "S-PERSON", "B-PERSON", "M-PERSON", "E-PERSON",
                "S-GPE", "B-GPE", "M-GPE", "E-GPE",
                "S-PRODUCT", "B-PRODUCT", "M-PRODUCT", "E-PRODUCT"]
    elif data_sign == "en_conll03":
        return ["O", "S-ORG", "B-ORG", "M-ORG", "E-ORG",
                "S-PER", "B-PER", "M-PER", "E-PER",
                "S-LOC", "B-LOC", "M-LOC", "E-LOC",
                "S-MISC", "B-MISC", "M-MISC", "E-MISC"]
    return ["0", "1"]


def load_data_in_conll(data_path: str) -> List[Tuple[List[str], List[str]]]:
    """
    Loads data in conll format

    Args:
        data_path: str
            path to dataset
    Returns:
        dataset: List[Tuple[List[str], List[str]]]
            list of tuples of lists containing words and labels of the provided dataset

    Example:
        $ cat path_to_file
          word1 label1
          word2 label2

          word3 label3
          word4 label4
          word5 label5
        >> load_data_in_conll(path_to_file)
          [([word1, word2], [label1, label2]),
           ([word3, word4, word5], [label3, label4, label5])]
    """
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        datalines = f.readlines()
    sentence, labels = [], []

    for line in datalines:
        line = line.strip()
        if len(line) == 0:
            dataset.append((sentence, labels))
            sentence, labels = [], []
        else:
            word, tag = line.split(" ")
            sentence.append(word)
            labels.append(tag)

    # append the last sentence and labels tuple if dataset file doesn't end with empty line
    if len(datalines) != 0 and len(datalines[-1]) != 0:
        dataset.append((sentence, labels))

    return dataset


class TaggerNERDataset(Dataset):
    """
    Tagger NER Dataset

    Args:
        data_path: str
            path to conll-style named entity data file
        dataset_signature: str
            name of the dataset
        tokenizer: BertTokenizer
            tokenizer object
        max_length: int
            max length of context
    """
    def __init__(self,
                 data_path: str,
                 tokenizer: AutoTokenizer,
                 dataset_signature: str,
                 max_length: int = 512,
                 pad_to_maxlen: bool = False):
        self.all_data = load_data_in_conll(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_maxlen = pad_to_maxlen
        self.pad_idx = 0
        self.cls_idx = 101
        self.sep_idx = 102
        self.label2idx = {label_item: label_idx for label_idx, label_item in enumerate(get_labels(dataset_signature))}

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        data = self.all_data[item]
        token_lst, label_lst = tuple(data)
        wordpiece_token_lst, wordpiece_label_lst = [], []

        for token_item, label_item in zip(token_lst, label_lst):
            tmp_token_lst = self.tokenizer.encode(token_item, add_special_tokens=False, return_token_type_ids=None)
            if len(tmp_token_lst) == 1:
                wordpiece_token_lst.append(tmp_token_lst[0])
                wordpiece_label_lst.append(label_item)
            else:
                len_wordpiece = len(tmp_token_lst)
                wordpiece_token_lst.extend(tmp_token_lst)
                tmp_label_lst = [label_item] + [-100 for _ in range((len_wordpiece - 1))]
                wordpiece_label_lst.extend(tmp_label_lst)

        # subtract 2 to add [CLS] and [SEP] tokens later
        if len(wordpiece_token_lst) > self.max_length - 2:
            wordpiece_token_lst = wordpiece_token_lst[:self.max_length-2]
            wordpiece_label_lst = wordpiece_label_lst[:self.max_length-2]

        wordpiece_token_lst = [self.cls_idx] + wordpiece_token_lst + [self.sep_idx]
        wordpiece_label_lst = [-100] + wordpiece_label_lst + [-100]
        # token_type_ids: segment token indices to indicate first and second portions of the inputs.
        # - 0 corresponds to a "sentence a" token
        # - 1 corresponds to a "sentence b" token
        token_type_ids = [0] * len(wordpiece_token_lst)
        # attention_mask: mask to avoid performing attention on padding token indices.
        # - 1 for tokens that are not masked.
        # - 0 for tokens that are masked.
        attention_mask = [1] * len(wordpiece_token_lst)
        is_wordpiece_mask = [1 if label_item != -100 else -100 for label_item in wordpiece_label_lst]
        wordpiece_label_idx_lst = [self.label2idx[label_item] if label_item != -100 else -100 for label_item in wordpiece_label_lst]

        return [torch.tensor(wordpiece_token_lst, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(wordpiece_label_idx_lst, dtype=torch.long),
                torch.tensor(is_wordpiece_mask, dtype=torch.long)]


def run_dataset():
    parser = argparse.ArgumentParser(description="run tagger ner dataset")
    parser.add_argument("--bert_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_sign", type=str, required=False, default='en_conll03')

    args = parser.parse_args()

    bert_path = args.bert_path
    dataset_path = args.dataset_path

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    dataset = TaggerNERDataset(data_path=dataset_path,
                               tokenizer=tokenizer,
                               dataset_signature=args.dataset_sign)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            )

    for batch in dataloader:
        print("----------------------- BATCH START -----------------------")
        for token_input_ids, token_type_ids, attention_mask, sequence_labels, is_wordpiece_mask in zip(*batch):
            token_input_ids = token_input_ids.tolist()
            print('tokens:', [tokenizer.decode(t) for t in token_input_ids])
            print('sequence_labels:', sequence_labels.numpy().tolist())
            print('is_wordpiece_mask:', is_wordpiece_mask.numpy().tolist())

            print("="*20)
            print(f"len: {len(token_input_ids)}", tokenizer.decode(token_input_ids, skip_special_tokens=False))


if __name__ == "__main__":
    run_dataset()
