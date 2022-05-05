#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: query_span_f1.py

import torch
import numpy as np

from typing import List, Sequence, Tuple

from utils.bmes_decode import bmes_decode


def query_span_f1(start_preds: torch.Tensor,
                  end_preds: torch.Tensor,
                  match_logits: torch.Tensor,
                  start_label_mask: torch.Tensor,
                  end_label_mask: torch.Tensor,
                  match_labels: torch.Tensor):
    """
    Calculates span f1 according to query-based model output

    Args:
        start_preds: torch.Tensor
            predictions of start token position,
            shape `[batch_size, seq_len]`
        end_preds: torch.Tensor
            predictions of end token position,
            shape `[batch_size, seq_len]`
        match_logits: torch.Tensor
            tensor which rows' numbers are start token positions and
            which columns' numbers are end token positions, logit at
            i-th row and j-th column will denote probability that predicted
            entity starts at i-th token and ends at j-th token in a tokens
            sequence,
            shape `[batch_size, seq_len, seq_len]`
        start_label_mask: torch.Tensor
            tensor containing mask for start tokens, contains 1s for tokens
            that are at the beginning of the word or include the whole word,
            shape `[batch_size, seq_len]`
        end_label_mask: torch.Tensor
            tensor containing mask for end tokens, contains 1s for tokens
            that are at the end of the word or include the whole word,
            shape `[batch_size, seq_len]`
        match_labels: torch.Tensor
            tensor containing 1s on rows which numbers are start token
            positions and on columns which numbers are end token positions,
            shape `[batch_size, seq_len, seq_len]`

    Returns:
        : torch.Tensor
            span-f1 counts, tensor of shape [3]: tp, fp, fn
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()
    match_labels = match_labels.bool()
    bsz, seq_len = start_label_mask.size()
    # [batch_size, seq_len, seq_len]
    match_preds = match_logits > 0
    # [batch_size, seq_len]
    start_preds = start_preds.bool()
    # [batch_size, seq_len]
    end_preds = end_preds.bool()

    match_preds = (match_preds &
                   start_preds.unsqueeze(-1).expand(-1, -1, seq_len) &
                   end_preds.unsqueeze(1).expand(-1, seq_len, -1))

    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len) &
                        end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))

    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds

    tp = (match_labels & match_preds).long().sum()
    fp = (~match_labels & match_preds).long().sum()
    fn = (match_labels & ~match_preds).long().sum()

    return torch.stack([tp, fp, fn])


def extract_nested_spans(start_preds: torch.Tensor,
                         end_preds: torch.Tensor,
                         match_preds: torch.Tensor,
                         start_label_mask: torch.Tensor,
                         end_label_mask: torch.Tensor,
                         pseudo_tag: str = "TAG"):
    """
    Calculates nested span f1

    Args:
        start_preds: torch.Tensor
            predictions of start token position,
            shape `[batch_size, seq_len]`
        end_preds: torch.Tensor
            predictions of end token position,
            shape `[batch_size, seq_len]`
        match_preds: torch.Tensor
            tensor which rows' numbers are start token positions and
            which columns' numbers are end token positions, logit at
            i-th row and j-th column will denote probability that predicted
            entity starts at i-th token and ends at j-th token in a tokens
            sequence,
            shape `[batch_size, seq_len, seq_len]`
        start_label_mask: torch.Tensor
            tensor containing mask for start tokens, contains 1s for tokens
            that are at the beginning of the word or include the whole word,
            shape `[batch_size, seq_len]`
        end_label_mask: torch.Tensor
            tensor containing mask for end tokens, contains 1s for tokens
            that are at the end of the word or include the whole word,
            shape `[batch_size, seq_len]`
        pseudo_tag: str
            dummy tag

    Returns:
        : List[Tuple[int, int, str]]
            list of tuples (start, end, tag)
    """
    start_label_mask = start_label_mask.bool()
    end_label_mask = end_label_mask.bool()

    bsz, seq_len = start_label_mask.size()

    start_preds = start_preds.bool()
    end_preds = end_preds.bool()

    match_preds = (match_preds &
                   start_preds.unsqueeze(-1).expand(-1, -1, seq_len) &
                   end_preds.unsqueeze(1).expand(-1, seq_len, -1))

    match_label_mask = (start_label_mask.unsqueeze(-1).expand(-1, -1, seq_len) &
                        end_label_mask.unsqueeze(1).expand(-1, seq_len, -1))

    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end
    match_preds = match_label_mask & match_preds
    match_pos_pairs = np.transpose(np.nonzero(match_preds.numpy())).tolist()

    return [(pos[1], pos[2], pseudo_tag) for pos in match_pos_pairs]


def extract_flat_spans(start_pred: Sequence[int],
                       end_pred: Sequence[int],
                       match_pred: Sequence[Sequence[int]],
                       label_mask: Sequence[int],
                       pseudo_tag: str = "TAG") -> List[Tuple[int, int, str]]:
    """
    Extract flat-ner spans

    Args:
        start_pred: Sequence[int]
            predictions of start token positions,
            1/True for start, 0/False for non-start,
            shape `[seq_len]`
        end_pred: Sequence[int]
            predictions of end token position,
            1/True for end, 0/False for non-end,
            shape `[seq_len]`
        match_pred: Sequence[Sequence[int]]
            2D sequence which rows' numbers are start token positions and
            which columns' numbers are end token positions,
            1/True at i-th row and j-th column means that predicted
            entity starts at i-th token and ends at j-th token
            shape `[seq_len, seq_len]`
        label_mask: Sequence[int]
            attention mask containing 1s for valid boundary,
            shape `[seq_len]`
        pseudo_tag: str
            dummy tag

    Returns:
        tags: List[Tuple[int, int, str]]
            list of tuples (start, end, tag)

    Examples:
        >> start_pred = [0, 1]
        >> end_pred = [0, 1]
        >> match_pred = [[0, 0], [0, 1]]
        >> label_mask = [1, 1]
        >> extract_flat_spans(start_pred, end_pred, match_pred, label_mask)
        [(1, 2, 'TAG')]
    """
    pseudo_input = "a"

    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"M-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"S-{pseudo_tag}"

    tags = bmes_decode([(pseudo_input, label) for label in bmes_labels])

    return [(entity.begin, entity.end, entity.tag) for entity in tags]


def remove_overlap(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Removes overlapped spans greedily for flat-ner

    Args:
        spans: List[Tuple[int, int, str]]
            list of tuples (start, end, tag)

    Returns:
        output: List[Tuple[int, int, str]]
            spans without overlap
    """
    output = []
    occupied = set()

    for start, end, tag in spans:
        if any(x in occupied for x in range(start, end+1)):
            continue
        output.append((start, end, tag))
        for x in range(start, end+1):
            occupied.add(x)

    return output
