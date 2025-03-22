#!/usr/bin/env python

# @Time    : 2024/8/20 10:23
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Tokenizer.py

from typing import List, Union, Optional
import torch


class Tokenizer:
    def __init__(self, device, add_special_token=True):
        self.tokens = [
            "<cls>",
            "<pad>",
            "<eos>",
            "<unk>",
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "X",
            "B",
            "U",
            "Z",
            "O",
            ".",
            "-",
            "<null_1>",
            "<mask>",
        ]
        self.all_special_ids = [0, 1, 2, 3, 31, 32]
        self.mask_token_id = 32
        self.pad_token_id = 1
        self.device = device
        self.add_special_token = add_special_token

    def __aa_2_tok__(self, sequence):
        idxs = []
        for aa in sequence:
            idx = self.tokens.index(aa)
            idxs.append(idx)
        return idxs

    def __add_special_token__(self, special):
        return [self.tokens.index(special)]

    def encode(self, peptide: List[str], hla: List[str], peptide_length: List[int], hla_length: List[int]):
        batch = len(peptide)

        if self.add_special_token:
            # lengths = lengths+
            lengths = [x + y + 3 for x, y in zip(peptide_length, hla_length)]
            max_length = max(lengths)
            forward_toks = torch.zeros([batch, max_length], dtype=torch.int64, device=self.device)
            segment_toks = torch.zeros([batch, max_length], dtype=torch.int64, device=self.device)
            for i in range(batch):
                forward_tok = self.__add_special_token__('<cls>') + self.__aa_2_tok__(
                    peptide[i]) + self.__add_special_token__('<eos>') + self.__aa_2_tok__(
                    hla[i]) + self.__add_special_token__('<eos>') + (
                                      (max_length - lengths[i]) * self.__add_special_token__('<pad>'))
                forward_toks[i] = torch.tensor(forward_tok, dtype=torch.int64, device=self.device)

                segment_tok = torch.cat((torch.zeros(peptide_length[i] + 2, dtype=torch.int64, device=self.device),
                                         torch.ones(hla_length[i] + 1, dtype=torch.int64, device=self.device),
                                         torch.full((max_length - lengths[i],), 2,
                                                    dtype=torch.int64, device=self.device)))
                segment_toks[i] = segment_tok

        else:
            lengths = [x + y for x, y in zip(peptide_length, hla_length)]
            max_length = max(lengths)
            forward_toks = torch.zeros([batch, max_length], dtype=torch.int64, device=self.device)
            segment_toks = torch.zeros([batch, max_length], dtype=torch.int64, device=self.device)
            for i in range(batch):
                forward_tok = self.__aa_2_tok__(
                    peptide[i]) + self.__aa_2_tok__(
                    hla[i]) + ((max_length - lengths[i]) * self.__add_special_token__('<pad>'))
                forward_toks[i] = torch.tensor(forward_tok, dtype=torch.int64, device=self.device)

                segment_tok = torch.cat((torch.zeros(peptide_length[i], dtype=torch.int64, device=self.device),
                                         torch.ones(hla_length[i], dtype=torch.int64, device=self.device),
                                         torch.full((max_length - lengths[i],), 2,
                                                    dtype=torch.int64, device=self.device)))
                segment_toks[i] = segment_tok

        return (forward_toks, segment_toks, lengths)
