#!/usr/bin/env python

# @Time    : 2024/8/20 10:23
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Dadaset.py

from torch.utils.data import Dataset
import pandas as pd


class HLAPEPDataset(Dataset):
    def __init__(self, file_path, mhc_dict_path):
        self.data = pd.read_csv(file_path)
        self.dict = pd.read_csv(mhc_dict_path)
        self.data.drop_duplicates(inplace=True)
        self.dataset = pd.merge(self.data, self.dict, how="left", left_on="hla", right_on="hla")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        hla_seq = self.data['mhc_seq'][idx]
        pep_seq = self.data['pep'][idx]
        label = self.data['label'][idx]
        len_hla_seq = len(hla_seq)
        len_pep_seq = len(pep_seq)

        return hla_seq, pep_seq, label, len_hla_seq, len_pep_seq

class HLAPEPDataset_infer(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hla_seq = self.data['mhc_seq'][idx]
        pep_seq = self.data['pep'][idx]
        len_hla_seq = len(hla_seq)
        len_pep_seq = len(pep_seq)

        return hla_seq, pep_seq, len_hla_seq, len_pep_seq