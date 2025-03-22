#!/usr/bin/env python

# @Time    : 2025/3/22 17:25
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Predict.py

# !/usr/bin/env python
import pandas as pd

# @Time    : 2024/7/22 9:24
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : infer.py

import pHLAmamba
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import Tokenizer
import Dataset
import argparse

parser = argparse.ArgumentParser(description='predict peptide-HLA binding with pHLA-Mamba')
parser.add_argument('--input', type=str, help='MHC-pep pairs csv with columns ["mhc_seq", "pep"]')
parser.add_argument('--mode', type=str, default="all",
                    help='"ba"/"el"/"all" for binding affinity/binding probability/both prediction')
parser.add_argument('--save_dir', type=str, default="./output/", help='output dir of result')
parser.add_argument('--batch_size', type=int, default=64, help='mini batchsize')
args = parser.parse_args()


if __name__ == "__mian__":
    args = parser.parse_args()
    device = 'cuda:0'
    tokenizer = Tokenizer.Tokenizer(add_special_token=True, device=device)
    infer_df = pd.read_csv(args.input)
    config = pHLAmamba.MambaConfig()
    model = pHLAmamba.MambaLMHeadModel(config=config, device=device)
    infer_data = Dataset.HLAPEPDataset_infer(file_path=args.input)
    infer_loader = DataLoader(infer_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.mode == "all":
        probs = []
        affs = []

        model.load_state_dict(torch.load("./model/epoch20_model.pt", map_location='cuda:0'))
        model.eval()
        with torch.no_grad():
            for step, (hla_seq, pep_seq, len_hla_seq, len_pep_seq) in enumerate(infer_loader):
                f_toks, seg_toks, pos = tokenizer.encode(pep_seq, hla_seq, len_pep_seq, len_hla_seq)
                output = model(input_ids=f_toks, seg=seg_toks, pos=pos)
                cls_logits = output.cls_logits
                affs += cls_logits.squeeze().tolist()
        infer_df['binding affinity'] = affs

        model.load_state_dict(torch.load("./model/iter29300_model.pt", map_location='cuda:0'))
        model.eval()
        with torch.no_grad():
            act = nn.Sigmoid().to(device)
            for step, (hla_seq, pep_seq, len_hla_seq, len_pep_seq) in enumerate(infer_loader):
                f_toks, seg_toks, pos = tokenizer.encode(pep_seq, hla_seq, len_pep_seq, len_hla_seq)
                output = model(input_ids=f_toks, seg=seg_toks, pos=pos)
                cls_logits = act(output.cls_logits)
                probs += cls_logits.squeeze().tolist()
        infer_df['binding probability'] = probs
        infer_df.to_csv(args.save_dir, index=False)

    if args.mode == "ba":
        affs = []

        model.load_state_dict(torch.load("./model/epoch20_model.pt", map_location='cuda:0'))
        model.eval()
        with torch.no_grad():
            for step, (hla_seq, pep_seq, len_hla_seq, len_pep_seq) in enumerate(infer_loader):
                f_toks, seg_toks, pos = tokenizer.encode(pep_seq, hla_seq, len_pep_seq, len_hla_seq)
                output = model(input_ids=f_toks, seg=seg_toks, pos=pos)
                cls_logits = output.cls_logits
                affs += cls_logits.squeeze().tolist()
        infer_df['binding affinity'] = affs
        infer_df.to_csv(args.save_dir, index=False)

    if args.model == "el":
        probs = []
        model.load_state_dict(torch.load("./model/iter29300_model.pt", map_location='cuda:0'))
        model.eval()
        with torch.no_grad():
            act = nn.Sigmoid().to(device)
            for step, (hla_seq, pep_seq, len_hla_seq, len_pep_seq) in enumerate(infer_loader):
                f_toks, seg_toks, pos = tokenizer.encode(pep_seq, hla_seq, len_pep_seq, len_hla_seq)
                output = model(input_ids=f_toks, seg=seg_toks, pos=pos)
                cls_logits = act(output.cls_logits)
                probs += cls_logits.squeeze().tolist()
        infer_df['binding probability'] = probs
        infer_df.to_csv(args.save_dir, index=False)

