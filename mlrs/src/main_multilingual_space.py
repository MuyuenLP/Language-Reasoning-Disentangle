
import os
import glob
import fire
from tqdm import tqdm

import pandas as pd
import torch
import numpy as np

from mlrs.lib._df import read_file, save_file


def build_aligner(rank: int, lan_emb):
    lan_mean_emb = {lan: np.mean(emb, axis=0) for lan, emb in lan_emb.items()}
    W = np.stack(list(lan_mean_emb.values())).T

    _, D = W.shape

    wc = W @ np.ones(D) / D
    u, s, vh = np.linalg.svd(W - wc.reshape(-1, 1) @ np.ones((1, D)))
    Ws, Gamma  = u[:, :rank], vh.T[:, :rank] @ np.diag(s[:rank])
    best_fit_W = wc.reshape(-1, 1) @ np.ones((1, D)) + Ws @ Gamma.T

    wc_new = np.linalg.pinv(best_fit_W).T @ np.ones(D)
    wc_new /= (wc_new ** 2).sum()
    prod = best_fit_W - wc_new.reshape(-1, 1) @ np.ones((1, D))

    u, s, vh = np.linalg.svd(prod)
    ws_new = u[:, :rank]

    return wc_new, ws_new



def main(
    acts_path : str,
    output_path = None,
):

    all_langs = []
    all_pt_path = []
    # search all pts in the acts_path and its subdirs
    list_dir = os.listdir(acts_path)
    for _item in list_dir:
        dir_path = os.path.join(acts_path, _item)
        if os.path.isdir(dir_path):
            all_langs.append(_item)
            pt_path = os.path.join(dir_path, "acts" , f"{_item}.pt")
            all_pt_path.append(pt_path)
    
    # load all acts
    all_acts = []
    for _item in all_pt_path:
        acts = torch.load(_item)
        all_acts.append(acts)
    
    # get the mean of all acts
    source_lan_emb = {}
    for i, _item in enumerate(all_langs):
        source_lan_emb[_item] = all_acts[i].transpose(0, 1)

    
    # build the aligner
    rank = len(all_acts) - 1
    preference_matrix, Wu_matrix = [], []
    for layer_num in tqdm(range(source_lan_emb["en"].shape[0]), desc='Computing preference matrix'):
        cur_source_lan_emb = {lang: emb[layer_num].to(torch.float).cpu().numpy() for lang, emb in source_lan_emb.items()}
        Wu, aligner = build_aligner(rank, cur_source_lan_emb)
        preference_matrix.append(torch.tensor(aligner.T))
        Wu_matrix.append(torch.tensor(Wu.T))
    preference_matrix = torch.stack(preference_matrix, dim=0)
    torch.save(preference_matrix,os.path.join(output_path ,"vector.pt"))

if __name__ == "__main__":
    fire.Fire(main)
    