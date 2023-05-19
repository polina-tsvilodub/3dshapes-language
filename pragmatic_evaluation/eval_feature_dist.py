import torch
import numpy as np
import pickle

def count_feature_vals(ids):
    """
    Helper iterating over IDs and counting how often different
    values of the features occurred.
    """
    labels_num = np.load("../data/3dshapes_labels.npy")

    with open("../sandbox/data/3dshapes_grammar_terminals.pkl", "rb") as f:
        terminals = pickle.load(f)
    
    counter_dict = {k: {} for k in terminals.keys()}
    terminals_keys = list(terminals.keys())
    for k in terminals.keys():
       counter_dict[k] = {j: 0 for j in terminals[k].keys()}

    for i in ids:
        l = labels_num[int(i)]
        for j, ft in enumerate(l):
            counter_dict[terminals_keys[j]][ft] += 1

    return counter_dict

def main():
    pretrain_ids = torch.load("../../../../mSc_thesis/mSc-thesis/code/src/train_logs/pretrain_img_IDs_unique_3dshapes_final_pretrain.pt")
    print("ids ", pretrain_ids[:5])
    # finetune_ids_random = torch.load()
    # finetune_ids_similar = torch.load()

    counter_dict = count_feature_vals(pretrain_ids)
    print("---- counter dict -----")
    print(counter_dict)

if __name__ == "__main__":
    main()