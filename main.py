import os
import random
import sys
import time

import numpy as np
import torch
from loguru import logger as lg
from torch import tensor

from datasets.datasets import get_dataset
from model.progress import k_folds_cross_validation
from param_parser import get_parser


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_n_filter_triples(dataset, feat_str, gfn_add_ak3=True,
                            gfn_reall=False, reddit_odeg10=True,
                            dd_odeg10_ak1=True):
    triples_filtered = []

    if dataset in ['REDDIT-BINARY']:
        feat_str += '+ak3'
    if dataset in ['COLLAB']:
        feat_str += '+randd0.2'

    if reddit_odeg10 and dataset in [
            'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
        feat_str = feat_str.replace('odeg100', 'odeg10')

    if dd_odeg10_ak1 and dataset in ['DD']:
        feat_str = feat_str.replace('odeg100', 'odeg10')
        feat_str = feat_str.replace('ak3', 'ak1')
    triples_filtered.append((dataset, feat_str))
    return triples_filtered
# def main(weight_decay, lr, epochs, batch_size, k,k2, capsule_dimensions, capsule_num,
#          lr_decay_step_size, theta, dropout, threshold):
def main(seed):
    args = get_parser()
    args.seed = seed
    set_seed(args.seed)


    lg.info(vars(args))


    dataset = args.dataset
    feat_str = 'deg+odeg100'
    sys.stdout.flush()
    for (dataset_name, feat_str) in create_n_filter_triples(dataset,feat_str):
        print(feat_str)
        dataset, max_node_num = get_dataset(name=dataset_name,feat_str= feat_str, root=args.data_root)

        print('Data: {}, Max Node Num: {}'.format(dataset, max_node_num))

        test_acc_valacc, test_acc_valloss = k_folds_cross_validation(args, dataset=dataset, max_node_num=max_node_num,
                                                                    folds=10)
    # return test_acc_valacc, test_acc_valloss
    return test_acc_valacc

