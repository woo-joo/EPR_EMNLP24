import sys

import contextlib
import io

import random
from itertools import product
from tqdm import tqdm
from copy import copy
from tabulate import tabulate

import torch

from train import main
from metric import metrics



model_grids = {'GSF': {'group_size': [32, 64, 128]},
               'CAR': {'num_head'  : [2, 4],
                       'num_block' : [2, 4, 6]}}
train_grids = {'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
               'optimizer'    : ['SGD', 'Adam', 'AdamW', 'Adagrad', 'RMSprop']}



class ARGS:
    def __init__(self, kwargs):
        self.model = 'CAR'
        self.plm = 'V-BERT-Specter'
        self.retrieve = 25
        self.content = False
        self.keyword = False
        self.expert = False

        self.num_epoch = 1
        self.patience = 30
        self.learning_rate = 0.001
        self.optimizer = 'Adam'
        self.gpu = -1
        self.seed = 42

        self.std = 0.02
        self.dropout = 0.2
        self.group_size = 64
        self.num_head = 2
        self.num_block = 2

        for k, v in kwargs.items():
            setattr(self, k, v)



@contextlib.contextmanager
def silence_output():
    new_target = io.StringIO()
    with contextlib.redirect_stdout(new_target):
        yield



def grid_search(args):
    random.seed(args.seed)
    seeds = [random.randint(1, 10000) for _ in range(3)]

    table = []
    best = {'result': float('-inf'), 'idx': -1, 'args': None}

    grids = model_grids.get(args.model, {}) | train_grids
    params, grids = grids.keys(), list(product(*grids.values()))

    for idx, grid in tqdm(enumerate(grids)):
        for k, v in zip(params, grid):
            setattr(args, k, v)
        
        with silence_output():
            results = []
            for seed in seeds:
                args.seed = seed
                results.append(main(args))
            results = torch.tensor(results)

        means = torch.mean(results, dim=0).tolist()
        stds = torch.std(results, dim=0).tolist()
        texts = [f'{mean:.3f} ± {std:.3f}' for mean, std in zip(means, stds)]
        table.append([*grid, *texts, ''])

        if best['result'] < means[-1]:
            best = {'result': means[-1], 'idx': idx, 'args': copy(args)}

    table[best['idx']][-1] = '<- best'
    print(tabulate(table, headers=[*params, *metrics, ''], tablefmt='pretty'))



if __name__ == '__main__':
    model = sys.argv[1]
    feature = sys.argv[2]
    gpu = int(sys.argv[3])

    kwargs = {'model': model, 'gpu': gpu}
    if 'C' in feature: kwargs['content'] = True
    if 'K' in feature: kwargs['keyword'] = True
    if 'E' in feature: kwargs['expert'] = True
    args = ARGS(kwargs)

    grid_search(args)
