import os
import random
from time import time

import numpy as np
import pandas as pd

import torch



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def get_dataloader(args):
    print(f'Preparing dataloader...')

    start = time()


    pair = pd.read_csv(f'data/pair.csv')
    pair['Topic'] = pair['Topic'].astype(str)
    session_train = pd.read_csv(f'data/session_train.csv')
    session_train['Topic'] = session_train['Topic'].astype(str)
    session_eval = pd.read_csv(f'data/session_eval.csv')
    session_eval['Topic'] = session_eval['Topic'].astype(str)

    split_strs = open(f'data/split.txt').readlines()
    split_str = random.choice(split_strs)
    split_topics = [split.split(',') for split in split_str.split()]


    session_size = 200
    dataloaders = []
    for i, topics in enumerate(split_topics):
        if i == 0: data = session_train[session_train['Topic'].isin(topics)]
        else: data = session_eval[session_eval['Topic'].isin(topics)]
        X = torch.tensor(data['ID'].values).long().reshape(-1, session_size)
        y = torch.tensor(data['Label'].values).float().reshape(-1, session_size)
        dataloaders.append((X, y))

    if args.model == 'RankNet':
        train_data = pair[pair['Topic'].isin(split_topics[0])]
        X = torch.tensor(train_data[['ID_pos', 'ID_neg']].values).long()
        y = torch.tensor([1] * len(train_data)).float()
        dataloaders[0] = (X, y)


    end = time()

    print('Done...')
    print('Consumed Time: {:>3.2f}s'.format(end - start), end='\n\n')
    
    return dataloaders



def get_model(args):
    print(f'Preparing model...')

    start = time()


    emb = torch.load(f'data/emb.pt')
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    embs = []
    if args.content: embs.append(emb[args.plm].to(device))
    if args.keyword: embs.append(emb['Keyword'].to(device))
    if args.expert: embs.append(emb['Expert'].to(device))
    assert embs != []
    args.embs = embs

    if args.model == 'Linear':
        from models.rank_net import RankNet
        model = RankNet(args)
    elif args.model == 'RankNet':
        from models.rank_net import RankNet
        model = RankNet(args)
    elif args.model == 'GSF':
        from models.gsf import GSF
        model = GSF(args)
    elif args.model == 'CAR':
        from models.car import CAR
        model = CAR(args)


    end = time()

    print('Done...')
    print('Consumed Time: {:>3.2f}s'.format(end - start), end='\n\n')

    return model
