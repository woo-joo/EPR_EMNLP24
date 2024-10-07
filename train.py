import argparse
import os.path
from time import time

import torch
from torch import nn, optim

from util import set_seed, get_dataloader, get_model
from metric import metrics, metric_funcs



def train(X, y, model, loss_func, optimizer, epoch, device):
    start = time()

    model.train()
    optimizer.zero_grad()

    X, y = X.to(device), y.to(device)
    if X.shape[1] == 2:
        pos, neg = X[:, 0], X[:, 1]
        pos, neg = model(pos), model(neg)
        y_hat = nn.Sigmoid()(pos - neg)
    else:
        y_hat = model(X)
    loss = loss_func(y_hat, y)

    loss.backward()
    optimizer.step()

    end = time()


    print(f'[Epoch {epoch:>3}] Train ({(end - start):>3.2f}s) | Loss         : {loss:>2.4f}')

    return loss



@torch.no_grad()
def eval(split, weight_path, X, y, model, retrieve, device):
    start = time()

    if split == 'Test':
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Testing with trained weights at epoch {}...'.format(checkpoint['epoch']))

    model.eval()

    X, y = X.to(device), y.to(device)
    n = int(X.shape[1] * retrieve / 100)
    X, y = X[:, :n], y[:, :n]

    y_hat = model(X)
    top_indices = torch.argsort(y_hat, descending=True)
    y = y[torch.arange(y.shape[0])[::, None].to(device), top_indices]

    results = [metric_func(y) for metric_func in metric_funcs]

    end = time()


    print(f'             {split:>4} ({(end - start):>3.2f}s) | {metrics[0]:<13}: {results[0]:>.4f}')
    for metric, result in zip(metrics[1:], results[1:]):
        print(f'                          | {metric:<13}: {result:>.4f}')

    return results



def test_bm25(_, y):
    start = time()

    results = [metric_func(y) for metric_func in metric_funcs]

    end = time()


    print(f'             Test ({(end - start):>3.2f}s) | {metrics[0]:<13}: {results[0]:>.4f}')
    for metric, result in zip(metrics[1:], results[1:]):
        print(f'                          | {metric:<13}: {result:>.4f}')

    return results



def main(args):
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'
    set_seed(args.seed)


    train_loader, val_loader, test_loader = get_dataloader(args)
    if args.model == 'BM25':
        return test_bm25(test_loader)


    model = get_model(args).to(device)
    loss_func = nn.BCELoss()
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.learning_rate)


    features = []
    if args.content: features.append(args.plm)
    if args.keyword: features.append('keyword')
    if args.expert: features.append('expert')
    features.append(str(args.retrieve))
    features = '_'.join(features)

    os.makedirs(f'results/logs/{args.model}', exist_ok=True)
    log_path = f'results/logs/{args.model}/{features}.log'
    log_file = open(log_path, 'w')
    
    os.makedirs(f'results/weights/{args.model}', exist_ok=True)
    weight_path = f'results/weights/{args.model}/{features}.pt'


    best_result, best_epoch = float('-inf'), 0
    for epoch in range(1, args.num_epoch+1):
        if epoch > best_epoch + args.patience:
            print('Early Stopped!!!')
            break

        loss = train(*train_loader, model, loss_func, optimizer, epoch, device)
        result = eval('Val', None, *val_loader, model, args.retrieve, device)
        log_file.write(f'{epoch},{loss},{result}\n')

        if best_result < result[-1]:
            best_result, best_epoch = result[-1], epoch
            checkpoint = {'state_dict': {k: v.cpu() for k, v
                                         in model.state_dict().items()},
                          'epoch'     : epoch,
                          'args'      : args}
            torch.save(checkpoint, weight_path)
    print()

    result = eval('Test', weight_path, *test_loader, model, args.retrieve, device)
    log_file.write(f'-,-,{result}\n')

    log_file.close()


    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='CAR',
                        choices=['BM25', 'Linear', 'RankNet', 'GSF', 'CAR'])
    parser.add_argument('--plm', type=str, default='V-BERT-Specter',
                        choices=['BERT', 'BERT-Specter',
                                 'SciBERT', 'SciBERT-Specter',
                                 'V-BERT', 'V-BERT-Specter'])
    parser.add_argument('--retrieve', type=int, default=25)
    parser.add_argument('--content', action='store_true')
    parser.add_argument('--keyword', action='store_true')
    parser.add_argument('--expert', action='store_true')

    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['SGD', 'Adam', 'AdamW', 'Adagrad', 'RMSprop'])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--std', type=float, default=0.02)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--num_head', type=int, default=2)
    parser.add_argument('--num_block', type=int, default=2)

    args = parser.parse_args()

    main(args)
