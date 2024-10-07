import torch



def precision_k(y, k):
    precisions = torch.mean(y[:, :k], dim=-1)
    precision = torch.mean(precisions)
    return precision.item()



def ndcg_k(y, k):
    pos = torch.arange(1, k+1)[None, ::].to(y.device)
    dcgs = torch.sum(y[:, :k] / torch.log2(pos+1), dim=-1)
    idcgs = torch.sum(1 / torch.log2(pos+1), dim=-1)
    ndcgs = dcgs / idcgs
    ndcg = torch.mean(ndcgs)
    return ndcg.item()



metrics = ['Precision@10', 'Precision@20', 'NDCG@10', 'NDCG@20']
metric_funcs = [lambda y: precision_k(y, 10),
                lambda y: precision_k(y, 20),
                lambda y: ndcg_k(y, 10),
                lambda y: ndcg_k(y, 20)]
