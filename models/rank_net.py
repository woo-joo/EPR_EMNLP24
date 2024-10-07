import torch
from torch import nn

from models.modules import Ranker, GroupwiseFeedForward



class RankNet(Ranker):
    def __init__(self, args):
        super().__init__(args)

        args.group_size = 1

        self.linears = nn.ModuleList()
        for emb in args.embs:
            args.num_latent = emb.shape[-1]
            if args.num_latent > 256: self.linears.append(GroupwiseFeedForward(args))
            else: self.linears.append(nn.Linear(args.num_latent, 1))
        
        if args.model == 'Linear':
            self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weight)


    def forward(self, x):
        logit = None
        for emb, linear in zip(self.embs, self.linears):
            output = emb[x]
            output = linear(output).squeeze(-1)
            if logit is None: logit = output
            else: logit = logit + output
        logit = logit / len(self.embs)
        try: logit = self.softmax(logit)
        except AttributeError: pass
        return logit
