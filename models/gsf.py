import torch
from torch import nn

from models.modules import Ranker, GroupwiseFeedForward



class GSF(Ranker):
    def __init__(self, args):
        super().__init__(args)

        self.group_size = args.group_size

        self.linears = nn.ModuleList()
        for emb in args.embs:
            args.num_latent = emb.shape[-1]
            if args.num_latent > 256: self.linears.append(GroupwiseFeedForward(args))
            else: self.linears.append(nn.Linear(args.num_latent * args.group_size, args.group_size))
        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weight)


    def forward(self, x):
        batch_size, num_document = x.shape[:2]
        score = torch.zeros(batch_size, num_document).to(x.device)
        for i in range(num_document - self.group_size + 1):
            group = x[:, i:i+self.group_size]
            logit = None
            for emb, linear in zip(self.embs, self.linears):
                group_emb = emb[group].flatten(start_dim=1)
                group_score = linear(group_emb)
                if logit is None: logit = group_score
                else: logit = logit + group_score
            logit = logit / len(self.embs)
            score[:, i:i+self.group_size] += logit
        score = self.softmax(score)
        return score
