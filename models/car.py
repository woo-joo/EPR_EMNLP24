import torch
from torch import nn

from models.modules import Ranker, TransformerEncoder



class CAR(Ranker):
    def __init__(self, args):
        super().__init__(args)

        self.encoders = nn.ModuleList()
        for emb in args.embs:
            args.num_latent = emb.shape[-1]
            if args.num_latent > 256:
                self.encoders.append(nn.Sequential(nn.Linear(args.num_latent, args.num_latent),
                                                   TransformerEncoder(args),
                                                   nn.Linear(args.num_latent, 1)))
            else:
                self.encoders.append(nn.Linear(args.num_latent, 1))
        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weight)


    def forward(self, x):
        logit = None
        for emb, encoder in zip(self.embs, self.encoders):
            output = emb[x]
            output = encoder(output)
            output = self.softmax(output.squeeze(-1))
            if logit is None: logit = output
            else: logit = logit + output
        logit = logit / len(self.embs)
        return logit
