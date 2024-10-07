import torch
from torch import nn



class Ranker(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.std = args.std
        self.embs = args.embs


    def init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight.data, mean=0.0, std=self.std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight.data)
            nn.init.zeros_(module.bias.data)
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias.data)



class GroupwiseFeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.w1 = nn.Linear(args.num_latent * args.group_size, args.num_latent // 4)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(args.num_latent // 4, args.group_size)
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, x):
        y = self.w2(self.relu(self.w1(x)))
        y = self.dropout(y)

        return y



class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.w1 = nn.Linear(args.num_latent, 4 * args.num_latent)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(4 * args.num_latent, args.num_latent)

        self.dropout = nn.Dropout(args.dropout)
        self.layernorm = nn.LayerNorm(args.num_latent, eps=1e-12)


    def forward(self, x):
        y = self.w2(self.relu(self.w1(x)))
        y = self.layernorm(x + self.dropout(y))

        return y



class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_head = args.num_head
        self.head_size = args.num_latent // args.num_head
        self.num_latent = args.num_latent

        self.w_Q = nn.Linear(args.num_latent, args.num_latent)
        self.w_K = nn.Linear(args.num_latent, args.num_latent)
        self.w_V = nn.Linear(args.num_latent, args.num_latent)

        self.w = nn.Linear(args.num_latent, args.num_latent)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm = nn.LayerNorm(args.num_latent, eps=1e-12)


    def split_head(self, x):
        new_shape = x.shape[:2] + (self.num_head, self.head_size)
        x = x.view(*new_shape)
        x = torch.permute(x, (0, 2, 1, 3))

        return x
    

    def concat_head(self, x):
        x = torch.permute(x, (0, 2, 1, 3)).contiguous()
        new_shape = x.shape[:2] + (self.num_latent,)
        x = x.view(*new_shape)

        return x


    def forward(self, x):
        Q = self.split_head(self.w_Q(x))
        K = self.split_head(self.w_K(x))
        V = self.split_head(self.w_V(x))

        attn_score = torch.matmul(Q, torch.transpose(K, -1, -2))
        attn_score = attn_score / (self.head_size ** 0.5)

        attn_prob = self.softmax(attn_score)
        attn_prob = self.dropout(attn_prob)

        context = torch.matmul(attn_prob, V)
        context = self.concat_head(context)

        y = self.w(context)
        y = self.layernorm(x + self.dropout(y))

        return y



class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.layer = MultiHeadAttention(args)
        self.ffn = FeedForward(args)


    def forward(self, x):
        return self.ffn(self.layer(x))



class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.blocks = nn.ModuleList([TransformerBlock(args)
                                     for _ in range(args.num_block)])


    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        return x
