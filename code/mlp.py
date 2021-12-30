import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, dropout):
        super(MLP, self).__init__()
        # Linear + BN + ReLU + Dropout
        f_layer = lambda in_dim, out_dim: [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout)]
        # in_dim -> 128 -> 32 -> out_dim
        self.clf = nn.Sequential(*(f_layer(in_dim, 128) + f_layer(128, 32) + [nn.Linear(32, 2)])) #

    def init_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.detach().normal_(-0.1, 0.1)
                if m.bias is not None:
                    m.bias.detach().normal_(-0.1, 0.1)

    def forward(self, x, extract=False):
        x = self.clf[:-1](x)
        if not extract: x = self.clf[-1](x)
        return x

def setup_model(args):
    model = MLP(args.in_dim, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l_2)
    return model, optimizer