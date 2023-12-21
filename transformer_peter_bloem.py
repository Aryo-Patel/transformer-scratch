import torch
from torch import nn
import torch.nn.functional as F

def basic_self_attn():
    # basic self attention
    # assume an input tensor x with size (b, t, k) ... b = minibatch size, t = # input sequences, k = encoding_dimension
    x = ...

    unnorm_weights = torch.bmm(x, x.transpose(1, 2))
    # resultant shape = (b, t, t) --> softmax over the nested rows
    norm_weights = F.softmax(unnorm_weights, dim = 2)
    # turn (b, t, t) * (b, t, k) --> (b, t, k)
    output = torch.bmm(norm_weights, x)

    # Linear transformation with matrix W --> 

class SelfAttention(nn.Module):
    def __init__(self, k = 256, heads = 4, mask = False):
        super(SelfAttention, self).__init__()
        assert k % heads == 0
        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k, bias = False)
        self.toqueries = nn.Linear(k, k, bias = False)
        self.tovalues = nn.Linear(k, k, bias = False)
        self.unifyheads = nn.Linear(k, k)


    def forward(self, x):
        """
        Loops bad -- resizing good 
        """
        # x size --> (b, t, k)
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        queries = queries.view(b, t, h, k//h)
        keys = keys.view(b, t, h, k//h)
        values = values.view(b, t, h, k//h)

        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k//h)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k//h)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k//h)

        dotp = torch.bmm(queries, keys.transpose(1, 2))
        dotp = dotp/(k**0.5)
        dotp = F.softmax(dotp, dim = 2)

        self_attn = torch.bmm(dotp, values).view(b, h, t, k//h)

        out = self_attn.transpose(1, 2).contiguous().view(b, t, k//h * h)
        return self.unifyheads(out)

    # At point of creating the transformer block
    print("i need a commit")

if __name__ == "__main__":
    inp = torch.rand((5, 12, 256))

    self_attn = SelfAttention()

    self_attn(inp)

        