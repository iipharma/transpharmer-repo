import torch
from einops import rearrange
from torch import einsum, nn


def get_rotary_emb(dim, max_seq_len, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    seq = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)
    freqs = einsum("i , j -> i j", seq, inv_freq)
    return torch.cat((freqs, freqs), dim=-1)


def apply_rotary_pos_emb(pos, t):
    def rotate_half(x):
        x = rearrange(x, "... (j d) -> ... j d", j=2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class TokenEmbedding(nn.Module):
    def __init__(self, n_embd, block_size, vocab_size, device):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.pos_emb = get_rotary_emb(n_embd, block_size, device=device) # token position
        self.val_emb = nn.Embedding(vocab_size, n_embd) # token value

    def forward(self, token):
        tok_emb = self.val_emb(token) # each index maps to a (learnable) vector
        pos_emb = self.pos_emb[:token.size(1)] # each position maps to a (learnable) vector
        tok_emb = apply_rotary_pos_emb(pos_emb, tok_emb)
        return tok_emb


class FpEmbedding(nn.Module):
    def __init__(self, fp_dim, n_embd, device):
        super().__init__()
        self.fp_dim = fp_dim
        self.device = device
        self.pair_emb = nn.Embedding(fp_dim//2, n_embd) # which fp pair
        self.bit_emb = nn.Embedding(2, n_embd) # which fp bit
        self.val_emb = nn.Embedding(2, n_embd) # fp value embed

    def forward(self, fp):
        embed = self.val_emb(fp.long())
        embed += self.pair_emb(torch.arange(self.fp_dim//2, device=self.device).repeat_interleave(2)).unsqueeze(0)
        embed += self.bit_emb(torch.arange(2, device=self.device).repeat(self.fp_dim//2)).unsqueeze(0)
        return embed


class CountEmbedding(nn.Module):
    def __init__(self, count_dim, n_embd, device):
        super().__init__()
        self.count_dim = count_dim
        self.device = device
        self.bit_emb = nn.Embedding(count_dim, n_embd) # which fp bit
        self.val_emb = nn.Embedding(100, n_embd) # count value embed

    def forward(self, count):
        embed = self.val_emb(count.long())
        embed += self.bit_emb(torch.arange(self.count_dim, device=self.device)).unsqueeze(0)
        return embed


class PropEmbedding(nn.Module):
    def __init__(self, num_props, n_embd, block_size, device, count_dim=8):
        super().__init__()
        self.num_props = num_props
        self.n_embd = n_embd
        self.block_size = block_size
        self.device = device
        self.count_dim = count_dim
        self.fp_dim = int(num_props-count_dim)
        self.type_emb = nn.Embedding(2, n_embd) # count or fp
        self.count_emb = CountEmbedding(count_dim, n_embd, self.device)
        self.fp_emb = FpEmbedding(self.fp_dim, n_embd, self.device)
        
    def forward(self, prop):
        b, t = prop.size()
        embed = torch.cat([self.count_emb(prop[:, :self.count_dim]), \
                        self.fp_emb(prop[:, self.count_dim:])], 1)
        embed += self.type_emb(torch.cat([torch.zeros(self.count_dim, dtype=torch.long, device=self.device), \
                                          torch.ones(t-self.count_dim, dtype=torch.long, device=self.device)], 0)).unsqueeze(0)
        return embed


class Embedding(nn.Module):
    def __init__(self, config, device, count_dim=8):
        super().__init__()
        self.device = device
        self.num_props = config.NUM_PROPS
        if self.num_props:
            self.type_emb = nn.Embedding(2, config.N_EMBD)
            self.prop_emb = PropEmbedding(config.NUM_PROPS, config.N_EMBD, config.MAX_LEN, self.device, count_dim)
        else:
            self.type_emb = nn.Embedding(1, config.N_EMBD)
        self.token_emb = TokenEmbedding(config.N_EMBD, config.MAX_LEN, config.VOCAB_SIZE, self.device)
        self.drop = nn.Dropout(config.EMBD_PDROP)

    def forward(self, token, prop=None):
        if prop is not None:
            embed = torch.cat([self.prop_emb(prop), self.token_emb(token)], 1)
            embed += self.type_emb(torch.cat([torch.zeros(size=[prop.size(1)], dtype=torch.long, device=self.device), \
                                            torch.ones(size=[token.size(1)], dtype=torch.long, device=self.device)], 0)).unsqueeze(0)
        else:
            embed = self.token_emb(token)
            embed += self.type_emb(torch.zeros(size=[token.size(1)], dtype=torch.long, device=self.device)).unsqueeze(0)
        embed = self.drop(embed)
        return embed