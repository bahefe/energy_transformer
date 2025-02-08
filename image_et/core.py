import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, Union, Sequence

TENSOR = torch.Tensor

class Lambda(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: TENSOR):
        return self.fn(x)

class Patch(nn.Module):
    def __init__(self, dim: int = 4, n: int = 32):
        super().__init__()
        self.transform = Lambda(
            lambda x: rearrange(x, "... c (h p1) (w p2) -> ... (h w) (c p1 p2)", p1=dim, p2=dim)
        )
        r = n // dim
        self.N = r ** 2
        self.revert = Lambda(
            lambda x: rearrange(
                x, "... (h w) (c p1 p2) -> ... c (h p1) (w p2)",
                h=r, w=r, p1=dim, p2=dim
            )
        )

    def forward(self, x: TENSOR, reverse: bool = False):
        return self.revert(x) if reverse else self.transform(x)

class EnergyLayerNorm(nn.Module):
    def __init__(self, in_dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(in_dim)) if bias else 0.0

    def forward(self, x: TENSOR):
        xu = x.mean(-1, keepdim=True)
        xm = x - xu
        o = xm / torch.sqrt((xm ** 2.0).mean(-1, keepdim=True) + self.eps)
        return self.gamma * o + self.bias

class PositionEncode(nn.Module):
    def __init__(self, dim: int, n: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, n, dim) * 0.02)

    def forward(self, x: TENSOR):
        return x + self.weight

class Hopfield(nn.Module):
    def __init__(self, in_dim: int, multiplier: float = 4.0, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, int(in_dim * multiplier), bias=bias)

    def forward(self, g: TENSOR):
        return -0.5 * (F.relu(self.proj(g)) ** 2).sum()

class Attention(nn.Module):
    def __init__(self, in_dim: int, qk_dim: int = 64, nheads: int = 12, beta: float = None, bias: bool = False):
        super().__init__()
        self.h, self.d = nheads, qk_dim
        self.beta = beta or 1.0 / (qk_dim ** 0.5)
        self.wq = nn.Parameter(torch.randn(nheads, qk_dim, in_dim) * 0.02)
        self.wk = nn.Parameter(torch.randn(nheads, qk_dim, in_dim) * 0.02)
        self.bq = nn.Parameter(torch.zeros(qk_dim)) if bias else None
        self.bk = nn.Parameter(torch.zeros(qk_dim)) if bias else None

    def forward(self, g: TENSOR, mask: TENSOR = None):
        q = torch.einsum("...kd,...hzd->...khz", g, self.wq)
        k = torch.einsum("...kd,...hzd->...khz", g, self.wk)
        if self.bq is not None:
            q += self.bq
            k += self.bk
        A = torch.einsum("...qhz,...khz->...hqk", q, k)
        if mask is not None:
            A *= mask
        return (-1.0 / self.beta) * torch.logsumexp(self.beta * A, dim=-1).sum()

class ETBlock(nn.Module):
    def __init__(self, in_dim: int, qk_dim: int = 64, nheads: int = 12, 
                 hn_mult: float = 4.0, attn_beta: float = None, 
                 attn_bias: bool = False, hn_bias: bool = False):
        super().__init__()
        self.hn = Hopfield(in_dim, hn_mult, hn_bias)
        self.attn = Attention(in_dim, qk_dim, nheads, attn_beta, attn_bias)

    def energy(self, g: TENSOR, mask: TENSOR = None):
        return self.attn(g, mask) + self.hn(g)

    def forward(self, g: TENSOR, mask: TENSOR = None):
        return self.energy(g, mask)

class ET(nn.Module):
    def __init__(self, x: TENSOR, patch: nn.Module, num_classes: int,
                 tkn_dim: int = 256, qk_dim: int = 64, nheads: int = 12,
                 hn_mult: float = 4.0, attn_beta: float = None,
                 attn_bias: bool = False, hn_bias: bool = False,
                 time_steps: int = 1, blocks: int = 1,
                 swap_interval: int = None, swap_strategy: int = 1):
        super().__init__()
        # Process a sample input x through patch to determine dimensions.
        x = patch(x)
        _, n, d = x.shape
        
        self.patch = patch
        self.encode = nn.Sequential(nn.Linear(d, tkn_dim))
        self.decode = nn.Sequential(
            EnergyLayerNorm(tkn_dim),
            nn.Linear(tkn_dim, num_classes)
        )
        self.pos = PositionEncode(tkn_dim, n + 1)
        self.cls = nn.Parameter(torch.randn(1, 1, tkn_dim))
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                EnergyLayerNorm(tkn_dim),
                ETBlock(tkn_dim, qk_dim, nheads, hn_mult, attn_beta, attn_bias, hn_bias)
            ]) for _ in range(blocks)
        ])
        self.K = time_steps

        # New attributes for automatic block swapping.
        # If swap_interval is not None, we swap blocks every swap_interval forward passes.
        self.swap_interval = swap_interval
        self.swap_strategy = swap_strategy
        self.iteration = 0

    def forward(self, x: TENSOR, alpha: float = 1.0):
        x = self.patch(x)
        x = self.encode(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = self.pos(x)
        
        for norm, et in self.blocks:
            for _ in range(self.K):
                g = norm(x)
                dEdg, E = torch.func.grad_and_value(et)(g)
                x = x - alpha * dEdg
        
        x = self.decode(x[:, 0])  # CLS token classification
        
        # Update the iteration counter and swap blocks if needed.
        if self.swap_interval is not None:
            self.iteration += 1
            if self.iteration % self.swap_interval == 0:
                self.swap_blocks(self.swap_strategy)
        
        return x

    def swap_blocks(self, strategy: int):
        if strategy == 1:
            self._swap_strategy1()
        elif strategy == 2:
            self._swap_strategy2()
        elif strategy == 3:
            self._swap_strategy3()
        elif strategy == 4:
            self._swap_strategy4()
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def _swap_strategy1(self):
        n = len(self.blocks)
        if n < 2:
            return
        i = torch.randint(0, n, (1,)).item()
        j = i + 1 if i < n - 1 else 0
        blocks = list(self.blocks)
        blocks[i], blocks[j] = blocks[j], blocks[i]
        self.blocks = nn.ModuleList(blocks)

    def _swap_strategy2(self):
        n = len(self.blocks)
        if n < 2:
            return
        perm = torch.randperm(n).tolist()
        new_blocks = [self.blocks[i] for i in perm]
        self.blocks = nn.ModuleList(new_blocks)

    def _swap_strategy3(self):
        n = len(self.blocks)
        if n < 3:
            return
        eligible_indices = list(range(1, n-1))
        eligible_count = len(eligible_indices)
        if eligible_count < 2:
            return
        idx = torch.randint(0, eligible_count, (1,)).item()
        i = eligible_indices[idx]
        next_idx = (idx + 1) % eligible_count
        j = eligible_indices[next_idx]
        blocks = list(self.blocks)
        blocks[i], blocks[j] = blocks[j], blocks[i]
        self.blocks = nn.ModuleList(blocks)

    def _swap_strategy4(self):
        n = len(self.blocks)
        if n < 3:
            return
        eligible_indices = list(range(1, n-1))
        eligible_count = len(eligible_indices)
        if eligible_count < 2:
            return
        perm = torch.randperm(eligible_count).tolist()
        eligible_blocks = [self.blocks[i] for i in eligible_indices]
        permuted_blocks = [eligible_blocks[i] for i in perm]
        blocks = list(self.blocks)
        for idx, pos in enumerate(eligible_indices):
            blocks[pos] = permuted_blocks[idx]
        self.blocks = nn.ModuleList(blocks)
