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
    """
    This version now supports multiple or single swap "times" in absolute epochs.
    Pass swap_interval as a float or a list of floats. E.g.:
      - 2.0 -> swap once at 2 epochs
      - [0.1, 2.0, 5.0] -> swap at 0.1, then at 2.0, then at 5.0 epochs
    """
    def __init__(self, 
                 x: TENSOR, 
                 patch: nn.Module, 
                 num_classes: int,
                 tkn_dim: int = 256, 
                 qk_dim: int = 64, 
                 nheads: int = 12,
                 hn_mult: float = 4.0, 
                 attn_beta: float = None,
                 attn_bias: bool = False, 
                 hn_bias: bool = False,
                 time_steps: int = 1, 
                 blocks: int = 1,
                 swap_interval: Optional[Union[float, Sequence[float]]] = None,
                 swap_strategy: int = 1):
        super().__init__()
        # Process sample input to obtain dimensions.
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

        # --- NEW LOGIC FOR SWAP INTERVALS ---
        if swap_interval is not None:
            # 1) Convert a single float/int to list, or validate if a list of floats
            if isinstance(swap_interval, (float, int)):
                self.swap_intervals = [float(swap_interval)]
            elif isinstance(swap_interval, (list, tuple)):
                self.swap_intervals = [float(x) for x in swap_interval]
            else:
                raise ValueError("swap_interval must be a float, int, list, or tuple.")

            # 2) Sort them in ascending order
            self.swap_intervals.sort()

            # 3) We'll keep track of which interval index we're checking next
            self.swap_index = 0
            # The very first swap time
            self.next_swap_time = self.swap_intervals[self.swap_index]
        else:
            self.swap_intervals = None
            self.swap_index = None
            self.next_swap_time = None

        self.swap_strategy = swap_strategy

        # Maintain a record of the block order and swap history.
        self.block_order = list(range(blocks))
        self.swap_history = []

    def forward(self, x: TENSOR, alpha: float = 1.0, epoch_progress: Optional[float] = None):
        """
        epoch_progress: a float representing "how many epochs" have elapsed, e.g.
                        current_epoch + (batch_idx / num_batches).
        If provided, we check whether it's >= self.next_swap_time, and
        trigger any scheduled swaps that are due.
        """
        # Standard forward path
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

        # --- NEW LOGIC: TRIGGER SWAPS if epoch_progress is given ---
        if epoch_progress is not None and self.swap_intervals is not None:
            # We may cross multiple swap points in one go
            while self.swap_index < len(self.swap_intervals) and epoch_progress >= self.next_swap_time:
                self.swap_blocks(self.swap_strategy)
                self.swap_index += 1
                if self.swap_index < len(self.swap_intervals):
                    self.next_swap_time = self.swap_intervals[self.swap_index]
                else:
                    self.next_swap_time = None
                    break  # No more swaps scheduled

        return x

    def reset_swap_schedule(self):
        """
        If you want to reuse the same swap schedule every epoch, call this 
        at the start of each epoch. This sets your model back to the first 
        interval. 
        """
        if self.swap_intervals is not None:
            self.swap_index = 0
            self.next_swap_time = self.swap_intervals[0]

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

    def _record_swap(self, strategy):
        print(f"Swap event: strategy {strategy}, new block order: {self.block_order}")
        self.swap_history.append({
            'strategy': strategy,
            'order': self.block_order.copy()
        })

    def _swap_strategy1(self):
        n = len(self.blocks)
        if n < 2:
            return
        i = torch.randint(0, n, (1,)).item()
        j = i + 1 if i < n - 1 else 0
        blocks = list(self.blocks)
        blocks[i], blocks[j] = blocks[j], blocks[i]
        self.blocks = nn.ModuleList(blocks)
        # Update the block order.
        self.block_order[i], self.block_order[j] = self.block_order[j], self.block_order[i]
        self._record_swap(1)

    def _swap_strategy2(self):
        n = len(self.blocks)
        if n < 2:
            return
        perm = torch.randperm(n).tolist()
        new_blocks = [self.blocks[i] for i in perm]
        self.blocks = nn.ModuleList(new_blocks)
        # Update the block order.
        self.block_order = [self.block_order[i] for i in perm]
        self._record_swap(2)

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
        # Update block_order.
        self.block_order[i], self.block_order[j] = self.block_order[j], self.block_order[i]
        self._record_swap(3)

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
        # Update block_order accordingly.
        eligible_order = [self.block_order[i] for i in eligible_indices]
        permuted_order = [eligible_order[i] for i in perm]
        for idx, pos in enumerate(eligible_indices):
            self.block_order[pos] = permuted_order[idx]
        self._record_swap(4)
