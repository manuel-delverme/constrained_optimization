
import torch
import torch.nn

class _SparseMultiplier(torch.nn.Embedding):
    def __init__(self, hi, *, positive=False, dtype=None):
        super().__init__(*hi.shape, _weight=torch.zeros(hi.shape, device=hi.device, dtype=dtype), sparse=True)
        self.positive = positive

    @property
    def shape(self):
        return self.weight.shape

    def forward(self, *args, **kwargs):
        batch_multipliers = super().forward(*args, **kwargs)
        if self.positive:
            batch_multipliers.data = torch.relu(batch_multipliers).data

        return batch_multipliers


class _DenseMultiplier(torch.nn.Module):
    def __init__(self, hi, *, positive=False, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hi.shape, device=hi.device, dtype=dtype))
        self.positive = positive

    @property
    def shape(self):
        return self.weight.shape

    def forward(self, h=None):
        w = self.weight
        if self.positive:
            w.data = torch.relu(w).data
        return w

    def __str__(self):
        return str(self.forward().item())
