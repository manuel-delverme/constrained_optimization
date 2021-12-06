
import torch
import torch.nn

class _DenseMultiplier(torch.nn.Module):
    def __init__(self, init, *, positive=False):
        super().__init__()
        self.weight = torch.nn.Parameter(init)
        self.positive = positive

    @property
    def shape(self):
        return self.weight.shape

    def forward(self):
        w = self.weight
        if self.positive:
            w.data = torch.relu(w).data
        return w

    def __str__(self):
        return str(self.forward().item())

class _SparseMultiplier(torch.nn.Embedding):
    def __init__(self, init, *, positive=False):
        # JCR TODO: verify this
        super().__init__(*init.shape, _weight=init, sparse=True)
        self.positive = positive

    @property
    def shape(self):
        return self.weight.shape

    def forward(self, *args, **kwargs):
        batch_multipliers = super().forward(*args, **kwargs)
        if self.positive:
            batch_multipliers.data = torch.relu(batch_multipliers).data

        return batch_multipliers
