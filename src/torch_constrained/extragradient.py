#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.

from torch.optim import Optimizer

required = object()


class ExtraSGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))
        self.old_iterate = []

    def extrapolation(self):
        is_empty = len(self.old_iterate) == 0

        for group in self.param_groups:
            for p in group['params']:
                u = self.update(p, group)
                if is_empty:
                    self.old_iterate.append(p.data.clone())
                if u is None:
                    continue

                # Update the current parameters
                p.data.add_(u)

    def update(self, p, group):
        if p.grad is None:
            return None
        d_p = p.grad.data
        return -group['lr'] * d_p

    def step(self, closure=None):
        if len(self.old_iterate) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                i += 1
                u = self.update(p, group)
                if u is None:
                    continue

                # Update the parameters saved during the extrapolation step
                p.data = self.old_iterate[i].add_(u)

        # Free the old parameters
        self.old_iterate = []
