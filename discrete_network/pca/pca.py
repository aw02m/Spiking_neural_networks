
import torch


class PCA:
    def __init__(self, num_components: int = 1):
        self._num_components = num_components

    def decompose(self, data: torch.Tensor, center: bool = True):
        X = torch.clone(data)
        if center:
            mean = torch.mean(X, 0)
            X -= mean.expand_as(X)
        U, _, _ = torch.svd(torch.t(X))
        out = torch.mm(X, U[:, : self._num_components])
        return out

    @property
    def num_components(self):
        return self._num_components

    @num_components.setter
    def num_components(self, value):
        self._num_components = value