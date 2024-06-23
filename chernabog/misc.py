"""Miscellaneous classes and functions for tensor operations."""
import torch
import numpy as np
import numpy.typing as npt
from typing import List


class CachingFields:
    """Implements caching properties with a decorator."""

    def __init__(self) -> None:
        self.cache: List[str] = []

    @staticmethod
    def cache_field(field_name: str, condition=lambda x: x is None):
        """Calculates function once and then stores its result in a cache."""

        def internal(func):
            def wrapper(self, *args):
                if not hasattr(self, field_name):
                    setattr(self, field_name, None)
                    self.cache.append(field_name)
                if condition(getattr(self, field_name)):
                    setattr(self, field_name, func(self, *args))
                return getattr(self, field_name)
            return wrapper
        return internal

    def clear_cache(self):
        for field in self.cache:
            setattr(self, field, None)


class Utils:
    """Useful functions for matrix and vectors manipulations."""

    @staticmethod
    def v_repeat(v: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Turns single vector into a matrix of vectors with the same value."""
        cols: torch.Tensor = v.repeat([1, w, 1])
        rows: torch.Tensor = cols.repeat([h, 1, 1])
        return rows

    @staticmethod
    def proj_on_plane(u: torch.Tensor, n: torch.Tensor):
        h, w, _ = u.shape
        n = Utils.v_repeat(n, w, h)

        u_n_dot = (
            u[:, :, 0] * n[:, :, 0] +
            u[:, :, 1] * n[:, :, 1] +
            u[:, :, 2] * n[:, :, 2]
        )

        n_norm2 = torch.pow(torch.norm(n, dim=2), 2)

        u_per = torch.unsqueeze(u_n_dot / n_norm2, dim=2) * n

        return u - u_per

    @staticmethod
    def m_norm(m: torch.Tensor) -> torch.Tensor:
        """Returns matrix (m, n, 3) of unit vectors with same directions
        as vectors of input matrix (m, n, 3) had."""
        return torch.nn.functional.normalize(m, p=2, dim=2)


class TensorFabric:
    """Fabric for producing tensors with a selected type on a
    selected device.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, t: List[float] | npt.NDArray) -> torch.Tensor:
        """Converts List or NDarray to torch.Tensor."""

        if isinstance(t, np.ndarray):
            result = torch.from_numpy(t).to(
                device=self.device,
                dtype=self.dtype
            )
            return result
        elif isinstance(t, list) and all(
            isinstance(x, float) or isinstance(x, int) for x in t
        ):
            return torch.tensor(t, dtype=self.dtype, device=self.device)
        else:
            raise TypeError("t must be List[float|int] or NDArray.")

    def n(self, t: torch.Tensor) -> npt.NDArray:
        """Converts torch.Tensor to torch.Tensor."""

        if not isinstance(t, torch.Tensor):
            raise TypeError("t must be torch.Tensor.")

        return t.cpu().numpy()
