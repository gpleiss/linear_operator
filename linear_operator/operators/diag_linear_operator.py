#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..utils.broadcasting import _collapse_batch_and_singleton_dimensions
from ..utils.memoize import cached
from .dense_linear_operator import DenseLinearOperator
from .linear_operator import LinearOperator
from .root_linear_operator import RootLinearOperator


class DiagLinearOperator(LinearOperator):
    """
    Diagonal linear operator. Supports arbitrary batch sizes.

    For example, supplying a `b1 x ... x bk x n` torch.Tensor will represent
    a `b1 x ... x bk`-sized batch of `n x n` diagonal matrices

    :param torch.Tensor diag: The diagonal.
    """

    def __init__(self, diag):
        super().__init__(diag)
        self._diag = diag

    def _bilinear_derivative(self, left_vecs, right_vecs):
        res = (left_vecs * right_vecs).sum(dim=-1)
        return (_collapse_batch_and_singleton_dimensions(res, target_shape=self._diag.shape),)

    def _diagonal(self):
        return self._diag

    def _matmul(self, rhs):
        if isinstance(rhs, DenseLinearOperator):
            return DenseLinearOperator(self._diag.unsqueeze(-1) * rhs.tensor)
        return self._diag.unsqueeze(-1) * rhs

    def _size(self):
        return self._diag.shape + self._diag.shape[-1:]

    @cached(name="symeig")
    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LinearOperator]]:
        evals = self._diag
        if eigenvectors:
            evecs = DiagLinearOperator(torch.ones_like(evals))
        else:
            evecs = None
        return evals, evecs

    @cached
    def to_dense(self):
        if self._diag.dim() == 0:
            return self._diag
        return torch.diag_embed(self._diag)

    def __add__(self, other):
        if isinstance(other, DiagLinearOperator):
            return self.__class__(self._diag + other._diag)
        elif isinstance(other, RootLinearOperator):
            from .low_rank_plus_diag_linear_operator import LowRankPlusDiagLinearOperator
            return LowRankPlusDiagLinearOperator(self, other)
        return super().__add__(other)
