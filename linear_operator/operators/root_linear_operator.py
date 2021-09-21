#!/usr/bin/env python3

from __future__ import annotations

from .linear_operator import LinearOperator


class RootLinearOperator(LinearOperator):
    """
    Linear operator that represents the square root of a matrix. Supports
    arbitrary batch sizes.

    :param torch.Tensor root: the square root.
    """

    def __init__(self, root):
        super().__init__(root)
        self._root = root

    def _matmul(self, rhs):
        return self._root.matmul(self._root.transpose(-1, -2).contiguous().matmul(rhs))

    def _t_matmul(self, rhs):
        # Matrix is symmetric
        return self._matmul(rhs)

    def _diagonal(self):
        return (self._root ** 2).sum(-1)

    def _size(self):
        return self._root.shape[:-1] + (self._root.shape[-2],)

    def __add__(self, other):
        from linear_operator.operators.diag_linear_operator import DiagLinearOperator

        if isinstance(other, DiagLinearOperator):
            from .low_rank_plus_diag_linear_operator import LowRankPlusDiagLinearOperator

            return LowRankPlusDiagLinearOperator(self, other)
        return super().__add__(other)
