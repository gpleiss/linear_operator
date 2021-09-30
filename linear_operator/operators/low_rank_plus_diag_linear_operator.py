#!/usr/bin/env python3

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

from linear_operator.operators.diag_linear_operator import DiagLinearOperator
from linear_operator.operators.root_linear_operator import RootLinearOperator
from linear_operator.operators.sum_linear_operator import SumLinearOperator
from linear_operator.utils.memoize import cached

from .linear_operator import implements

# thanks jake for writing this


def givens(x, y):
    r = torch.sqrt(x ** 2 + y ** 2)
    s = -y / r
    c = x / r
    return s, c, r


def rank_k_update_householder(R, u):
    """
    Fully vectorized rank k update of a Cholesky decomposition R.

    This function finds an upper triangular matrix R^{*} so that $R^{*\top}R^{*} = R^{\top}R + U^{\top}U$.

    The high level idea is to run QR with Householder transformations on the matrix [R ; U].

    The update takes O(n^2) time. To achieve this, we exploit the fact that
    the Householder reflection H used to zero out a column of the form [a, 0, 0, ..., 0, u_1, ..., u_k]
    only has n + 2k + k^2 non-zero entries, and in particular each iteration only requires us to update U
    and one row of R.

    Args:
        - R (n x n): Initial cholesky decomposition, upper triangular
        - U (k x n): Set of vectors to update with.
    Returns:
        - Upper triangular square root of R'R + U'U.
    """
    n = R.size(-1)  # Size of matrix
    k = u.size(-2)  # Rank of update
    Ru = torch.cat((R, u), dim=-2)

    for i in range(n):
        x = Ru[i:, i]  # Column we are working on

        # Only non-zero entries in x the diagonal and the new rows (we never work on rows above the diagonal)
        norm_x = (x[0] ** 2 + (x[-k:] ** 2).sum()).sqrt()

        # Construct v vector corresponding to householder reflection to zero out last k entries of x.
        e_0 = torch.zeros_like(x)
        e_0[0] = norm_x
        v = x - e_0
        norm_v = (v[0] ** 2 + (v[-k:] ** 2).sum()).sqrt()
        v = v / norm_v

        top_left_bit = 1 - 2 * v[0] * v[0]
        top_right_bit = -2 * v[0] * v[-k:]
        bottom_left_bit = top_right_bit
        bottom_right_bit = torch.eye(k, device=R.device, dtype=R.dtype) - 2 * v[-k:].unsqueeze(-1) @ v[-k:].unsqueeze(
            -2
        )

        # Update for row of R
        row_update = top_left_bit * Ru[i, i:] + (top_right_bit.unsqueeze(-1) * Ru[-k:, i:]).sum(-2)

        # Update for u
        u_update = bottom_left_bit.unsqueeze(-1) * Ru[i, i:] + bottom_right_bit @ Ru[-k:, i:]

        Ru[i, i:] = row_update
        Ru[-k:, i:] = u_update

    return Ru[:-k, :]


def rank_1_update_givens(R, u):
    """
    Args:
        R - Upper triangular matrix, n x n
        u - low rank vector, 1 x n

    Returns a root of R'R + u'u
    """
    if u.size(-2) > 1:
        raise RuntimeError("Givens rotations can only be used for rank 1 updates.")

    GRu = torch.cat((R, u), dim=-2)
    for i in range(GRu.size(-1)):
        x = GRu[i, i]
        y = GRu[-1, i]
        s, c, r = givens(x, y)

        # row i <- c * row_i - s * u
        row_i_update = c * GRu[i, i:] - s * GRu[-1, i:]

        # u <- s * row_i + c * u
        u_row_update = s * GRu[i, i:] + c * GRu[-1, i:]

        GRu[i, i:] = row_i_update
        GRu[-1, i:] = u_row_update

    GRu = GRu[:-1, :]
    return GRu


class LowRankPlusDiagLinearOperator(SumLinearOperator):
    def __init__(self, *linear_operators: Tuple[Union[DiagLinearOperator, RootLinearOperator]], **kwargs):
        super().__init__(*linear_operators)
        if isinstance(linear_operators[0], DiagLinearOperator):
            self._diag_lo = linear_operators[0]
            assert isinstance(linear_operators[1], RootLinearOperator)
            self._root_lo = linear_operators[1]
        elif isinstance(linear_operators[1], DiagLinearOperator):
            self._diag_lo = linear_operators[1]
            assert isinstance(linear_operators[0], RootLinearOperator)
            self._root_lo = linear_operators[0]
        else:
            raise NotImplementedError(
                "Must provide a DiagLinearOperator and a RootLinearOperator to LowRankPlusDiagLinearOperator"
            )

        del self.linear_operators
        self.linear_operators = tuple([self._diag_lo, self._root_lo])

    @cached(name="cholesky")
    def _cholesky(self, upper: bool = False) -> Tensor:
        # cholesky of diagonal is the sqrt of the matrix
        # it is also upper triangular be definition
        diagonal_root = torch.diag_embed(self._diag_lo._diag.clamp(min=1e-6).pow(0.5))
        updated_cholesky = rank_k_update_householder(R=diagonal_root, u=self._root_lo._root.transpose(-1, -2),)
        if upper:
            return updated_cholesky
        else:
            return updated_cholesky.transpose(-1, -2)

    @implements(torch.linalg.cholesky)
    def cholesky(self, upper: bool = False) -> Tensor:
        return self._cholesky(upper=upper)

    @implements(torch.linalg.cholesky_ex)
    def cholesky_ex(self, upper: bool = False) -> Tensor:
        class mocked_class:
            info = torch.zeros(1)
            l = self.cholesky(upper=upper)  # noqa

        return mocked_class()

    def _diagonal(self):
        return self._diag_lo._diagonal() + self._root_lo._diagonal()

    def expand(self, batch_shape):
        # this is broke!
        return self
