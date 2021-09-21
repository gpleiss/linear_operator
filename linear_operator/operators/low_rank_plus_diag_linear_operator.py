#!/usr/bin/env python3

from __future__ import annotations
from linear_operator.utils.memoize import cached
from typing import Tuple, Union
from linear_operator.operators.sum_linear_operator import SumLinearOperator
from linear_operator.operators.root_linear_operator import RootLinearOperator
from linear_operator.operators.diag_linear_operator import DiagLinearOperator

import torch

from torch import Tensor

from .linear_operator import implements


def _rank_one_update_to_cholesky(chol_factor: Tensor, vec: Tensor) -> Tensor:
    # implements a rank one update to a cholesky factor based off of
    # \tilde A = RR' + vv'
    # https://stackoverflow.com/a/16160905
    n = vec.shape[-1]
    for k in range(n):
        r = torch.sqrt(chol_factor[..., k, k].pow(2.0) + vec[..., k].pow(2.0))
        c = r / chol_factor[..., k, k]
        s = vec[..., k] / chol_factor[..., k, k]
        chol_factor[..., k, k] = r
        if k < n - 1:
            chol_factor[..., (k + 1) :, k] = (
                chol_factor[..., (k + 1) :, k] + s * vec[..., (k + 1) :]
            ) / c
            vec[..., (k + 1) :] = (
                c * vec[..., (k + 1) :] - s * chol_factor[..., (k + 1) :, k]
            )

    return chol_factor


def _rank_p_update_to_cholesky(chol_factor: Tensor, low_rank_mat: Tensor) -> Tensor:
    # loops through p rank-1 updates to the cholesky

    mat = chol_factor.matmul(chol_factor.transpose(-1, -2))

    for p in range(low_rank_mat.shape[-1]):
        vec = low_rank_mat[..., p]
        chol_factor = _rank_one_update_to_cholesky(chol_factor, vec)
        mat += vec.unsqueeze(-1).matmul(vec.unsqueeze(-2))

    return chol_factor


class LowRankPlusDiagLinearOperator(SumLinearOperator):
    def __init__(
        self,
        *linear_operators: Tuple[Union[DiagLinearOperator, RootLinearOperator]],
        **kwargs
    ):
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
        diagonal_root = torch.diag_embed(self._diag_lo._diag.clamp(min=0.0).pow(0.5))
        updated_cholesky = _rank_p_update_to_cholesky(diagonal_root, self._root_lo._root)
        if not upper:
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
            # noqa
            l = self.cholesky(upper=upper)

        return mocked_class()

    def _diagonal(self):
        return self._diag_lo._diagonal() + self._root_lo._diagonal()

    def expand(self, batch_shape):
        # this is broke!
        return self
