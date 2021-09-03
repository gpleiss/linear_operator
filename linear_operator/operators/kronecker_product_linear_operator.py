#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.memoize import cached
from .dense_linear_operator import DenseLinearOperator
from .linear_operator import LinearOperator


def _kron_diag(outer_diag, inner_diag):
    # Given the diagonals of outer and inner,
    # returns the diagonal of outer \kron inner
    diag = outer_diag.unsqueeze(-1) * inner_diag.unsqueeze(-2)
    diag = diag.view(*diag.shape[:-2], -1)
    return diag


class KroneckerProductLinearOperator(LinearOperator):
    r"""
    Represents the Kronecker product :math:`\mathbf A \otimes \mathbf B`, where :math:`\mathbf A, \mathbf B`
    are LinearOperators.

    If :attr:`self` is (... x M x N) and :attr:`other` is (... x P x Q), the resulting linear operator
    will have size (... x MP x QN).

    :param outer: :math:`\mathbf A`
    :type outer: ~linear_operator.operators.LienarOperator or torch.Tensor
    :param inner: :math:`\mathbf B`
    :type inner: ~linear_operator.operators.LienarOperator or torch.Tensor
    """

    def __init__(self, outer: Union[LinearOperator, torch.Tensor], inner: Union[LinearOperator, torch.Tensor]):
        # Wrap outer/inner as DenseLinearOperator if they are torch.Tensors
        if torch.is_tensor(outer):
            outer = DenseLinearOperator(outer)
        if torch.is_tensor(inner):
            inner = DenseLinearOperator(inner)

        super().__init__(outer, inner)
        self.outer = outer
        self.inner = inner

    def _diagonal(self) -> torch.Tensor:
        return _kron_diag(self.outer._diagonal(), self.inner._diagonal())

    def _logdet(self):
        outer_logdet = self.outer.logdet()
        inner_logdet = self.inner.logdet()
        return self.inner.size(-1) * outer_logdet + self.outer.size(-1) * inner_logdet

    def _matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        # For algorithm, see Algorithm 15 (page 137) of http://mlg.eng.cam.ac.uk/pub/pdf/Saa11.pdf
        X = rhs.view(*rhs.shape[:-2], self.outer.size(-1), -1)
        Z = self.outer._matmul(X)
        Z = Z.view(*Z.shape[:-2], self.outer.size(-2), -1, rhs.size(-1)).transpose(-3, -2)

        X = Z.contiguous().view(*Z.shape[:-3], self.inner.size(-1), -1)
        Z = self.inner._matmul(X)
        Z = Z.view(*Z.shape[:-2], self.inner.size(-2), -1, rhs.size(-1)).transpose(-3, -2)

        X = Z.contiguous().view(*Z.shape[:-3], -1, rhs.size(-1))
        return X

    def _size(self):
        batch_shape = _mul_broadcast_shape(self.outer.shape[:-2], self.inner.shape[:-2])
        matrix_shape = torch.Size(
            [
                outer_size * inner_size
                for outer_size, inner_size in zip(self.outer.matrix_shape, self.inner.matrix_shape)
            ]
        )
        return batch_shape + matrix_shape

    @cached(name="symeig")
    def _symeig(self, eigenvectors: bool = False) -> Tuple[Tensor, Optional[LinearOperator]]:
        evals, evecs = [], []
        evals_outer, evecs_outer = self.outer._symeig(eigenvectors=eigenvectors)
        evals_inner, evecs_inner = self.inner._symeig(eigenvectors=eigenvectors)
        evals = _kron_diag(evals_outer, evals_inner)
        if eigenvectors:
            evecs = KroneckerProductLinearOperator(evecs_outer, evecs_inner)
        else:
            evecs = None
        return evals, evecs
