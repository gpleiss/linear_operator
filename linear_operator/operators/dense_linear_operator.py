#!/usr/bin/env python3

from __future__ import annotations

import torch

from ..utils.broadcasting import _collapse_batch_and_singleton_dimensions
from .linear_operator import LinearOperator


class DenseLinearOperator(LinearOperator):
    def __init__(self, tsr):
        """
        A LinearOperator that is represented by a dense matrix.
        This is essentially a simple wrapper around torch.Tensor.

        :param torch.Tensor tensor: the tensor representing the LinearOperator
        """
        super(DenseLinearOperator, self).__init__(tsr)
        self.tensor = tsr

    def _bilinear_derivative(self, left_vecs, right_vecs):
        res = left_vecs.matmul(right_vecs.transpose(-1, -2))
        return (_collapse_batch_and_singleton_dimensions(res, target_shape=self.tensor.shape),)

    def _diagonal(self):
        return self.tensor.diagonal(dim1=-2, dim2=-1)

    def _matmul(self, rhs):
        return torch.matmul(self.tensor, rhs)

    def _size(self):
        return self.tensor.size()

    def to_dense(self) -> torch.Tensor:
        """
        Explicitly evaluates the matrix this LinearOperator represents. This function
        should return a :obj:`torch.Tensor` storing an exact representation of this LinearOperator.
        """
        return self.tensor


def to_linear_operator(obj):
    """
    A function which ensures that `obj` is a LinearOperator.

    If `obj` is a LinearOperator, this function does nothing.
    If `obj` is a (normal) Tensor, this function wraps it with a `DenseLinearOperator`.
    """

    if torch.is_tensor(obj):
        return DenseLinearOperator(obj)
    elif isinstance(obj, LinearOperator):
        return obj
    else:
        raise TypeError("object of class {} cannot be made into a LinearOperator".format(obj.__class__.__name__))


__all__ = ["DenseLinearOperator", "to_linear_operator"]
