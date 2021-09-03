#!/usr/bin/env python3

from __future__ import annotations

from typing import Tuple, Union

import torch

from ..utils.broadcasting import _mul_broadcast_shape
from .dense_linear_operator import DenseLinearOperator
from .linear_operator import LinearOperator

# from .broadcasted_linear_operator import BroadcastedLinearOperator


class SumLinearOperator(LinearOperator):
    """
    A :class:`~linear_operator.LinearOperator` that represents the sum of multiple
    sub-LinearOperators.

    :param linear_operators:
    :type linear_operators: tuple(~linear_operator.LinearOperator or torch.Tensor)
    """

    def __init__(self, *linear_operators: Tuple[Union[LinearOperator, torch.Tensor]], **kwargs):

        # If we have any tensors, make sure that singleton dimensions are appropriately broadcast
        broadcast_shape = None
        if any(torch.is_tensor(lo) for lo in linear_operators):
            broadcast_shape = _mul_broadcast_shape(*[lo.shape for lo in linear_operators])

        linear_operators = list(linear_operators)
        for i, linear_operator in enumerate(linear_operators):
            if torch.is_tensor(linear_operator):
                tensor = linear_operator.expand(*linear_operator.shape[:-2], *broadcast_shape[-2:])
                linear_operators[i] = DenseLinearOperator(tensor)
            elif not isinstance(linear_operator, LinearOperator):
                raise TypeError("All arguments of a SumLinearOperator should be LinearOperators or Tensors")

        super().__init__(*linear_operators, **kwargs)
        self.linear_operators = tuple(linear_operators)

    def _bilinear_derivative(self, left_vecs: torch.Tensor, right_vecs: torch.Tensor) -> torch.Tensor:
        res = tuple(
            var
            for linear_operator in self.linear_operators
            for var in linear_operator._bilinear_derivative(left_vecs, right_vecs)
        )
        return res

    def _matmul(self, rhs: torch.Tensor) -> torch.Tensor:
        return sum(linear_operator._matmul(rhs) for linear_operator in self.linear_operators)

    def _size(self) -> torch.Shape:
        return _mul_broadcast_shape(*[lt.shape for lt in self.linear_operators])

    def __add__(self, other: Union[torch.Tensor, "LinearOperator"]) -> LinearOperator:
        if isinstance(other, SumLinearOperator):
            return SumLinearOperator(*self.linear_operators, *other.linear_operators)
        elif isinstance(other, LinearOperator):
            return SumLinearOperator(*self.linear_operators, other)
        elif torch.is_tensor(other):
            other = other.expand(*other.shape[:-2], *self.matrix_shape)
            return SumLinearOperator(*self.linear_operators, DenseLinearOperator(other))
        else:
            raise AttributeError("other must be a LinearOperator")
