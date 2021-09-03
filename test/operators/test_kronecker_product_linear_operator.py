#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from linear_operator.operators import DenseLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, SymmetricLinearOperatorTestCase


def kron(a, b):
    r"""
    This differs from torch.kron in that we do not extend the kronecker product to batch dimensions.
    E.g. (b1 x p x q) \kron (b1 x m x n) = (b1 x pm x qn)
    """
    res = []
    for i in range(a.size(-2)):
        row_res = []
        for j in range(a.size(-1)):
            row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
        res.append(torch.cat(row_res, -1))
    return torch.cat(res, -2)


class TestKroneckerProductLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        torch.manual_seed(0)
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(5, 2, requires_grad=True)
        c = torch.randn(6, 4, requires_grad=True)
        kp_linear_operator = torch.kron(DenseLinearOperator(a), DenseLinearOperator(b).kron(DenseLinearOperator(c)))
        return kp_linear_operator

    def evaluate_linear_operator(self, linear_operator):
        res = kron(
            linear_operator.outer.tensor, kron(linear_operator.inner.outer.tensor, linear_operator.inner.inner.tensor)
        )
        return res


class TestKroneckerProductLinearOperatorMultiBatch(TestKroneckerProductLinearOperator):
    seed = 0

    def create_linear_operator(self):
        torch.manual_seed(0)
        a = torch.randn(3, 4, 2, 3, requires_grad=True)
        b = torch.randn(3, 4, 5, 2, requires_grad=True)
        c = torch.randn(3, 4, 6, 4, requires_grad=True)
        kp_linear_operator = torch.kron(DenseLinearOperator(a), DenseLinearOperator(b).kron(DenseLinearOperator(c)))
        return kp_linear_operator


class TestSymmetricKroneckerProductLinearOperator(SymmetricLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_operator = torch.kron(DenseLinearOperator(a), DenseLinearOperator(b).kron(DenseLinearOperator(c)))
        return kp_linear_operator

    def evaluate_linear_operator(self, linear_operator):
        res = kron(
            linear_operator.outer.tensor, kron(linear_operator.inner.outer.tensor, linear_operator.inner.inner.tensor)
        )
        return res


class TestSymmetricKroneckerProductLinearOperatorBatch(TestSymmetricKroneckerProductLinearOperator):
    seed = 0

    def create_linear_operator(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float).repeat(3, 1, 1)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float).repeat(3, 1, 1)
        c = torch.tensor([[4, 0, 1, 0], [0, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float).repeat(3, 1, 1)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_linear_operator = torch.kron(DenseLinearOperator(a), DenseLinearOperator(b).kron(DenseLinearOperator(c)))
        return kp_linear_operator


if __name__ == "__main__":
    unittest.main()
