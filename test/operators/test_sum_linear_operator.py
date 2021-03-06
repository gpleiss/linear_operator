#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from linear_operator.operators import DenseLinearOperator, DiagLinearOperator, to_linear_operator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase


class TestSumLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        m1 = torch.tensor(
            [[5, 1, 2, 0], [1, 5, 1, 2], [2, 1, 5, 1], [0, 2, 1, 5]], dtype=torch.float, requires_grad=True
        )
        t1 = DenseLinearOperator(m1)
        d2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
        t2 = DiagLinearOperator(d2)
        return t1 + t2

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.to_dense() for lt in linear_operator.linear_operators]
        return sum(tensors)


class TestSumLinearOperatorBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        m1 = torch.tensor(
            [[5, 1, 2, 0], [1, 5, 1, 2], [2, 1, 5, 1], [0, 2, 1, 5]], dtype=torch.float, requires_grad=True
        )
        t1 = DenseLinearOperator(m1)
        d2 = torch.tensor([[2, 0.5, 0, 0], [6, 0, 1, -1]], dtype=torch.float, requires_grad=True)
        t2 = DiagLinearOperator(d2)
        return t1 + t2

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.to_dense() for lt in linear_operator.linear_operators]
        return sum(tensors)


class TestSumLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        m1 = torch.tensor(
            [[5, 1, 2, 0], [1, 5, 1, 2], [2, 1, 5, 1], [0, 2, 1, 5]], dtype=torch.float, requires_grad=True
        )
        t1 = DenseLinearOperator(m1)
        d2 = torch.tensor(
            [[[2, 0.5, 0, 0], [5, 1, 2, 0]], [[2, 0.5, 0, 0], [6, 0, 1, -1]]], dtype=torch.float, requires_grad=True,
        )
        t2 = DiagLinearOperator(d2)
        return t1 + t2

    def evaluate_linear_operator(self, linear_operator):
        tensors = [lt.to_dense() for lt in linear_operator.linear_operators]
        return sum(tensors)


class TestSumLinearOperatorBroadcasting(unittest.TestCase):
    def test_broadcast_same_shape(self):
        test1 = to_linear_operator(torch.randn(30, 30))

        test2 = torch.randn(30, 30)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.to_dense() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.to_dense() - torch_res).sum(), 0.0)

    def test_broadcast_tensor_shape(self):
        test1 = to_linear_operator(torch.randn(30, 30))

        test2 = torch.randn(30, 1)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.to_dense() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.to_dense() - torch_res).sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
