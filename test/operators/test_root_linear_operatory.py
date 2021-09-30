#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from linear_operator.operators import RootLinearOperator
from linear_operator.test.linear_operator_test_case import SymmetricLinearOperatorTestCase


class TestRootLinearOperator(SymmetricLinearOperatorTestCase, unittest.TestCase):
    def create_linear_operator(self, seed=0):
        torch.random.manual_seed(seed)
        root = torch.randn(3, 5, requires_grad=True)
        return RootLinearOperator(root)

    def evaluate_linear_operator(self, lazy_tensor):
        root = lazy_tensor._root
        res = root.matmul(root.transpose(-1, -2))
        return res


class TestRootLinearOperatorBatch(TestRootLinearOperator):
    def create_lazy_tensor(self, seed=1):
        torch.random.manual_seed(seed)
        root = torch.randn(3, 5, 5) + torch.eye(5)
        root.requires_grad_(True)
        return RootLinearOperator(root)


class TestRootLinearOperatorMultiBatch(TestRootLinearOperator):
    def create_lazy_tensor(self, seed=2):
        torch.random.manual_seed(seed)
        root = torch.randn(2, 3, 5, 5) + torch.eye(5)
        root.requires_grad_(True)
        return RootLinearOperator(root)
