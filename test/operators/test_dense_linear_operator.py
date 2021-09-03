#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from linear_operator.operators import DenseLinearOperator
from linear_operator.test.linear_operator_test_case import LinearOperatorTestCase, SymmetricLinearOperatorTestCase


class TestDenseLinearOperator(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        torch.manual_seed(0)
        mat = torch.randn(5, 6)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.tensor


class TestDenseLinearOperatorMultiBatch(LinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        torch.manual_seed(0)
        mat = torch.randn(2, 3, 5, 6)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.tensor


class TestSymmetricDenseLinearOperator(SymmetricLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        torch.manual_seed(0)
        mat = torch.randn(5, 6)
        mat = mat @ mat.transpose(-1, -2)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.tensor


class TestSymmetricDenseLinearOperatorMultiBatch(SymmetricLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        torch.manual_seed(0)
        mat = torch.randn(2, 3, 5, 6)
        mat = mat @ mat.transpose(-1, -2)
        mat.requires_grad_(True)
        return DenseLinearOperator(mat)

    def evaluate_linear_operator(self, linear_operator):
        return linear_operator.tensor
