#!/usr/bin/env python3

from __future__ import annotations

import unittest

import torch

from linear_operator.operators import DiagLinearOperator
from linear_operator.test.linear_operator_test_case import SymmetricLinearOperatorTestCase


class TestDiagLinearOperator(SymmetricLinearOperatorTestCase, unittest.TestCase):
    seed = 0

    def create_linear_operator(self):
        diag = torch.tensor([1.0, 2.0, 4.0, 5.0, 3.0], requires_grad=True)
        return DiagLinearOperator(diag)

    def evaluate_linear_operator(self, linear_operator):
        diag = linear_operator._diag
        return diag.diag()


class TestDiagLinearOperatorBatch(TestDiagLinearOperator):
    seed = 0

    def create_linear_operator(self):
        diag = torch.tensor(
            [[1.0, 2.0, 4.0, 6.0, 3.0], [7.0, 0.5, 2.0, 1.0, 4.0], [1.0, 2.0, 5.0, 3.0, 4.0]], requires_grad=True
        )
        return DiagLinearOperator(diag)

    def evaluate_linear_operator(self, linear_operator):
        diag = linear_operator._diag
        return torch.cat([diag[i].diag().unsqueeze(0) for i in range(3)])


class TestDiagLinearOperatorMultiBatch(TestDiagLinearOperator):
    seed = 0

    def create_linear_operator(self):
        torch.manual_seed(0)
        diag = torch.randn(6, 3, 5).pow_(2)
        diag.requires_grad_(True)
        return DiagLinearOperator(diag)

    def evaluate_linear_operator(self, linear_operator):
        diag = linear_operator._diag
        flattened_diag = diag.view(-1, diag.size(-1))
        res = torch.cat([flattened_diag[i].diag().unsqueeze(0) for i in range(18)])
        return res.view(6, 3, 5, 5)


if __name__ == "__main__":
    unittest.main()
