from __future__ import annotations
from linear_operator.operators.diag_linear_operator import DiagLinearOperator
from linear_operator.operators.root_linear_operator import RootLinearOperator

import unittest

import torch

from linear_operator.operators import LowRankPlusDiagLinearOperator
from linear_operator.test.linear_operator_test_case import SquareLinearOperatorTestCase


class TestLowRankPlusDiagLinearOperator(SquareLinearOperatorTestCase, unittest.TestCase):
    def create_linear_operator(self, seed=0):
        torch.random.manual_seed(seed)
        tensor = torch.randn(5, 2)
        diag = torch.tensor([1.0, 2.0, 4.0, 2.0, 3.0])
        lo = LowRankPlusDiagLinearOperator(
            DiagLinearOperator(diag), RootLinearOperator(tensor)
        )
        return lo

    def evaluate_linear_operator(self, lazy_tensor):
        diag = lazy_tensor._diag_lo._diag
        root = lazy_tensor._root_lo._root
        return root @ root.transpose(-1, -2) + diag.diag_embed(dim1=-2, dim2=-1)


class TestLowRankPlusDiagLinearOperatorBatch(TestLowRankPlusDiagLinearOperator):
    def create_lazy_tensor(self, seed=0):
        torch.random.manual_seed(seed)
        tensor = torch.randn(3, 5, 2)
        diag = torch.tensor(
            [
                [1.0, 2.0, 4.0, 2.0, 3.0],
                [2.0, 1.0, 2.0, 1.0, 4.0],
                [1.0, 2.0, 2.0, 3.0, 4.0],
            ]
        )
        return RootLinearOperator(tensor) + DiagLinearOperator(diag)

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_lo._diag
        root = lazy_tensor._lazy_lo._root
        return root @ root.transpose(-1, -2) + diag.diag_embed(dim1=-2, dim2=-1)


# class TestLowRankPlusDiagLinearOperatorMultiBatch(TestLowRankPlusDiagLinearOperator):
#     seed = 4
#     # Because these LTs are large, we'll skil the big tests
#     should_test_sample = False
#     skip_slq_tests = True

#     def create_lazy_tensor(self):
#         tensor = torch.randn(4, 3, 5, 2)
#         diag = torch.tensor([[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]]).repeat(
#             4, 1, 1
#         )
#         lt = LowRankRootLinearOperator(tensor).add_diag(diag)
#         assert isinstance(lt, LowRankPlusDiagLinearOperator)
#         return lt

#     def evaluate_lazy_tensor(self, lazy_tensor):
#         diag = lazy_tensor._diag_tensor._diag
#         root = lazy_tensor._lazy_tensor.root.tensor
#         return root @ root.transpose(-1, -2) + diag.diag_embed(dim1=-2, dim2=-1)
