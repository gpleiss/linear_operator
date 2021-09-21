#!/usr/bin/env python3

from __future__ import annotations

import itertools
from abc import abstractmethod

import torch

from ..operators import LinearOperator
from .base_test_case import BaseTestCase


def _ensure_symmetric_grad(grad):
    """
    A gradient-hook hack to ensure that symmetric matrix gradients are symmetric
    """
    res = torch.add(grad, grad.transpose(-1, -2)).mul(0.5)
    return res


class LinearOperatorTestCase(BaseTestCase):
    @abstractmethod
    def create_linear_operator(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_linear_operator(self):
        raise NotImplementedError()

    def _test_matmul(self, rhs):
        linear_operator = self.create_linear_operator()
        linear_operator_copy = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator_copy)

        res = torch.matmul(linear_operator, rhs)
        actual = evaluated.matmul(rhs)
        self.assertAllClose(res, actual)

        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
            if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                self.assertAllClose(arg.grad, arg_copy.grad, rtol=1e-3)

    def test_add(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        rhs = torch.randn(linear_operator.shape)
        self.assertAllClose((torch.add(linear_operator, rhs)).to_dense(), evaluated + rhs)

        rhs = torch.randn(2, *linear_operator.shape)
        self.assertAllClose((linear_operator + rhs).to_dense(), evaluated + rhs)

    def test_bilinear_derivative(self):
        linear_operator = self.create_linear_operator()
        linear_operator_clone = self.create_linear_operator()
        left_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-2), 2)
        right_vecs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 2)

        deriv_custom = linear_operator._bilinear_derivative(left_vecs, right_vecs)
        deriv_auto = LinearOperator._bilinear_derivative(linear_operator_clone, left_vecs, right_vecs)

        for dc, da in zip(deriv_custom, deriv_auto):
            self.assertAllClose(dc, da)

    def test_to_dense(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)
        self.assertAllClose(linear_operator.to_dense(), evaluated)

    def test_matmul_vec(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(linear_operator.size(-1))

        # We skip this test if we're dealing with batch LinearOperators
        # They shouldn't multiply by a vec
        if linear_operator.ndimension() > 2:
            return
        else:
            return self._test_matmul(rhs)

    def test_matmul_matrix(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 4)
        return self._test_matmul(rhs)

    def test_matmul_matrix_broadcast(self):
        linear_operator = self.create_linear_operator()

        # Right hand size has one more batch dimension
        batch_shape = torch.Size((3, *linear_operator.batch_shape))
        rhs = torch.randn(*batch_shape, linear_operator.size(-1), 4)
        self._test_matmul(rhs)

        if linear_operator.ndimension() > 2:
            # Right hand size has one fewer batch dimension
            batch_shape = torch.Size(linear_operator.batch_shape[1:])
            rhs = torch.randn(*batch_shape, linear_operator.size(-1), 4)
            self._test_matmul(rhs)

            # Right hand size has a singleton dimension
            batch_shape = torch.Size((*linear_operator.batch_shape[:-1], 1))
            rhs = torch.randn(*batch_shape, linear_operator.size(-1), 4)
            self._test_matmul(rhs)


class SquareLinearOperatorTestCase(LinearOperatorTestCase):
    def _test_solve(self, rhs):
        linear_operator = self.create_linear_operator()
        linear_operator_copy = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator_copy)

        # Create a test right hand side and left hand side
        rhs.requires_grad_(True)
        rhs_copy = rhs.clone().detach().requires_grad_(True)

        # Perform the solve
        res = torch.linalg.solve(linear_operator, rhs)
        actual = evaluated.inverse().matmul(rhs_copy)
        print(res, actual)
        self.assertAllClose(res, actual, rtol=0.02, atol=1e-5)

        # Perform backward pass
        grad = torch.randn_like(res)
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
            if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                self.assertAllClose(arg.grad, arg_copy.grad, rtol=0.03, atol=1e-5)
        self.assertAllClose(rhs.grad, rhs_copy.grad, rtol=0.03, atol=1e-5)

    def test_diagonal(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)

        res = torch.diagonal(linear_operator, dim1=-2, dim2=-1)
        actual = evaluated.diagonal(dim1=-2, dim2=-1)
        self.assertAllClose(res, actual, rtol=1e-2, atol=1e-5)

    def test_solve_vector(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(linear_operator.size(-1))

        # We skip this test if we're dealing with batch LinearOperators
        # They shouldn't multiply by a vec
        if linear_operator.ndimension() > 2:
            return
        else:
            return self._test_solve(rhs)

    def test_solve_matrix(self):
        linear_operator = self.create_linear_operator()
        rhs = torch.randn(*linear_operator.batch_shape, linear_operator.size(-1), 5)
        return self._test_solve(rhs)

    def test_solve_matrix_broadcast(self):
        linear_operator = self.create_linear_operator()

        # Right hand size has one more batch dimension
        batch_shape = torch.Size((3, *linear_operator.batch_shape))
        rhs = torch.randn(*batch_shape, linear_operator.size(-1), 5)
        self._test_solve(rhs)

        if linear_operator.ndimension() > 2:
            # Right hand size has one fewer batch dimension
            batch_shape = torch.Size(linear_operator.batch_shape[1:])
            rhs = torch.randn(*batch_shape, linear_operator.size(-1), 5)
            self._test_solve(rhs)

            # Right hand size has a singleton dimension
            batch_shape = torch.Size((*linear_operator.batch_shape[:-1], 1))
            rhs = torch.randn(*batch_shape, linear_operator.size(-1), 5)
            self._test_solve(rhs)

    def test_logdet(self):
        linear_operator = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator)
        self.assertAllClose(torch.logdet(linear_operator), evaluated.logdet(), atol=1e-2, rtol=1e-2)


class SymmetricLinearOperatorTestCase(SquareLinearOperatorTestCase):
    def test_eigh(self):
        linear_operator = self.create_linear_operator()
        linear_operator_copy = self.create_linear_operator()
        evaluated = self.evaluate_linear_operator(linear_operator_copy)

        # Perform forward pass
        evals_unsorted, evecs_unsorted = torch.linalg.eigh(linear_operator)
        evecs_unsorted = evecs_unsorted.to_dense()

        # since LinearOperator.symeig does not sort evals, we do this here for the check
        evals, idxr = torch.sort(evals_unsorted, dim=-1, descending=False)
        evecs = torch.gather(evecs_unsorted, dim=-1, index=idxr.unsqueeze(-2).expand(evecs_unsorted.shape))

        evals_actual, evecs_actual = torch.linalg.eigh(evaluated.double())
        evals_actual = evals_actual.to(dtype=evaluated.dtype)
        evecs_actual = evecs_actual.to(dtype=evaluated.dtype)

        # Check forward pass
        self.assertAllClose(evals, evals_actual, rtol=1e-4, atol=1e-3)
        lt_from_eigendecomp = evecs @ torch.diag_embed(evals) @ evecs.transpose(-1, -2)
        self.assertAllClose(lt_from_eigendecomp, evaluated, rtol=1e-4, atol=1e-3)

        # if there are repeated evals, we'll skip checking the eigenvectors for those
        any_evals_repeated = False
        evecs_abs, evecs_actual_abs = evecs.abs(), evecs_actual.abs()
        for idx in itertools.product(*[range(b) for b in evals_actual.shape[:-1]]):
            eval_i = evals_actual[idx]
            if torch.unique(eval_i.detach()).shape[-1] == eval_i.shape[-1]:  # detach to avoid pytorch/pytorch#41389
                self.assertAllClose(evecs_abs[idx], evecs_actual_abs[idx], rtol=1e-4, atol=1e-3)
            else:
                any_evals_repeated = True

        # Perform backward pass
        symeig_grad = torch.randn_like(evals)
        ((evals * symeig_grad).sum()).backward()
        ((evals_actual * symeig_grad).sum()).backward()

        # Check grads if there were no repeated evals
        if not any_evals_repeated:
            for arg, arg_copy in zip(linear_operator.representation(), linear_operator_copy.representation()):
                if arg_copy.requires_grad and arg_copy.is_leaf and arg_copy.grad is not None:
                    self.assertAllClose(arg.grad, arg_copy.grad, rtol=1e-4, atol=1e-3)

        # Test with eigenvectors=False
        _evals = torch.linalg.eigvalsh(linear_operator)
        _evals, _ = torch.sort(_evals, dim=-1, descending=False)
        self.assertAllClose(_evals, evals_actual)
