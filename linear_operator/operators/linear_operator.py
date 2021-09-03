#!/usr/bin/env python3

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch

from ..utils.memoize import cached
from .linear_operator_representation_tree import LinearOperatorRepresentationTree

HANDLED_FUNCTIONS = {}
HANDLED_SECOND_ARG_FUNCTIONS = {}
COMPATIBLE_TYPES = {}


def implements(torch_function, types=None):
    """Register a torch function override for LinearOperator"""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        COMPATIBLE_TYPES[torch_function] = types
        return func

    return decorator


def implements_second_arg(torch_function, types=None):
    """
    Register a torch function override for LinearOperator,
    where the first argument of the function is a torch.Tensor and the
    second argument is a LinearOperator

    Examples of this include :meth:`torch.cholesky_solve`, `torch.solve`,
    or `torch.matmul`.
    """

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_SECOND_ARG_FUNCTIONS[torch_function] = func
        COMPATIBLE_TYPES[torch_function] = types
        return func

    return decorator


class LinearOperator(ABC):
    r"""
    Base class for LinearOperators.

    :ivar torch.Size batch_shape: The dimensions of the batches
        represented by the :obj:`~linear_operator.operators.LinearOperator`.
        For example, a LinearOperator with shape (4 x 3 x 5 x 6) would have a batch
        shape of (4 x 3).
    :ivar torch.dtype dtype: Data type represented by the LinearOperator.
    :ivar str device: Device where LinearOperator is stored.
    :ivar bool is_square: Whether or not the LinearOperator is a square
        operator.
    :ivar torch.Size matrix_shape: The 2-dimensional shape of the implicit
        matrix represented by the :obj:`~linear_operator.operators.LinearOperator`.
        In other words: a :obj:`torch.Size` that consists of the operators'
        output dimension and input dimension.
    :ivar torch.Size shape: The overall operator shape: :attr:`batch_shape` +
        :attr:`matrix_shape`.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    ####
    # The following methods need to be defined by the LinearOperator
    ####
    @abstractmethod
    def _matmul(self, rhs):
        r"""
        Performs a matrix multiplication :math:`\mathbf KM` with the (... x M x N) matrix :math:`\mathbf K`
        that this LinearOperator represents. Should behave as
        :func:`torch.matmul`. If the LinearOperator represents a batch of
        matrices, this method should therefore operate in batch mode as well.

        ..note::
            This method is intended to be used only internally by various
            Functions that support backpropagation (e.g., :class:`Matmul`).
            Once this method is defined, it is strongly recommended that one
            use :func:`~linear_operator.operators.LinearOperator.matmul` instead, which makes use of this
            method properly.

        :param rhs: the matrix :math:`\mathbf M` to multiply with.
        :type rhs: torch.Tensor (... x N x C)
        :return: :math:`\mathbf KM`
        :rtype: torch.Tensor (... x M x C)
        """
        raise NotImplementedError("The class {} requires a _matmul function!".format(self.__class__.__name__))

    def _diagonal(self) -> torch.Tensor:
        """
        This method only needs to be implemented for square linear operators

        :return: The (batched) diagonal of the (... x N x N) linear operator.
        :rtype: torch.Tensor (... x N)
        """
        raise NotImplementedError

    @abstractmethod
    def _size(self):
        """
        Returns the size of the resulting Tensor that the linear operator represents.

        ..note::
            This method is used internally by the related function
            :func:`~linear_operator.operators.LinearOperator.size`,
            which does some additional work. Calling this method directly is discouraged.

        :return: The size of the (batched) matrix :math:`\mathbf K` represented by this LinearOperator
        :rtype: torch.Size
        """
        raise NotImplementedError("The class {} requires a _size function!".format(self.__class__.__name__))

    def _bilinear_derivative(self, left_vecs, right_vecs):
        r"""
        Given :math:`\mathbf u` (left_vecs) and :math:`\mathbf v` (right_vecs),
        Computes the derivatives of (:math:`\mathbf u^\top \mathbf K \mathbf v`) w.r.t. :math:`\mathbf K`.

        ..note::
            This method is intended to be used only internally by various
            Functions that support backpropagation.  For example, this method
            is used internally by :func:`~linear_operator.operators.LinearOperator.inv_quad_logdet`.
            It is not likely that users will need to call this method directly.

        :return: Derivative with respect to the arguments that are actually
            used to represent this this LinearOperator.
        :rtype: tuple(torch.Tensor)
        """
        from collections import deque

        args = tuple(self.representation())
        args_with_grads = tuple(arg for arg in args if arg.requires_grad)

        # Easy case: if we don't require any gradients, then just return!
        if not len(args_with_grads):
            return tuple(None for _ in args)

        # Normal case: we'll use the autograd to get us a derivative
        with torch.autograd.enable_grad():
            loss = (left_vecs * self._matmul(right_vecs)).sum()
            loss.requires_grad_(True)
            actual_grads = deque(torch.autograd.grad(loss, args_with_grads, allow_unused=True))

        # Now make sure that the object we return has one entry for every item in args
        grads = []
        for arg in args:
            if arg.requires_grad:
                grads.append(actual_grads.popleft())
            else:
                grads.append(None)

        return tuple(grads)

    ####
    # These methods are necessary to hack autograd to accept LinearOperators
    ####

    @property
    def _args(self):
        return self._args_memo

    @_args.setter
    def _args(self, args):
        self._args_memo = args

    # TODO: make this method private
    def representation(self) -> Tuple[torch.Tensor]:
        representation = []
        for arg in self._args:
            if torch.is_tensor(arg):
                representation.append(arg)
            elif hasattr(arg, "representation") and callable(arg.representation):  # Is it a LinearOperator?
                representation += list(arg.representation())
            else:
                raise RuntimeError("Representation of a LinearOperator should consist only of Tensors")
        return tuple(representation)

    # TODO: make this method private
    def representation_tree(self) -> LinearOperatorRepresentationTree:
        return LinearOperatorRepresentationTree(self)

    ####
    # These private methods contain default implementations of standard torch.Tensor methods,
    # without any argument checks.
    # LinearOperator subclasses should override these methods, and NOT their corresponding public methods.
    ####

    def _logdet(self):
        """Method that allows implementing special-cased logdets. Should not be called directly"""
        return self.to_dense().logdet()

    def _solve(self, rhs):
        """Method that allows implementing special-cased linear solves. Should not be called directly"""
        # We'll just default to using torch.solve here.
        # Smarter iterative methods should be used in the future.
        return torch.linalg.solve(self.to_dense(), rhs)

    @cached(name="symeig")
    def _symeig(self, eigenvectors: bool = False) -> Tuple[torch.Tensor, Optional["LinearOperator"]]:
        """Method that allows implementing special-cased symeig computation. Should not be called directly"""
        from .dense_linear_operator import DenseLinearOperator

        dtype = self.dtype  # perform decomposition in double precision for numerical stability
        # TODO: Use fp64 registry once #1213 is addressed
        if eigenvectors:
            evals, evecs = torch.linalg.eigh(self.to_dense().to(dtype=torch.double))
            evecs = DenseLinearOperator(evecs.to(dtype=dtype))
        else:
            evals = torch.linalg.eigvalsh(self.to_dense().to(dtype=torch.double))
            evecs = None
        evals = evals.to(dtype=dtype)
        return evals, evecs

    ###
    # Public properties
    ###

    @property
    def batch_shape(self):
        return torch.Size(self.shape[:-2])

    @property
    def dtype(self) -> torch.dtype:
        return self._args[0].dtype

    @property
    def device(self) -> str:
        return self._args[0].device

    @property
    def is_square(self):
        return self.matrix_shape[0] == self.matrix_shape[1]

    @property
    def matrix_shape(self):
        return torch.Size(self.shape[-2:])

    @property
    def shape(self):
        return self.size()

    ###
    # Public methods. These are generally designed to follow the API of torch.Tensor.
    ###

    @implements(torch.add)
    def add(self, other: Union[torch.Tensor, "LinearOperator"], alpha: float = 1.0) -> LinearOperator:
        r"""
        Each element of the tensor :attr:`other` is multiplied by the scalar :attr:`alpha`
        and added to each element of the :obj:`~linear_operator.operators.LinearOperator`.
        The resulting :obj:`~linear_operator.operators.LinearOperator` is returned.

        .. math::
            \text{out} = \text{self} + \text{alpha} ( \text{other} )

        :param other: object to add to :attr:`self`.
        :type other: torch.Tensor or ~linear_operator.operators.LinearOperator
        :param float alpha: Optional scalar multiple to apply to :attr:`other`.
        :return: :math:`\mathbf A + \alpha \mathbf O`, where :math:`\mathbf A`
            is the linear operator and :math:`\mathbf O` is :attr:`other`.
        """
        if alpha == 1.0:
            return self + other
        else:
            raise NotImplementedError(
                f"LinearOperator#add is currently only implemented for alpha=1. Got alpha={alpha}."
            )

    # TODO: rename to diagonal
    @implements(torch.diagonal)
    def diagonal(self, offset: int = 0, dim1: int = -2, dim2: int = -1) -> torch.Tensor:
        r"""
        As :func:`torch.diagonal`, returns the diagonal of the matrix
        :math:`\mathbf A` this LinearOperator represents as a vector.

        .. note::
            This method is only implemented for when :attr:`dim1` and :attr:`dim2` are equal
            to -2 and -1, respectfully, and :attr:`offset = 0`.

        :rtype: torch.Tensor (... x N)
        :return: The diagonal (or batch of diagonals) of :math:`\mathbf A`.
        """
        if not (offset == 0 and dim1 == -2 and dim2 == -1):
            raise NotImplementedError(
                "LinearOperator#diagonal is only implemented for when :attr:`dim1` and :attr:`dim2` are equal "
                "to -2 and -1, respectfully, and :attr:`offset = 0`. "
                f"Got: offset={offset}, dim1={dim1}, dim2={dim2}."
            )
        elif not self.is_square:
            raise RuntimeError("LinearOperator#diagonal is only implemented for square operators.")
        return self._diagonal()

    def dim(self) -> int:
        """
        Alias of :meth:`~linear_operator.operators.LinearOperator.ndimension`
        """
        return self.ndimension()

    @implements(torch.linalg.eigh)
    def eigh(self) -> Tuple[torch.Tensor, "LinearOperator"]:
        """
        Compute the symmetric eigendecomposition of the linear operator.
        This can be very slow for large tensors.
        Should be special-cased for tensors with particular structure.

        .. note::
            This method does NOT sort the eigenvalues.

        :rtype: torch.Tensor, ~linear_opeator.operators.LinearOperator
        :return:
            - The eigenvalues (... x N)
            - The eigenvectors (... x N x N).  If :attr:`eigenvectors = False`, then this is None.
        """
        return self._symeig(eigenvectors=True)

    @implements(torch.linalg.eigvalsh)
    def eigvalsh(self) -> torch.Tensor:
        """
        Compute the eigenvalues of symmetric linear operator.
        This can be very slow for large tensors.
        Should be special-cased for tensors with particular structure.

        .. note::
            This method does NOT sort the eigenvalues.

        :rtype: torch.Tensor
        :return: the eigenvalues (... x N)
        """
        return self._symeig(eigenvectors=False)[0]

    @implements(torch.kron)
    def kron(self, other: Union[torch.Tensor, "LinearOperator"]) -> "LinearOperator":
        r"""
        Produces a :class:`~linear_operator.operators.KroneckerProductLinearOperator`, which represents
        the Kronecker product (denoted by :math:`\otimes`) of :attr:`self` and :attr:`other`.

        If :attr:`self` is (... x M x N) and :attr:`other` is (... x P x Q), the resulting linear operator
        will have size (... x MP x QN).

        :param other:
        :type other: torch.Tensor or ~linear_operator.operators.LinearOperator
        :return: operator representing :math:`\text{self } \otimes \text{ other}`
        :rtype: ~linear_operator.operators.KroneckerProductLinearOperator
        """

        from .kronecker_product_linear_operator import KroneckerProductLinearOperator

        return KroneckerProductLinearOperator(self, other)

    @implements(torch.logdet)
    def logdet(self) -> torch.Tensor:
        r"""
        Computes the log determinant :math:`\log \vert \mathbf A \vert`.
        If the linear operator represents a (... x N x N) batch of square operators,
        the result will be a tensor of size (...).

        :return: (batch of) log determinant(s)
        :rtype: torch.Tensor
        """
        if not self.is_square:
            raise RuntimeError(
                "LinearOperator#logdet can only be called on (batched) square operators. "
                f"Tried to call logdet on a {self.shape} operator."
            )
        res = self._logdet()
        return res

    @implements(torch.matmul)
    def matmul(self, other: torch.Tensor) -> torch.Tensor:
        r"""
        Performs :math:`\mathbf A \mathbf B`, where :math:`\mathbf A \in
        \mathbb R^{M \times N}` is the LinearOperator and :math:`\mathbf B`
        is a right hand side :obj:`torch.Tensor` (or :obj:`~linear_operator.operators.LinearOperator`).

        :param other: :math:`\mathbf B` - the matrix or vector to multiply against.
        :type other: torch.Tensor or ~linear_operator.operators.LinearOperator (... x N x D)
        :rtype: torch.Tensor or ~linear_operator.operators.LinearOperator (... x M x D)
        :return: The resulting of applying the linear operator to :math:`\mathbf B`.
            The return type will be the same as :attr:`other`'s type.
        """
        is_vec = other.dim() == 1
        if is_vec:
            other = other.unsqueeze(-1)
        res = self._matmul(other)
        if is_vec:
            res = res.squeeze(-1)
        return res

    def ndimension(self) -> int:
        """
        :return: The number of dimensions.
        :rtype: int
        """
        return len(self.shape)

    def size(self, dim: int = None) -> Union[torch.Size, int]:
        """
        :rtype: torch.Size or int
        :return: The size of the LinearOperator (along the specified dimension).
        """
        size = self._size()
        if dim is not None:
            return size[dim]
        return size

    @implements(torch.linalg.solve)
    def solve(self, right_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Computes a linear solve (w.r.t self = :math:`\mathbf A`) with several
        right hand sides :math:`\mathbf R`.
        I.e. computes

        .. math::
           \begin{equation}
               \mathbf A^{-1} \mathbf R,
           \end{equation}

        where :math:`\mathbf R` is :attr:`right_tensor` and :math:`\mathbf A` is the LinearOperator.

        :param right_tensor: :math:`\mathbf R` - the right hand side
        :type right_tensor: torch.Tensor (... x N x K)
        :rtype: torch.Tensor (... x N x K or ... x M x K)
        :return: :math:`\mathbf A^{-1} \mathbf R` or :math:`\mathbf L \mathbf A^{-1} \mathbf R`.
        """
        if not self.is_square:
            raise RuntimeError(
                "solve only operates on (batches of) square LinearOperators. "
                "Got a {} of size {}.".format(self.__class__.__name__, self.size())
            )

        is_vec = False
        if self.dim() == 2 and right_tensor.dim() == 1:
            is_vec = True
            if self.shape[-1] != right_tensor.numel():
                raise RuntimeError(
                    "LinearOperator (size={}) cannot be multiplied with right-hand-side Tensor (size={}).".format(
                        self.shape, right_tensor.shape
                    )
                )
            right_tensor = right_tensor.unsqueeze(-1)

        res = self._solve(right_tensor)
        if is_vec:
            res = res.squeeze(-1)
        return res

    @cached
    def to_dense(self) -> torch.Tensor:
        """
        Explicitly evaluates the matrix this LinearOperator represents. This function
        should return a :obj:`torch.Tensor` storing an exact representation of this LinearOperator.
        """
        _, num_cols = self.matrix_shape
        eye = torch.eye(num_cols, dtype=self.dtype, device=self.device)
        res = self.matmul(eye)
        return res

    def __add__(self, other: Union[torch.Tensor, "LinearOperator"]) -> LinearOperator:
        from .sum_linear_operator import SumLinearOperator

        if isinstance(other, SumLinearOperator):
            return SumLinearOperator(self, *other.linear_operators)
        else:
            return SumLinearOperator(self, other)

    def __radd__(self, other: Union[torch.Tensor, "LinearOperator"]) -> LinearOperator:
        return self + other

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if not isinstance(args[0], LinearOperator):
            if func not in HANDLED_SECOND_ARG_FUNCTIONS:
                name = func.__name__.replace("linalg_", "linalg.")
                raise NotImplementedError(f"torch.{name} is not implemented for {self.__class__.__name__}.")
            return HANDLED_SECOND_ARG_FUNCTIONS[func](args[1], args[0], *args[2:], **kwargs)
        else:
            if func not in HANDLED_FUNCTIONS:
                name = func.__name__.replace("linalg_", "linalg.")
                raise NotImplementedError(f"torch.{name} is not implemented for {self.__class__.__name__}.")
            return HANDLED_FUNCTIONS[func](*args, **kwargs)


__all__ = ["LinearOperator"]
