# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This submodule contains the operation SpecialUnitary and
its utility functions.
"""
# pylint: disable=arguments-differ, import-outside-toplevel
from functools import lru_cache
from itertools import product

import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation

_pauli_matrices = np.array(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
)
"""Single-qubit Paulis:    I                 X                   Y                  Z"""

_pauli_letters = "IXYZ"
"""Single-qubit Pauli letters that make up Pauli words."""

@lru_cache
def pauli_basis_strings(num_wires):
    r"""Compute all :math:`n`-qubit Pauli words except ``"I"*num_wires``,
    corresponding to the Pauli basis of the Lie algebra :math:`\mathfrak{su}(N)`.

    Args:
        num_wires (int): The number of wires, or number of letters per word.

    Returns:
        list[str]: All Pauli words on ``num_wires`` qubits, except from the identity.

    There are :math:`d=4^n-1` Pauli words that are not the identity. They are ordered
    (choose the description that suits you most)

      - lexicographically.

      - such that the term acting on the last qubit changes fastest, the one acting on the first
        qubit changes slowest when iterating through the output.

      - such that the basis index, written in base :math:`4`, contains the indices for the list
        ``["I", "X", "Y", "Z"]``, in the order of the qubits

      - such that for three qubits, the first Pauli words are
        ``"IIX", ""IIY", "IIZ", "IXI", "IXX", "IXY", "IXZ", "IYI"...``

    **Example**

    >>> pauli_basis_strings(1)
    ['X', 'Y', 'Z']
    >>> len(pauli_basis_strings(3))
    63
    """
    return ["".join(letters) for letters in product(_pauli_letters, repeat=num_wires)][1:]


class HomogeneousGate(Operation):
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, theta, wires, basis, id=None):
        num_wires = 1 if isinstance(wires, int) else len(wires)
        self.hyperparameters["num_wires"] = num_wires
        self._basis = basis

        theta_shape = qml.math.shape(theta)
        expected_dim = self._basis.shape[0]
        if len(theta) != basis.shape[0]:
            raise ValueError("Expected the parameters to have the same number of dimensions as the basis. "
                             f"The parameters have shape {theta_shape} and the basis has shape {basis.shape}")

        if len(theta_shape) not in {1, 2}:
            raise ValueError(
                "Expected the parameters to have one or two dimensions without or with "
                f"broadcasting, respectively. The parameters have shape {theta_shape}"
            )

        if theta_shape[-1] != expected_dim:
            raise ValueError(
                f"Expected the parameters to have shape ({expected_dim},) or (batch_size, "
                f"{expected_dim}). The parameters have shape {theta_shape}"
            )

        super().__init__(theta, wires=wires, id=id)

    # @staticmethod
    def compute_matrix(self, theta, num_wires):

        interface = qml.math.get_interface(theta)
        if interface == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
        A = qml.math.tensordot(theta, self._basis, axes=[[-1], [0]])
        if interface == "jax" and qml.math.ndim(theta) > 1:
            # jax.numpy.expm does not support broadcasting
            return qml.math.stack([qml.math.expm(1j * _A) for _A in A])

        return qml.math.expm(1j * A, like='jax')

    def get_one_parameter_generators(self, interface=None):
        theta = self.data[0]
        if len(qml.math.shape(theta)) > 1:
            raise ValueError("Broadcasting is not supported.")

        def split_matrix(theta):
            """Compute the real and imaginary parts of the special unitary matrix."""
            mat = self.compute_matrix(theta, self._basis)
            return qml.math.real(mat), qml.math.imag(mat)

        if interface == "jax":
            import jax

            theta = qml.math.cast_like(theta, 1j)
            # These lines compute the Jacobian of compute_matrix every time -> to be optimized
            jac = jax.jacobian(self.compute_matrix, argnums=0, holomorphic=True)(theta, self._basis)

        elif interface == "torch":
            import torch

            rjac, ijac = torch.autograd.functional.jacobian(split_matrix, theta)
            jac = rjac + 1j * ijac

        elif interface in ("tensorflow", "tf"):
            import tensorflow as tf

            with tf.GradientTape(persistent=True) as tape:
                mats = qml.math.stack(split_matrix(theta))

            rjac, ijac = tape.jacobian(mats, theta)
            jac = qml.math.cast_like(rjac, 1j) + 1j * qml.math.cast_like(ijac, 1j)

        elif interface == "autograd":
            # TODO check whether we can add support for Autograd using eigenvalue decomposition
            raise NotImplementedError(
                "The matrix exponential expm is not differentiable in Autograd."
            )

        else:
            raise ValueError(f"The interface {interface} is not supported.")

        # Compute the Omegas from the Jacobian. The adjoint of U(theta) is realized via -theta
        U_dagger = self.compute_matrix(-qml.math.detach(theta), None)
        # After contracting, move the parameter derivative axis to the first position
        return qml.math.transpose(qml.math.tensordot(U_dagger, jac, axes=[[1], [0]]), [2, 0, 1])

    # def get_one_parameter_coeffs(self, interface):
    #     r"""Compute the Pauli basis coefficients of the generators of one-parameter groups
    #     that reproduce the partial derivatives of a special unitary gate.
    #
    #     Args:
    #         interface (str): The auto-differentiation framework to be used for the
    #             computation.
    #
    #     Returns:
    #         tensor_like: The Pauli basis coefficients of the effective generators
    #         that reproduce the partial derivatives of the special unitary gate
    #         defined by ``theta``. There are :math:`d=4^n-1` generators for
    #         :math:`n` qubits and :math:`d` Pauli coefficients per generator, so that the
    #         output shape is ``(4**num_wires-1, 4**num_wires-1)``.
    #
    #     Given a generator :math:`\Omega` of a one-parameter group that
    #     reproduces a partial derivative of a special unitary gate, it can be decomposed in
    #     the Pauli basis of :math:`\mathfrak{su}(N)` via
    #
    #     .. math::
    #
    #         \Omega = \sum_{m=1}^d \omega_m P_m
    #
    #     where :math:`d=4^n-1` is the size of the basis for :math:`n` qubits and :math:`P_m` are the
    #     Pauli words making up the basis. As the Pauli words are orthonormal with respect to the
    #     `trace or Frobenius inner product <https://en.wikipedia.org/wiki/Frobenius_inner_product>`__
    #     (rescaled by :math:`2^n`), we can compute the coefficients using this
    #     inner product (:math:`P_m` is Hermitian, so we skip the adjoint :math:`{}^\dagger`):
    #
    #     .. math::
    #
    #         \omega_m = \frac{1}{2^n}\operatorname{tr}\left[P_m \Omega \right]
    #
    #     The coefficients satisfy :math:`\omega_m^\ast=-\omega_m` because :math:`\Omega` is
    #     skew-Hermitian. Therefore they are purely imaginary.
    #
    #     .. warning::
    #
    #         An auto-differentiation framework is required by this function.
    #         The matrix exponential is not differentiable in Autograd. Therefore this function
    #         only supports JAX, Torch and Tensorflow.
    #
    #     .. seealso:: :meth:`~.SpecialUnitary.get_one_parameter_generators`
    #
    #     """
    #     num_wires = self.hyperparameters["num_wires"]
    #     basis = quotient_basis_matrices(num_wires)
    #     generators = self.get_one_parameter_generators(interface)
    #     return qml.math.tensordot(basis, generators, axes=[[1, 2], [2, 1]]) / 2 ** num_wires
    #
    # def decomposition(self):
    #     theta = self.data[0]
    #     if qml.math.requires_grad(theta):
    #         interface = qml.math.get_interface(theta)
    #         # Get all Pauli words for the basis of the Lie algebra for this gate
    #         words = pauli_basis_strings(self.hyperparameters["num_wires"])
    #
    #         # Compute the linear map that transforms between the Pauli basis and effective generators
    #         # Consider the mathematical derivation for the prefactor 2j
    #         omega = qml.math.real(2j * self.get_one_parameter_coeffs(interface))
    #
    #         # Create zero parameters for each Pauli rotation gate that take over the trace of theta
    #         detached_theta = qml.math.detach(theta)
    #         zeros = theta - detached_theta
    #         # Apply the linear map omega to the zeros to create the correct preprocessing Jacobian
    #         zeros = qml.math.tensordot(omega, zeros, axes=[[1], [0]])
    #
    #         # Apply Pauli rotations that yield the Pauli basis derivatives
    #         paulirots = [
    #             TmpPauliRot(zero, word, wires=self.wires, id="SU(N) byproduct")
    #             for zero, word in zip(zeros, words)
    #         ]
    #         return paulirots + [SymmetricGate(detached_theta, self.wires, self.type)]
    #
    #     return [qml.QubitUnitary(self.matrix(), wires=self.wires)]
    #
    # def adjoint(self):
    #     return SymmetricGate(-self.data[0], self.wires, self.type)

#
# class TmpPauliRot(PauliRot):
#     r"""A custom version of ``PauliRot`` that is inserted with rotation angle zero when
#     decomposing ``SpecialUnitary``. The differentiation logic makes use of the gradient
#     recipe of ``PauliRot``, but deactivates the matrix property so that a decomposition
#     of differentiated tapes is forced. During this decomposition, this private operation
#     removes itself if its angle remained at zero.
#
#     For details see :class:`~.PauliRot`.
#
#     .. warning::
#
#         Do not add this operation to the supported operations of any device.
#         Wrong results and/or severe performance degradations may result.
#     """
#
#     # Deactivate the matrix property of qml.PauliRot in order to force decomposition
#     has_matrix = False
#
#     @staticmethod
#     def compute_decomposition(theta, wires, pauli_word):
#         r"""Representation of the operator as a product of other operators (static method). :
#
#         .. math:: O = O_1 O_2 \dots O_n.
#
#
#         .. seealso:: :meth:`~.TmpPauliRot.decomposition`.
#
#         Args:
#             theta (float): rotation angle :math:`\theta`
#             wires (Iterable, Wires): the wires the operation acts on
#             pauli_word (string): the Pauli word defining the rotation
#
#         Returns:
#             list[Operator]: decomposition into an empty list of operations for
#             vanishing ``theta``, or into a list containing a single :class:`~.PauliRot`
#             for non-zero ``theta``.
#
#         .. note::
#
#             This operation is used in a differentiation pipeline of :class:`~.SpecialUnitary`
#             and most likely should not be created manually by users.
#         """
#         if qml.math.isclose(theta, theta * 0):
#             return []
#         return [PauliRot(theta, pauli_word, wires)]
#
#     def __repr__(self):
#         return f"TmpPauliRot({self.data[0]}, {self.hyperparameters['pauli_word']}, wires={self.wires.tolist()})"
#
#
# if __name__ == '__main__':
#     c = np.random.randn(8)
#     gate = SymmetricGate(c, wires=(0, 1), inv_type='AIII', p=2, q=2)
#     U = gate.compute_matrix(c, gate.basis)
