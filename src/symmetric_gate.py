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

from src.involutions import get_m_basis

_pauli_matrices = np.array(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]
)
"""Single-qubit Paulis:    I                 X                   Y                  Z"""

_pauli_letters = "IXYZ"
"""Single-qubit Pauli letters that make up Pauli words."""


@lru_cache
def quotient_basis_matrices(num_wires):
    if num_wires < 1:
        raise ValueError(f"Require at least one wire, got {num_wires}.")
    if num_wires > 7:
        raise ValueError(
            f"Creating the Pauli basis tensor for more than 7 wires (got {num_wires}) is deactivated."
        )
    n = 2 ** num_wires
    basis = []
    for j in range(1, n):
        E0j = np.zeros((n, n), dtype=complex)
        E0j[0, j] = 1.0
        basis.append(E0j - E0j.T)
        basis.append(-1j * (E0j + E0j.conj().T))

    return 1j * np.stack(basis)


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


class SymmetricGate(Operation):
    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (1,)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = None
    """Gradient computation method."""

    def __init__(self, theta, wires, inv_type, p=None, q=None, id=None):
        num_wires = 1 if isinstance(wires, int) else len(wires)
        self.hyperparameters["num_wires"] = num_wires
        self.inv_type = inv_type
        self._basis = get_m_basis(wires, inv_type, p, q)

        theta_shape = qml.math.shape(theta)
        expected_dim = self.basis.shape[0]

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

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, val):
        self._basis = val

    # @staticmethod
    def compute_matrix(self, theta, num_wires):

        interface = qml.math.get_interface(theta)
        if interface == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)
        A = qml.math.tensordot(theta, self.basis, axes=[[-1], [0]])
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
            mat = self.compute_matrix(theta, self.basis)
            return qml.math.real(mat), qml.math.imag(mat)

        if interface == "jax":
            import jax

            theta = qml.math.cast_like(theta, 1j)
            # These lines compute the Jacobian of compute_matrix every time -> to be optimized
            jac = jax.jacobian(self.compute_matrix, argnums=0, holomorphic=True)(theta, self.basis)

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

