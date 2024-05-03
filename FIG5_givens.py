import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as spla
import numpy as np
import jax
import pennylane as qml

from src.reductive_geometry import get_pauli_basis, VectorSpace, comm
from src.homogeneous_gate import HomogeneousGate

I = np.eye(2).astype(complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)


def total_spin(phi):
    s_tot = np.kron(I, Z) + np.kron(Z, I)
    return -(np.real(phi.conj().T @ s_tot @ phi) / 2) + 1


def main():
    np.random.seed(129)
    su4_dict = get_pauli_basis(2)
    so4 = ['IY', 'XY', 'YI', 'YX', 'YZ', 'ZY']
    so4_dict = dict(zip(so4, [su4_dict[p] for p in so4]))
    so4_vspace = VectorSpace(np.stack(list(so4_dict.values())))

    g = VectorSpace(so4_vspace.basis)
    k = []
    k.append(np.block([[0, 0, 0, 0],
                       [0, 0, -1j, 0],
                       [0, 1j, 0, 0],
                       [0, 0, 0, 0]]))
    for k_i in k:
        assert np.allclose(k_i, k_i.conj().T)
    for g_i in g.basis:
        assert np.allclose(g_i, g_i.conj().T)
    subspace_k = []
    # Embed k in g
    for i, p in enumerate(k):
        subspace_k.append(g.project_coeffs(p))
    # Find the kernel of k
    k_ortho = spla.null_space(np.stack(subspace_k))
    m = np.einsum("in,ijk->njk", k_ortho, g.basis)
    diff_method = "backprop"
    # Use the DefaultQubit PennyLane device
    dev_type = "default.qubit"
    # Choose the operation to use in
    dev = qml.device(dev_type, wires=2)
    initial_state = np.array([0., 1., 0., 0.])
    print(np.einsum("ijk,i->jk", m, np.random.randn(5)))

    # initial_state = np.array([0., 1., 0., 0.])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(params):
        qml.StatePrep(initial_state, wires=(0, 1))
        HomogeneousGate(params, wires=(0, 1), basis=m)
        return qml.state()

    params = np.random.randn(m.shape[0])
    cost_function = jax.jit(circuit)
    print(total_spin(initial_state))
    out_state = cost_function(params)
    print(total_spin(out_state))


if __name__ == '__main__':
    main()
