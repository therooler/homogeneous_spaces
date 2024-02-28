"""Functions to run the VQE example in the paper and create the corresponding plots."""
import matplotlib.pyplot as plt
import os

import pennylane as qml
import numpy as np
from tqdm import tqdm
import scipy
import jax

from src.homogeneous_gate import HomogeneousGate
from src.reductive_geometry import get_pauli_basis, HomogeneousSpace
from src.adam import ADAM
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

paulis = {'I': scipy.sparse.csr_matrix(np.eye(2).astype(np.complex64)),
          'x': scipy.sparse.csr_matrix(np.array([[0, 1], [1, 0]]).astype(np.complex64)),
          'y': scipy.sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]).astype(np.complex64)),
          'z': scipy.sparse.csr_matrix(np.array([[1, 0], [0, -1]]).astype(np.complex64))}


def get_random_state(N: int):
    state = np.random.randn(2 ** N) + 1j * np.random.randn(2 ** N)
    return state / np.linalg.norm(state)


def apply_op_in_layers(theta, Op, depth, nqubits, basis):
    """Apply a (callable) operation in layers.

    Args:
        theta (tensor_like): The arguments passed to the operations. The expected shape
            is ``(depth, k, num_params_op)``, where ``k`` is determined by ``fix_pars_per_layer``
            and ``num_params_op`` is the number of paramters each operation takes.
        Op (callable): The operation to apply
        depth (int): The number of layers to apply

    """

    for d in range(depth):
        # Even-odd qubit pairs
        idx = 0
        for i in range(0, nqubits, 2):
            Op(theta[d, idx], [i, i + 1], basis)
        idx = 1
        # Odd-even qubit pairs
        if nqubits > 2:
            for i in range(1, nqubits, 2):
                Op(theta[d, idx], [i, (i + 1) % nqubits], basis)
        if nqubits > 2:
            Op(theta[d, 0], wires=[nqubits - 1, 0], basis=basis)


def exact_diagonalization_heisenberg(nqubits, J_i):
    interactions_nn = []
    for i in range(nqubits - 1):
        interactions_nn.append((i, i + 1))
    interactions_nn.append((nqubits - 1, 0))

    H = scipy.sparse.csr_matrix((int(2 ** nqubits), (int(2 ** nqubits))), dtype=complex)
    if isinstance(J_i, np.ndarray):
        couplings = np.copy(J_i)
    else:
        couplings = np.array([J_i] * nqubits)
    for term in ['x', 'y', 'z']:
        for n, interac in enumerate(interactions_nn):
            tprod = ["I" for _ in range(nqubits)]
            for loc in interac:
                tprod[loc] = term
            p = paulis[tprod[0]]
            for op in range(1, nqubits):
                p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
            H += couplings[n] * p

    return scipy.sparse.linalg.eigsh(H, which='SA', k=8), H


def Sz_operator(nqubits):
    Sz = scipy.sparse.csr_matrix((int(2 ** nqubits), (int(2 ** nqubits))), dtype=complex)
    for loc in range(nqubits):
        tprod = ["I" for _ in range(nqubits)]
        tprod[loc] = 'z'
        p = paulis[tprod[0]]
        for op in range(1, nqubits):
            p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
        Sz += p
    return Sz


def Spin_operator(nqubits):
    S = scipy.sparse.csr_matrix((int(2 ** nqubits), (int(2 ** nqubits))), dtype=complex)
    for term in ['x', 'y', 'z']:
        for loc in range(nqubits):
            tprod = ["I" for _ in range(nqubits)]
            tprod[loc] = term
            p = paulis[tprod[0]]
            for op in range(1, nqubits):
                p = scipy.sparse.kron(p, paulis[tprod[op]], format='csr')
            S += p
    return S / 2


def vqe(
        seed_list,
        J_i,
        nqubits=None,
        depth=None,
        max_steps=None,
        learning_rate=None,
        gate='hom',
        data_header="",
        state_0='random',
):
    """Run the VQE numerical experiment.

    Args:
        seed_list (list[int]): Sequence of randomness seeds for the observable creation.
            One VQE experiment is run for each seed
        nqubits (int): Number of qubits
        depth (int): Number of layers of operations to use in the circuit ansatz
        opname (str): Which operation to use in the circuit ansatz, may be ``"su4"`` for
            out proposed SU(N) parametrization on two qubits, or ``"decomp"`` for the
            gate sequence created by ``gate_decomp_su4``.
        p_q ((int,int)): p and q of cartan involution.
        max_steps (int): The number of steps for which to run the gradient descent
            optimizer of the VQE.
        learning_rate (float): The learning rate of the gradient descent optimizer.
        data_header (str): Subdirectory of ``./data`` to save data to

    This function executes the full VQE workflow, including

      - the generation of the observable and storage of its key energy information

      - the setup of the cost function by composing the chosen operation in a fabric of layers
        and measuring the observable expectation value afterwards

      - the optimization of the cost function for the indicated number of steps, using vanilla
        gradient descent with a fixed learning rate

      - the storage of the optimization curves on disk

    """
    if isinstance(J_i, np.ndarray):
        data_path = f"./data/random_chain_su2_state_{state_0}/{data_header}/{nqubits}/{depth}/"
    else:
        data_path = f"./data/uniform_chain_su2_state_{state_0}/{data_header}/{nqubits}/{depth}/"

    # Set the differentiation method to use backpropagation
    diff_method = "backprop"
    # Use the DefaultQubit PennyLane device
    dev_type = "default.qubit"
    # Choose the operation to use in the circuit ansatz
    Op = HomogeneousGate

    if gate == 'hom':
        su2_paulis = get_pauli_basis(1)
        I = np.eye(2, dtype=complex)
        k = np.stack([np.kron(su2_paulis['X'], I) + np.kron(I, su2_paulis['X']),
                      np.kron(su2_paulis['Y'], I) + np.kron(I, su2_paulis['Y']),
                      np.kron(su2_paulis['Z'], I) + np.kron(I, su2_paulis['Z'])]) / 2
        hom_space = HomogeneousSpace(2, k)
        basis_m = jax.numpy.stack(hom_space.m.basis)

    elif gate == 'equiv':
        basis_m = [qml.SWAP((0, 1)).compute_matrix()]
        basis_m = jax.numpy.stack(basis_m)
    else:
        raise NotImplementedError
    dev = qml.device(dev_type, wires=nqubits)
    (energies, states), observable = exact_diagonalization_heisenberg(nqubits, J_i)
    Sz = Sz_operator(nqubits)
    S = Spin_operator(nqubits)
    for i in range(len(energies)):
        state = np.reshape(states.T[i], 2 ** nqubits)
        sz = (state.conj().T @ Sz @ state).real
        s = (state.conj().T @ S @ state).real
        print(f"E = {energies[i]},  Sz = {sz} - s = {s}")

    observable = observable.toarray()
    S = S.toarray()
    commutes_with_su2_generator = np.allclose(S @ observable - observable @ S, np.zeros_like(S))
    print("commutes_with_su2_generator", commutes_with_su2_generator)
    assert commutes_with_su2_generator
    for seed in tqdm(seed_list):
        np.random.seed(seed)
        if state_0 == 'random':
            initial_state = get_random_state(nqubits)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def cost_function(params, obs):
            """Cost function of the VQE."""

            if state_0 == 'random':
                qml.StatePrep(initial_state, wires=list(range(nqubits)))
            elif state_0 == 'zero':
                pass
            elif state_0 == 'psi_+':
                for i in range(0, nqubits, 2):
                    qml.Hadamard(i)
                    qml.CNOT((i, i + 1))
            elif state_0 == 'psi_-':
                for i in range(0, nqubits, 2):
                    qml.PauliX(i)
                    qml.Hadamard(i)
                    qml.CNOT((i, i + 1))
            elif state_0 == 'phi_+':
                for i in range(0, nqubits, 2):
                    qml.Hadamard(i)
                    qml.PauliX(i + 1)
                    qml.CNOT((i, i + 1))
            elif state_0 == 'phi_-':
                for i in range(0, nqubits, 2):
                    qml.PauliX(i)
                    qml.PauliX(i + 1)
                    qml.Hadamard(i)
                    qml.CNOT((i, i + 1))
            apply_op_in_layers(params, Op, depth, nqubits, basis_m)
            return qml.expval(qml.Hermitian(obs, wires=list(range(nqubits))))

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def get_state(params):
            """Cost function of the VQE."""

            if state_0 == 'random':
                qml.StatePrep(initial_state, wires=list(range(nqubits)))
            elif state_0 == 'zero':
                pass
            elif state_0 == 'psi_+':
                for i in range(0, nqubits, 2):
                    qml.Hadamard(i)
                    qml.CNOT((i, i + 1))
            elif state_0 == 'psi_-':
                for i in range(0, nqubits, 2):
                    qml.PauliX(i)
                    qml.Hadamard(i)
                    qml.CNOT((i, i + 1))
            elif state_0 == 'phi_+':
                for i in range(0, nqubits, 2):
                    qml.Hadamard(i)
                    qml.PauliX(i + 1)
                    qml.CNOT((i, i + 1))
            elif state_0 == 'phi_-':
                for i in range(0, nqubits, 2):
                    qml.PauliX(i)
                    qml.PauliX(i + 1)
                    qml.Hadamard(i)
                    qml.CNOT((i, i + 1))
            apply_op_in_layers(params, Op, depth, nqubits, basis_m)
            return qml.state()

        cost_function_jax = jax.jit(lambda x: cost_function(x, observable))
        grad_function = jax.grad(cost_function_jax)

        cost_path = data_path + f"cost_{seed}.npy"
        spin_path = data_path + f"spin_per_step_{seed}.npy"

        # Check whether the optimization curves already exist on disk
        if not os.path.exists(cost_path):
            # Parameter shape: depth x (2 layers) x (15 = 4**2-1)
            shape = (depth, 2, basis_m.shape[0])
            # Create initial parameters
            params = jax.numpy.ones(shape) * np.random.uniform(0.01)
            optimizer = ADAM(params, learning_rate)
            cost_history = []
            initial_state_theta = get_state(params)
            sz = (initial_state_theta.conj().T @ Sz @ initial_state_theta).real
            s = (initial_state_theta.conj().T @ S @ S @ initial_state_theta).real
            print(f"Start,  Sz = {sz} - s = {s}")
            spin_per_step = []
            for step in tqdm(range(max_steps)):
                # Record old cost
                cost_history.append(cost_function_jax(params))
                # Make step
                params = params - optimizer.update_params(grad_function(params))
                if (not (step + 1) % 100) or (step == 0):
                    phi = get_state(params)
                    spin_per_step.append((step, phi.conj().T @ S @ S @ phi))
            # Record final cost
            cost_history.append(cost_function_jax(params))
            print(cost_history[-1])
            # Save cost to disk
            final_state_theta = get_state(params)
            sz = (final_state_theta.conj().T @ Sz @ final_state_theta).real
            s = (final_state_theta.conj().T @ S @ S @ final_state_theta).real
            print(f"End,  Sz = {sz} - s = {s}")
            np.save(cost_path, np.array(cost_history))
            np.save(spin_path, np.array(spin_per_step))

        else:
            print(f"{cost_path} exists")
        # Store the ground state and maximal energy of the created observable on disk
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if not os.path.exists(data_path + f"gs_energy.npy"):
            gs_energy, _ = exact_diagonalization_heisenberg(nqubits, J_i)[0]
            np.save(data_path + f"gs_energy_{seed}.npy", gs_energy)


data_header = "/"

if __name__ == "__main__":
    nruns = 1  # Number of VQE runs
    max_steps = 5000  # number of optimization steps in each VQE
    learning_rate = 1e-2  # Learning rate for gradient descent in the optimization
    nqubits = 8  # Number of qubits
    depth = 8  # Number of layers in the circuit ansatz. The VQE is run for each depth
    RUN = False  # Whether to run the computation. If results are present, computations are skipped
    PLOT = True  # Whether to create plots of the results
    np.random.seed(1234)
    chain_type = 'uniform_chain'
    # chain_type = 'random_chain'
    if chain_type == 'random_chain':
        J_i = np.random.randn(nqubits)
    elif chain_type == 'uniform_chain':
        J_i = 1.0
    else:
        raise NotImplementedError
    # Generate seeds (deterministically)
    seed_lists = [i * 37 for i in range(nruns)]

    # Run computation if requested. If results are present already, computations are skipped
    if RUN:
        for state_0 in ['random', 'zero', 'psi_+', 'psi_-', 'phi_+', 'phi_-']:
            for gate in ['hom', 'equiv']:
                data_header = gate

                # Set up path and create missing directories
                data_path = f"./data/{chain_type}_su2_state_{state_0}/{data_header}/{nqubits}/{depth}/"
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                # Mappable version of ``vqe``.
                func = vqe(
                    seed_lists,
                    J_i,
                    nqubits=nqubits,
                    depth=depth,
                    max_steps=max_steps,
                    learning_rate=learning_rate,
                    gate=gate,
                    data_header=data_header,
                    state_0=state_0,
                )

    # Create plots if requested
    if PLOT:
        plt.style.use("science")
        plt.rcParams.update({"font.size": 15})
        fig, axs = plt.subplots(1, 2)
        left, bottom, width, height = [0.35, 0.7, 0.2, 0.2]
        ax_inset_0 = inset_axes(axs[0], width="30%", height="30%", loc='upper right')
        ax_inset_1 = inset_axes(axs[1], width="30%", height="30%", loc='upper right')
        inset_axs = [ax_inset_0, ax_inset_1]
        fig.set_size_inches(9, 5)

        # Add circuit diagram
        state_0_list = ['random', 'zero', 'phi_+', 'phi_-']
        state_0_labels = ['Haar', r'$|0\rangle$', r'$|\Psi_+\rangle$', r'$|\Psi_-\rangle$', r'$|\Phi_+\rangle$',
                          r'$|\Phi_-\rangle$', r'$|\Phi_-(\mathrm{single})\rangle$']
        gate_list = ['hom', 'equiv']
        data = np.zeros((len(state_0_list), len(gate_list), len(seed_lists), max_steps + 1))
        total_spin = np.zeros((len(state_0_list), len(gate_list), len(seed_lists), 51, 2))
        for i, state_0 in enumerate(state_0_list):
            for j, gate in enumerate(gate_list):
                data_header = gate
                data_path = f"./data/{chain_type}_su2_state_{state_0}/{data_header}/{nqubits}/{depth}"
                for k, seed in enumerate(seed_lists):
                    try:
                        gs_energy = np.load(f"{data_path}/gs_energy_{seed}.npy")

                        curve = np.load(f"{data_path}/cost_{seed}.npy") - min(gs_energy)
                        spin = np.load(f"{data_path}/spin_per_step_{seed}.npy")
                        data[i, j, k] = np.abs(curve).flatten()
                        total_spin[i, j, k] = spin
                    except FileNotFoundError:
                        print(f"File {data_path}/cost_{seed}.npy not found, skipping...")

        for i, state_0 in enumerate(state_0_list):
            for j, gate in enumerate(gate_list):
                # for k, seed in enumerate(seed_list):
                if state_0 == 'psi_-':
                    prev_plot = axs[j].plot(list(range(1, max_steps + 2)), np.mean(data[i, j], axis=0),
                                            label=state_0_labels[i], linewidth=2,
                                            marker='.',
                                            markevery=50, markersize=10, linestyle='dashed' if j == 1 else '-')
                    inset_axs[j].plot(np.mean(total_spin[i, j], axis=0)[:, 0] + 1,
                                      np.mean(total_spin[i, j], axis=0)[:, 1],
                                      marker='.', markevery=5, markersize=10, label=state_0_labels[i], linewidth=2,
                                      linestyle='dashed' if j == 1 else '-',
                                      color=prev_plot[0].get_color())

                else:
                    prev_plot = axs[j].plot(list(range(1, max_steps + 2)), np.mean(data[i, j], axis=0),
                                            label=state_0_labels[i], linewidth=2,
                                            linestyle='dashed' if j == 1 else '-')
                    inset_axs[j].plot(np.mean(total_spin[i, j], axis=0)[:, 0] + 1,
                                      np.mean(total_spin[i, j], axis=0)[:, 1],
                                      label=state_0_labels[i], linewidth=2, linestyle='dashed' if j == 1 else '-',
                                      color=prev_plot[0].get_color())
        for ax in axs:
            ax.set_xlabel("$N_{\mathrm{S}}$")
            ax.set_xscale("log")
            ax.set_ylim([1e-3, 5e3])
            ax.grid()

        for ax in inset_axs:
            ax.set_ylabel(r'$\langle S^2 \rangle$')
            ax.set_xscale('log')

        axs[0].set_yscale("log")
        axs[0].set_ylabel(r"$\Delta \bar{E}$")
        axs[0].tick_params(axis="x", which='both', top=False, right=False)
        axs[0].tick_params(axis="y", which='both', top=False, right=False)
        axs[0].legend(prop={'size': 12}, loc='lower left')

        axs[1].set_yscale("log")
        axs[1].set_yticks(())
        axs[0].tick_params(axis="x", which='both', top=False, right=False)
        axs[1].tick_params(axis="y", which='both', top=False, right=False, left=False)

        # axs[0].set_title(r'(a) Horizontal', y=-0.3)
        # axs[1].set_title(r'(b) Equivariant', y=-0.3)
        axs[0].set_title(r'Horizontal')
        axs[1].set_title(r'Equivariant')

        # axs[2].set_ylabel(r'$\langle S^2 \rangle$', y=-0.3)
        fig.subplots_adjust(wspace=0.0)
        # position = axs[2].get_position()
        # position.x0 = position.x0 + 0.05  # Adjust this value to control the distance
        # axs[2].set_position(position)

        # Save plot
        # plt.tight_layout()
        fig.savefig(f"./figures/FIG1_{chain_type}_{nqubits}_{depth}_qubit_comparison.pdf", dpi=300)
        plt.show()
