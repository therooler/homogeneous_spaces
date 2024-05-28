"""Functions to run the VQE example in the paper and create the corresponding plots."""
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from src.involutions import involution_types, get_m_basis
from src.symmetric_gate import SymmetricGate
from tqdm import tqdm
import optax
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""Run the VQE example in the paper and create the corresponding plots."""
import os
from functools import partial


def apply_op_in_layers(theta, Op, depth, nqubits):
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
            Op(theta[d, idx], wires=[i, i + 1])
            if i == nqubits - 2:
                idx += 1

        # Odd-even qubit pairs
        for i in range(1, nqubits, 2):
            Op(theta[d, idx], wires=[i, (i + 1) % nqubits])


def gate_decomp_su4(params, wires):
    """Apply a sequence of 15 single-qubit rotations and three ``CNOT`` gates to
    compose an arbitrary SU(4) operation.

    Args:
        params (tensor_like): Parameters for single-qubit rotations, expected to have shape (15,)
        wires (list[int]): Wires on which to apply the gate sequence

    See Theo. 5 in https://arxiv.org/pdf/quant-ph/0308006.pdf for more details.
    """
    i, j = wires
    # Single U(2) parameterization on qubit 1
    qml.RY(params[0], wires=i)
    qml.RX(params[1], wires=i)
    qml.RY(params[2], wires=i)
    # Single U(2) parameterization on qubit 2
    qml.RY(params[3], wires=j)
    qml.RX(params[4], wires=j)
    qml.RY(params[5], wires=j)
    # CNOT with control on qubit 2
    qml.CNOT(wires=[j, i])
    # Rz and Ry gate
    qml.RZ(params[6], wires=i)
    qml.RY(params[7], wires=j)
    # CNOT with control on qubit 1
    qml.CNOT(wires=[i, j])
    # Ry gate on qubit 2
    qml.RY(params[8], wires=j)
    # CNOT with control on qubit 2
    qml.CNOT(wires=[j, i])
    # Single U(2) parameterization on qubit 1
    qml.RY(params[9], wires=i)
    qml.RX(params[10], wires=i)
    qml.RY(params[11], wires=i)
    # Single U(2) parameterization on qubit 2
    qml.RY(params[12], wires=j)
    qml.RX(params[13], wires=j)
    qml.RY(params[14], wires=j)


def gate_decomp_so4(params, wires):
    """Apply a sequence of 6 single-qubit rotations and three ``CNOT`` gates to
    compose an arbitrary SO(4) operation.

    Args:
        params (tensor_like): Parameters for single-qubit rotations, expected to have shape (15,)
        wires (list[int]): Wires on which to apply the gate sequence

    See Theo. 3 in https://arxiv.org/pdf/quant-ph/0308006.pdf for more details.
    """
    i, j = wires
    # Single S1 block
    qml.RZ(np.pi / 2, wires=i)
    qml.RZ(np.pi / 2, wires=j)
    # Single R1 block
    qml.RY(np.pi / 2, wires=j)
    # CNOT with control on qubit 2
    qml.CNOT(wires=[j, i])
    # Rz and Ry gate
    # Single U(2) parameterization on qubit 1
    qml.RY(params[0], wires=i)
    qml.RX(params[1], wires=i)
    qml.RY(params[2], wires=i)
    # Single U(2) parameterization on qubit 2
    qml.RY(params[3], wires=j)
    qml.RX(params[4], wires=j)
    qml.RY(params[5], wires=j)
    # CNOT with control on qubit 2
    qml.CNOT(wires=[j, i])
    # Single R1* block
    qml.RY(-np.pi / 2, wires=j)
    # Single S1* block
    qml.RZ(-np.pi / 2, wires=i)
    qml.RZ(-np.pi / 2, wires=j)


def make_observable_gue(wires, seed):
    """Generate a random Hermitian matrix from the Gaussian Unitary Ensemble (GUE), as well
    as a ``qml.Hermitian`` observable.

    Args:
        wires (list[int]): Wires on which the observable should be measured.
        seed (int): Seed for random matrix sampling.

    Returns:
        Hermitian: The Hermitian observable
        tensor_like: The matrix of the observable

    For ``n`` entries in ``wires``, the returned observable matrix has size ``(2**n, 2**n)``.
    """
    np.random.seed(seed)
    num_wires = len(wires)
    d = 2 ** num_wires
    # Random normal complex-valued matrix
    mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    # Make the random matrix Hermitian and divide by two to match the GUE.
    observable_matrix = (mat + mat.conj().T) / 2
    return qml.Hermitian(observable_matrix, wires=wires), observable_matrix


def make_observable_goe(wires, seed):
    """Generate a random Hermitian matrix from the Gaussian Orthogonal Ensemble (GOE), as well
    as a ``qml.Hermitian`` observable.

    Args:
        wires (list[int]): Wires on which the observable should be measured.
        seed (int): Seed for random matrix sampling.

    Returns:
        Hermitian: The Hermitian observable
        tensor_like: The matrix of the observable

    For ``n`` entries in ``wires``, the returned observable matrix has size ``(2**n, 2**n)``.
    """
    np.random.seed(seed)
    num_wires = len(wires)
    d = 2 ** num_wires
    # Random normal real-valued matrix
    mat = np.random.randn(d, d)  # + 1j * np.random.randn(d, d)
    # Make the random matrix Symmetric and divide by two to match the GOE.
    observable_matrix = (mat + mat.T) / 2
    return qml.Hermitian(observable_matrix, wires=wires), observable_matrix


def symplectic_check(mat):
    n = mat.shape[0] // 2
    zero_n = np.zeros((n, n))
    Jn = np.block([[zero_n, np.eye(n)],
                   [-np.eye(n), zero_n]])
    assert np.allclose(Jn @ mat, -mat.T @ Jn)


def make_observable_gse(wires, seed):
    """Generate a random Hermitian matrix from the Gaussian Symplectic Ensemble (GSE), as well
    as a ``qml.Hermitian`` observable. See http://assets.press.princeton.edu/chapters/s9237.pdf (Definition 1.3.2)

    Args:
        wires (list[int]): Wires on which the observable should be measured.
        seed (int): Seed for random matrix sampling.

    Returns:
        Hermitian: The Hermitian observable
        tensor_like: The matrix of the observable

    For ``n`` entries in ``wires``, the returned observable matrix has size ``(2**n, 2**n)``.
    """
    np.random.seed(seed)
    num_wires = len(wires)
    d = 2 ** num_wires // 2
    # Random normal real-valued matrix
    x = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    x = x - x.T.conj()
    y = 1j * np.random.randn(d, d)
    y = y + y.T
    # Make the random matrix Symmetric and divide by two to match the GSE.
    observable_matrix = -1j * np.block([[x, y],
                                        [y, x.conj()]]) / 2
    return qml.Hermitian(observable_matrix, wires=wires), observable_matrix


def vqe(
        seed,
        nqubits=None,
        depth=None,
        opname=None,
        p_q=(None, None),
        max_steps=None,
        learning_rate=None,
        data_header="",
        cost_fn='gue'
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
    if p_q[0] is not None:
        data_path = f"./data/{data_header}{cost_fn}/{nqubits}/{depth}/{opname}_{p_q[0]}_{p_q[1]}/"
    else:
        data_path = f"./data/{data_header}{cost_fn}/{nqubits}/{depth}/{opname}/"
    # Set the differentiation method to use backpropagation
    diff_method = "backprop"
    # Use the DefaultQubit PennyLane device
    dev_type = "default.qubit"
    # Choose the operation to use in the circuit ansatz
    if opname == "decomp":
        if cost_fn == 'gue':
            Op = gate_decomp_su4
            parameters_per_gate = 15
        if cost_fn == 'goe':
            Op = gate_decomp_so4
            parameters_per_gate = 6
        if cost_fn == 'gse':
            raise NotImplementedError
    elif opname in involution_types.keys():
        Op = SymmetricGate
        p, q = p_q
        parameters_per_gate = get_m_basis((0, 1), opname, p, q).shape[0]
    elif opname is None:
        raise ValueError
    dev = qml.device(dev_type, wires=nqubits)

    if cost_fn == 'gue':
        observable, observable_matrix = make_observable_gue(dev.wires, seed)
    elif cost_fn == 'goe':
        observable, observable_matrix = make_observable_goe(dev.wires, seed)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def cost_function(params):
        """Cost function of the VQE."""
        if opname != "decomp":
            def op_inv(theta, wires):
                return Op(theta, wires, opname, p, q)

            apply_op_in_layers(params, op_inv, depth, nqubits)
        else:
            apply_op_in_layers(params, Op, depth, nqubits)
        return qml.expval(observable)

    grad_function = jax.jit(jax.grad(cost_function))
    cost_function = jax.jit(cost_function)

    # Store the ground state and maximal energy of the created observable on disk
    if not os.path.exists(data_path + f"gs_energy_{seed}.npy"):
        energies = np.linalg.eigvalsh(observable_matrix)
        gs_energy = energies[0]
        max_energy = energies[-1]
        np.save(data_path + f"gs_energy_{seed}.npy", gs_energy)
        np.save(data_path + f"max_energy_{seed}.npy", max_energy)

    cost_path = data_path + f"cost_{seed}.npy"

    # Check whether the optimization curves already exist on disk
    if not os.path.exists(cost_path):
        # Parameter shape: depth x (2 layers) x (15 = 4**2-1)
        shape = (depth, 2, parameters_per_gate)
        # Create initial parameters
        params = jax.numpy.zeros(shape)
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(params)
        cost_history = []
        for step in tqdm(range(max_steps), mininterval=1):
            # Record old cost
            cost_history.append(cost_function(params))
            grads = grad_function(params)
            # Make step
            updates, opt_state = optimizer.update(grads, opt_state)
            if step > 100 and (step + 1) % 100:
                if np.isclose(cost_history[-1], cost_history[-100], atol=1e-7):
                    print(f'Early stopping at step: {step}')
                    break
            params = optax.apply_updates(params, updates)
        # Record final cost
        cost_history.append(cost_function(params))
        cost_history.extend([cost_history[-1]] * (max_steps - len(cost_history) + 1))
        # Save cost to disk
        np.save(cost_path, np.array(cost_history))
    else:
        # Load cost from disk
        print(f"{cost_path} exists, loaded data!")


def load_data(nqubits, depth, seed_list, max_steps, data_header, opname="decomp", p_q=(None, None), cost_fn='gue'):
    """Load the VQE optimization curve data and process it into
    relative energy errors.

    Args:
        nqubits (int): Number of qubits
        depth (list[int]): All depth to show
        seed_list (list[int]): The randomness seeds for all runs
        max_steps (int): The number of optimization steps
        data_header (str): Subdirectory of ``./data`` to load data from

    Returns:
        tensor_like: Data for gate sequence operation
        tensor_like: Data for SU(N) gate

    """

    if p_q[0] is not None:
        data_path = f"./data/{data_header}{cost_fn}/{nqubits}/{depth}/{opname}_{p_q[0]}_{p_q[1]}/"
    else:
        data_path = f"./data/{data_header}{cost_fn}/{nqubits}/{depth}/{opname}/"

    data = np.zeros((len(seed_list), max_steps + 1))

    for i, seed in enumerate(seed_list):
        try:
            gs_energy = np.load(f"{data_path}/gs_energy_{seed}.npy")
            max_energy = np.load(f"{data_path}/max_energy_{seed}.npy")
            spectrum_width = max_energy - gs_energy
            data[i] = (np.load(f"{data_path}/cost_{seed}.npy") - gs_energy) / spectrum_width
        except FileNotFoundError:
            # Just skip files that were not found
            data[i] = [np.nan] * (max_steps + 1)
            print(f"File {data_path}/cost_{seed}.npy not found, skipping...")

    return data


def plot_optim_curves(nqubits, depth, gates, seed_list, max_steps, data_header, cost_fn):
    """Plot the relative energy error optimization curves of the VQE runs.

    Args:
        nqubits (int): Number of qubits
        depth (int): Depth of the circuit ansatz
        seed_list (list[int]): Randomness seeds for all VQE runs
        max_steps (int): Number of optimization steps
        data_header (str): Subdirectory of ``./data`` to load data from
    """
    plt.style.use("science")
    plt.rcParams.update({"font.size": 15})
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 4)

    # Load data
    data = []
    n_params = []
    for (opname, p_q) in gates:
        data.append(
            load_data(nqubits, depth, seed_list, max_steps, data_header, opname=opname, p_q=p_q, cost_fn=cost_fn))
        if opname != 'decomp':
            b = get_m_basis((0, 1), opname, p_q[0], p_q[1])
            n_params.append(b.shape[0] * 2 * depth)
        else:
            n_params.append(15 * 2 * depth)

    colors = [plt.get_cmap('Blues'),
              plt.get_cmap('Reds'),
              plt.get_cmap('Greens'),
              plt.get_cmap('Oranges'),
              plt.get_cmap('Purples')]
    num_seeds = len(seed_list)
    label_names = {'decomp': 'Decomp.',
                   'AIII': r"$\mathrm{SU}(4)/\mathrm{U}(3)$",
                   'BDI': r"$\mathrm{SO}(4)/\mathrm{O}(3)$"}
    for i in range(num_seeds):
        for j, (opname, p_q) in enumerate(gates):
            if i == (num_seeds // 2):
                axs.plot(np.array(list(range(max_steps + 1))), data[j][i],
                         linewidth=1, color=colors[j]((i + 1) / num_seeds),
                         label=label_names[opname])
            else:
                axs.plot(np.array(list(range(max_steps + 1))), data[j][i],
                         linewidth=1, color=colors[j]((i + 1) / num_seeds))
    axs.legend(loc='lower left')
    axs.set_xscale('log')
    axs.set_ylim([0.3, 0.65])
    axs.set_xlabel(r"Step")
    axs.set_ylabel(r"$\bar{E}$")
    # axs.grid()
    plt.tight_layout()
    left, bottom, width, height = [0.5, 0.7, 0.4, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    # sort_order = np.argsort(data[0][:, -1])
    ax2.plot(list(range(1, num_seeds + 1)),
             np.sort(data[0][:, -1] - data[1][:, -1]), color=colors[2](0.7), linewidth=1, marker='.')
    ax2.plot(list(range(1, num_seeds + 1)), [0.0] * num_seeds, color='gray', linestyle='--')
    ax2.set_ylabel(r"$\Delta\Bar{E}_{\mathrm{final}}$")
    ax2.set_xlabel(r"Instance")
    # ax2.set_yscale(r"log")
    # ax2.set_ylim([1e-6, 1e-1])

    for ax in [axs, ax2]:
        ax.tick_params(axis="x", which='both', top=False, right=False)
        ax.tick_params(axis="y", which='both', top=False, right=False)
    fig.savefig(f"./figures/FIG2_{nqubits}_qubit_{depth}_{cost_fn}_trajectories_quotients.pdf")
    plt.show()


def main(cost_fn, seed_idx, nqubits, depth, run=True, plot=False):
    # Choose the following settings
    #  These are the settings used for the paper
    nruns = 100  # Number of VQE runs
    max_steps = 10000  # number of optimization steps in each VQE
    learning_rate = 1e-2  # Learning rate for gradient descent in the optimization
    RUN = run  # Whether to run the computation. If results are present, computations are skipped
    print(f"Cost fn = {cost_fn}")
    print(f"Nruns = {nruns}")
    print(f"max_steps = {max_steps}")
    print(f"learning_rate = {learning_rate:1.3f}")
    print(f"Nqubits = {nqubits}")
    print(f"Depth = {depth}")
    # Directory name to save results to. They will be in f"./data/{data_header}/"
    data_header = ""
    data_header = data_header.strip("/")
    # Generate seeds (deterministically)
    seed_list = [i * 37 for i in range(nruns)]
    if seed_idx is not None:
        seed_list = [seed_list[seed_idx]]
    # Store the global variables to allow for later investigation of settings
    global_config_path = f"./data/{data_header}/{cost_fn}/{nqubits}/"
    if not os.path.exists(global_config_path):
        os.makedirs(global_config_path)
    if cost_fn == 'gue':
        gates = [("decomp", (None, None)),
                 ("AIII", (1, 3)),
                 # ("AIII", (2, 2)),
                 # ("AII", (None, None))
                 ]
    elif cost_fn == 'goe':
        gates = [("decomp", (None, None)),
                 ("BDI", (1, 3)),
                 # ("BDI", (2, 2)),
                 # ("DIII", (None, None))
                 ]

    # Run computation if requested. If results are present already, computations are skipped
    if RUN:
        for (opname, p_q) in gates:
            # Set up path and create missing directories
            if p_q[0] is not None:
                data_path = f"./data/{data_header}/{cost_fn}/{nqubits}/{depth}/{opname}_{p_q[0]}_{p_q[1]}/"
            else:
                data_path = f"./data/{data_header}/{cost_fn}/{nqubits}/{depth}/{opname}/"
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            # Mappable version of ``vqe``.
            func = partial(
                vqe,
                nqubits=nqubits,
                depth=depth,
                opname=opname,
                p_q=p_q,
                max_steps=max_steps,
                learning_rate=learning_rate,
                data_header=data_header,
                cost_fn=cost_fn
            )
            # Map ``vqe`` across the partial seed lists for parallel execution
            for seed in tqdm(seed_list):
                func(seed)

    # Create plots if requested
    if plot:
        plot_optim_curves(nqubits, depth, gates, seed_list, max_steps, data_header, cost_fn=cost_fn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_idx', default=None)  # If running these scripts in parallel
    parser.add_argument('--N', default=12)
    parser.add_argument('--L', default=12)
    parser.add_argument('--run', default=False)
    args = parser.parse_args()
    seed_idx = args.seed_idx
    if seed_idx is not None:
        seed_idx = int(seed_idx)
        print(f"Running seed {seed_idx}")
        PLOT = False
    else:
        PLOT = True
    main('gue', seed_idx, nqubits=int(args.N), depth=int(args.L), run=bool(args.run), plot=PLOT)
    main('goe', seed_idx, nqubits=int(args.N), depth=int(args.L), run=bool(args.run), plot=PLOT)
