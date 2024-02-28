"""Functions to run the VQE example in the paper and create the corresponding plots."""
import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np
from src.involutions import involution_types, get_m_basis
from src.symmetric_gate import SymmetricGate
from tqdm import tqdm

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""Run the VQE example in the paper and create the corresponding plots."""
import os
from functools import partial
from itertools import product
from multiprocessing import Pool
from dill import dump

# Choose the following settings

#  These are the settings used for the paper
nruns = 10  # Number of VQE runs
max_steps = 10000  # number of optimization steps in each VQE
learning_rate = 1e-3  # Learning rate for gradient descent in the optimization
nqubits = 6  # Number of qubits
cost_fn = 'goe'  # type of cost
depths = [5, ]  # Number of layers in the circuit ansatz. The VQE is run for each depth
RUN = True  # Whether to run the computation. If results are present, computations are skipped
PLOT = True  # Whether to create plots of the results
num_workers = 1  # Number of threads to use in parallel. Needs to be set machine-dependently

"""
#  These are some settings with much lower computational cost, for illustration and testing
nruns = 4  # Number of VQE runs
max_steps = 1000  # number of optimization steps in each VQE
learning_rate = 1e-3  # Learning rate for gradient descent in the optimization
nqubits = 4  # Number of qubits
depths = [1, 2, 3]  # Number of layers in the circuit ansatz. The VQE is run for each depth
RUN = True  # Whether to run the computation. If results are present, computations are skipped
PLOT = True  # Whether to create plots of the results
num_workers = 4  # Number of threads to use in parallel. Needs to be set machine-dependently
"""

# Directory name to save results to. They will be in f"./data/{data_header}/"
data_header = ""


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
        seed_list,
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

    for seed in tqdm(seed_list):
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

            cost_history = []
            for _ in tqdm(range(max_steps)):
                # Record old cost
                cost_history.append(cost_function(params))
                # Make step
                params = params - learning_rate * grad_function(params)

            # Record final cost
            cost_history.append(cost_function(params))
            # Save cost to disk
            np.save(cost_path, np.array(cost_history))
        else:
            # Load cost from disk
            cost_history = np.load(cost_path)
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
        gs_energy = np.load(f"{data_path}/gs_energy_{seed}.npy")
        max_energy = np.load(f"{data_path}/max_energy_{seed}.npy")
        spectrum_width = max_energy - gs_energy
        try:
            data[i] = (np.load(f"{data_path}/cost_{seed}.npy") - gs_energy) / spectrum_width
        except FileNotFoundError:
            # Just skip files that were not found
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
    for i in range(num_seeds):
        for j, (opname, p_q) in enumerate(gates):
            if i == (num_seeds // 2):
                axs.plot(np.array(list(range(max_steps + 1))) * n_params[j], data[j][i],
                         linewidth=1, color=colors[j]((i + 1) / num_seeds),
                         label=opname + f"{p_q if p_q[0] is not None else ''}")
            else:
                axs.plot(np.array(list(range(max_steps + 1))) * n_params[j], data[j][i],
                         linewidth=1, color=colors[j]((i + 1) / num_seeds))
    axs.legend()
    # axs.set_xscale('log')
    # axs.set_yscale('log')
    axs.set_xlabel(r"Step $\times$ \#Parameters")
    axs.set_ylabel(r"$\bar{E}$")
    plt.tight_layout()
    left, bottom, width, height = [0.35, 0.7, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    for j, (opname, p_q) in enumerate(gates):
        ax2.plot(np.sort(data[j][:, -1]), color=colors[j](0.7))

    fig.savefig(f"./figures/FIG2_{nqubits}_qubit_{depth}_{cost_fn}_trajectories_quotients.pdf")
    plt.show()


if __name__ == "__main__":
    data_header = data_header.strip("/")
    # Generate seeds (deterministically)
    runs_per_worker = nruns // num_workers
    seed_lists = [
        [i * 37 for i in range(j * runs_per_worker, (j + 1) * runs_per_worker)]
        for j in range(num_workers)
    ]

    # Store the global variables to allow for later investigation of settings
    global_config_path = f"./data/{data_header}/{cost_fn}/{nqubits}/"
    if not os.path.exists(global_config_path):
        os.makedirs(global_config_path)
    with open(global_config_path + "globals.dill", "wb") as file:
        dump(globals(), file)
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
        for depth, (opname, p_q) in product(depths, gates):
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
            with Pool(num_workers) as p:
                p.map(func, seed_lists)

    # Create plots if requested
    if PLOT:
        seed_list = sum(seed_lists, start=[])
        plot_optim_curves(nqubits, max(depths), gates, seed_list, max_steps, data_header, cost_fn=cost_fn)
