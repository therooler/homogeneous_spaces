from tqdm import tqdm

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

"""Run the VQE example in the paper and create the corresponding plots."""

import jax
import jax.numpy as jnp
from src.reductive_geometry import HomogeneousSpace, get_pauli_basis, comm
from src.involutions import get_k_basis

jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt


def get_random_observable(N: int, basis):
    c = np.random.randn(4 ** N - 1)
    return sum(c[i] * basis[i] for i in range(len(basis)))


def get_random_state(N: int):
    state = np.random.randn(2 ** N) + 1j * np.random.randn(2 ** N)
    return state / np.linalg.norm(state)


def construct_operator(basis, coeffs):
    coeffs = jnp.array(coeffs).flatten()
    return jnp.einsum("ijk,i->jk", basis, coeffs)


def get_U(theta, basis):
    H = jnp.tensordot(theta, basis, axes=[[-1], [0]])
    return jax.scipy.linalg.expm(1j * H)


@jax.jit
def get_rho(psi):
    return jnp.outer(psi, psi.conj().T)


@jax.jit
def cost_fn(O, psi):
    return jnp.real(psi.conj().T @ O @ psi)


def main_su4_su3():
    n = 2
    # SU(N) / SU(N-1)
    k = get_k_basis(list(range(n)), 'AIII', 2 ** n - 1, 1) / np.sqrt(2)
    hom_space = HomogeneousSpace(n, k)
    su_basis = jnp.array(list(get_pauli_basis(n).values()))
    np.random.seed(145)
    # Random problem and initialization
    obs = get_random_observable(n, su_basis)
    assert np.allclose(obs, obs.conj().T)
    state = get_random_state(n)
    energies = np.linalg.eigh(obs)[0]
    gs_energy = energies[0]
    print(f"Spectrum {energies}")
    print(f'Minimum at {gs_energy}')
    costs = [cost_fn(obs, state)]
    lr = 1e-1
    n_iter = 1000
    with tqdm(total=n_iter) as pbar:
        for step in range(n_iter):
            rho = get_rho(state)
            grad_f = comm(rho, obs)
            assert np.allclose(grad_f, -grad_f.conj().T)
            grad_f_m = hom_space.m.project_basis(grad_f)
            assert np.allclose(grad_f_m, -grad_f_m.conj().T)
            riemmannian_gradient = jax.scipy.linalg.expm(-lr * grad_f_m)
            state = riemmannian_gradient @ state
            costs.append(cost_fn(obs, state))
            pbar.set_postfix_str(f"Loss = {np.abs(costs[-1] - gs_energy)}")
            pbar.update()


def main_su():
    n = 2
    paulis = jnp.array(list(get_pauli_basis(n).values()))
    basis = paulis

    np.random.seed(145)
    # Random problem and initialization
    obs = get_random_observable(n, basis)
    assert np.allclose(obs, obs.conj().T)
    state = get_random_state(n)
    energies = np.linalg.eigh(obs)[0]
    gs_energy = energies[0]
    print(f"Spectrum {energies}")
    print(f'Minimum at {gs_energy}')
    costs = [cost_fn(obs, state)]
    lr = 1e-2
    n_iter = 1000
    with tqdm(total=n_iter) as pbar:
        for step in range(n_iter):
            rho = get_rho(state)
            grad_f = comm(rho, obs)
            riemmannian_gradient = jax.scipy.linalg.expm(lr * grad_f)
            state = riemmannian_gradient @ state
            costs.append(cost_fn(obs, state))
            pbar.set_postfix_str(f"Loss = {np.abs(costs[-1] - gs_energy)}")
            pbar.update()


def main_heisenberg():
    n = 2
    paulis = jnp.array(list(get_pauli_basis(1).values()))
    # SU(N) / SU(2)
    k = np.array([np.kron(paulis[0], paulis[0]),
                  np.kron(paulis[1], paulis[1]),
                  np.kron(paulis[2], paulis[2])])
    hom_space = HomogeneousSpace(n, k)
    np.random.seed(145)
    # Random problem and initialization
    obs = np.kron(paulis[0], paulis[0]) + np.kron(paulis[1], paulis[1]) + np.kron(paulis[2], paulis[2])
    assert np.allclose(obs, obs.conj().T)
    state = get_random_state(n)
    energies = np.linalg.eigh(obs)[0]
    gs_energy = energies[0]
    print(f"Spectrum {energies}")
    print(f'Minimum at {gs_energy}')
    costs = [cost_fn(obs, state)]
    lr = 1e-1
    n_iter = 1000
    with tqdm(total=n_iter) as pbar:
        for step in range(n_iter):
            rho = get_rho(state)
            grad_f = comm(rho, obs)
            assert np.allclose(grad_f, -grad_f.conj().T)
            grad_f_m = hom_space.m.project_basis(grad_f)
            assert np.allclose(grad_f_m, -grad_f_m.conj().T)
            riemmannian_gradient = jax.scipy.linalg.expm(-lr * grad_f_m)
            state = riemmannian_gradient @ state
            costs.append(cost_fn(obs, state))
            res = np.abs(costs[-1] - gs_energy)
            pbar.set_postfix_str(f"Loss = {res}")
            pbar.update()
            if res < 1e-6:
                break
    plt.plot(costs)
    plt.show()


if __name__ == "__main__":
    # main_su()
    # main_su4_su3()
    main_heisenberg()
