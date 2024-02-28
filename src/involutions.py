import numpy as np

"""All cartan involutions for the symmetric spaces"""


def build_basis(b, N):
    b_real = b.real
    b_imag = b.imag
    basis = []
    for i in range(N):
        for j in range(i, N):
            if ~np.isclose(b_real[i, j], 0.0):
                b = np.zeros([N] * 2)
                b[i, j] = b_real[i, j]
                b[j, i] = b_real[j, i]
                basis.append(b)
                if i == j:
                    basis[-1] *= np.sqrt(2)
            if ~np.isclose(b_imag[i, j], 0.0):
                b = np.zeros([N] * 2)
                b[i, j] = b_imag[i, j]
                b[j, i] = b_imag[j, i]
                basis.append(1j * b)
                if i == j:
                    basis[-1] *= np.sqrt(2)

    return -1j*np.stack(basis)


def get_m_basis(wires, inv_type, p, q, g='su'):
    # Get the function for the involution
    n = len(wires)
    N = 2 ** n
    involution = involution_types[inv_type]
    # Get a skew-hermitian matrix of ones
    ones = np.ones([N] * 2)
    if g == 'su':
        A = 1j * ones + (np.tril(ones) - np.triu(ones))
    elif g == 'so':
        A = np.tril(ones) - np.triu(ones)
    else:
        raise NotImplementedError
    # perform the involution
    theta_A = involution(A, p, q)
    # m is the negative eigenspace
    m = 0.5 * (A - theta_A)
    return build_basis(m, N)


def get_k_basis(wires, inv_type, p, q, g='su'):
    # Get the function for the involution
    n = len(wires)
    N = 2 ** n
    involution = involution_types[inv_type]
    # Get a skew-hermitian matrix of ones
    ones = np.ones([N] * 2)
    if g == 'su':
        A = 1j * ones + (np.tril(ones) - np.triu(ones))
    elif g == 'so':
        A = np.tril(ones) - np.triu(ones)
    else:
        raise NotImplementedError
    # perform the involution
    theta_A = involution(A, p, q)
    # m is the negative eigenspace
    k = 0.5 * (A + theta_A)
    return build_basis(k, N)


def involution_AI(A: np.ndarray, p: int, q: int) -> np.ndarray:
    return A.conj()


def involution_AII(A: np.ndarray, p: int, q: int) -> np.ndarray:
    n = A.shape[0] // 2
    zero_n = np.zeros((n, n))
    Jn = np.block([[zero_n, np.eye(n)],
                   [-np.eye(n), zero_n]])
    return Jn @ A.conj() @ Jn.T


def involution_AIII(A: np.ndarray, p: int, q: int) -> np.ndarray:
    Ip = np.eye(p)
    Iq = np.eye(q)
    zero_pq = np.zeros((p, q))
    Ipq = np.block([[-Ip, zero_pq],
                    [zero_pq.T, Iq]])
    return Ipq @ A @ Ipq


def involution_BDI(A: np.ndarray, p: int, q: int) -> np.ndarray:
    Ip = np.eye(p)
    Iq = np.eye(q)
    zero_pq = np.zeros((p, q))
    Ipq = np.block([[-Ip, zero_pq],
                    [zero_pq.T, Iq]])
    return Ipq @ A @ Ipq


def involution_CI(A: np.ndarray, p: int, q: int):
    return A.conj()


def involution_CII(A: np.ndarray, p: int, q: int):
    I_p = np.eye(p)
    I_q = np.eye(q)
    top_left = -I_p
    top_right = np.zeros((p, q))
    bottom_left = np.zeros((q, p))
    bottom_right = I_q
    block = np.block([[top_left, top_right],
                      [bottom_left, bottom_right]])
    Kpq = np.block([[block, np.zeros((p + q, p + q))],
                    [np.zeros((p + q, p + q)), block]])
    return Kpq @ A @ Kpq


def involution_DIII(A: np.ndarray, p: int, q: int) -> np.ndarray:
    n = A.shape[0] // 2
    zero_n = np.zeros((n, n))
    Jn = np.block([[zero_n, np.eye(n)],
                   [-np.eye(n), zero_n]])
    return Jn @ A @ Jn.T


involution_types = {'AI': involution_AI, 'AII': involution_AII, 'AIII': involution_AIII, 'BDI': involution_BDI,
                    'CI': involution_CI, 'CII': involution_CII, 'DIII': involution_DIII}
