import numpy as np
import itertools as it
import scipy.linalg as spla


def get_pauli_basis(n):
    I = np.eye(2).astype(complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    name_lookup = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    basis = []
    names = []
    for comb in list(it.product([0, 1, 2, 3], repeat=n))[1:]:
        p = 1.
        name = []
        for c in comb:
            if c == 0:
                p = np.kron(p, I)
            elif c == 1:
                p = np.kron(p, X)
            elif c == 2:
                p = np.kron(p, Y)
            elif c == 3:
                p = np.kron(p, Z)
            name.append(name_lookup[c])
        basis.append(p / np.sqrt(2 ** n))
        names.append("".join(name))
    return dict(zip(names, basis))


def comm(a, b):
    """Commutator"""
    return a @ b - b @ a


def B(x, y):
    """Killing form"""
    N = x.shape[0]
    return 2 * N * np.trace(x @ y)


def R(x, y, z):
    """Curvature tensor"""
    return 1 / 4 * comm(comm(x, y), z)


def K(x, y):
    """Sectional curvature"""
    return 1 / 4 * B(comm(x, y), comm(x, y)) / (B(x, x) * B(y, y) - B(x, y) ** 2)


class VectorSpace:
    def __init__(self, g):
        self.N = g.shape[1]
        self.g = g

    def project_coeffs(self, op):
        if self.N > 0:
            return np.einsum("ijk, kj->i", self.g, op.conj().T)
        else:
            return np.array([])

    def project_basis(self, op):
        if self.N > 0:
            coeffs = self.project_coeffs(op)
            return np.einsum("i, ijk->jk", coeffs, self.g)
        else:
            return np.array([])

    def construct_op(self, coeffs):
        if self.N > 0:
            return np.einsum("i, ijk->jk", coeffs, self.g)
        else:
            return np.array([])

    @property
    def basis(self):
        return self.g

    @property
    def shape(self):
        return self.g.shape

    def __getitem__(self, item):
        return self.g[item]

    def __iter__(self):
        for g in self.g:
            yield g

    def get_orthogonal_complement(self, g):
        subspace_k = []
        for i, p in enumerate(self.g):
            subspace_k.append(g.project_coeffs(p))
        k_ortho = spla.null_space(np.stack(subspace_k))
        if not k_ortho.size == 0:
            m = np.einsum("in,ijk->njk", k_ortho, g.basis)
            norm = np.trace(m[0] @ m[0])
            return VectorSpace(m / norm)
        else:
            return VectorSpace(np.array([[]]))


class LieAlgebra(VectorSpace):
    def __init__(self, g):
        super(LieAlgebra, self).__init__(g)

    def ricci_curvature(self, x, y):
        res = 0
        for Ei in self.g:
            res += B(comm(x, Ei), comm(y, Ei))
        return 1 / 4 * res

    def sectional_curvature(self, x, y):
        return 1 / 4 * B(comm(x, y), comm(x, y)) / (B(x, x) * B(y, y) - B(x, y) ** 2)


def orthonormal_check(basis):
    delta_tensor = np.einsum("ijk,nkj->in", basis, basis)
    return np.allclose(delta_tensor, np.eye(basis.shape[0]))


class HomogeneousSpace:
    """See page 62 of Arvanitogeorgos """

    def __init__(self, n, k, m=None, g_type='su', orth=True):
        basis = get_pauli_basis(n)
        if g_type == 'su':
            g = np.stack([p for p in basis.values()])
            self.names = list(basis.keys())

        elif g_type == 'so':
            g = []
            self.names = []
            for key, p in basis.items():
                if np.allclose(p, -p.T):
                    g.append(p)
                    self.names.append(key)
            g = np.stack(g)
        else:
            raise NotImplementedError
        if orth:
            assert orthonormal_check(g)
            assert orthonormal_check(k)

        self.N = 2 ** n
        self.g = LieAlgebra(g)
        self.k = LieAlgebra(k)

        # Find m
        if m is None:
            subspace_k = []
            for i, p in enumerate(k):
                subspace_k.append(self.g.project_coeffs(p))
            k_ortho = spla.null_space(np.stack(subspace_k))
            # for element in k_ortho.T:
            #     print(np.nonzero(element))
            #     print(element[np.nonzero(element)[0]])
            m = np.einsum("in,ijk->njk", k_ortho, self.g.basis)
        assert orthonormal_check(m)
        self.m = VectorSpace(m)

        lhs = np.zeros([self.m.shape[0]] * 3, dtype=complex)
        rhs = np.zeros([self.m.shape[0]] * 3, dtype=complex)

        for i, x in enumerate(self.m):
            for m, y in enumerate(self.m):
                for o, z in enumerate(self.m):
                    lhs[i, m, o] = B(self.m.project_basis(comm(x, y)), z)
                    rhs[i, m, o] = B(x, self.m.project_basis(comm(y, z)))

        self._natural = np.allclose(lhs, rhs)
        print("Natural?", self.natural)
        if self._natural:
            def sectional_curvature(x, y):
                comm_xy_k = self.k.project_basis(comm(x, y))
                comm_xy_m = self.m.project_basis(comm(x, y))
                return -B(comm_xy_k, comm_xy_k) / self.N ** 2 - \
                       1 / 4 * B(comm_xy_m, comm_xy_m) / self.N ** 2

            def ricci_curvature(x):
                res = -1 / 4 * B(x, x)
                for Vi in self.k:
                    proj_comm = self.k.project_basis(comm(x, Vi))
                    res -= 0.5 * B(proj_comm, proj_comm)
                return res

        else:
            def sectional_curvature(x, y):
                return -B(self.k.project_basis(comm(x, y)), self.k.project_basis(comm(x, y))) - \
                       1 / 4 * B(self.m.project_basis(comm(x, y)), self.m.project_basis(comm(x, y)))

            def ricci_curvature(x):
                res = -1 / 2 * B(x, x)
                for Vi in self.m:
                    proj_comm_i = self.m.project_basis(comm(x, Vi))
                    res -= 0.5 * B(proj_comm_i, proj_comm_i) ** 2
                    for Vj in self.m:
                        proj_comm_i_j = self.m.project_basis(comm(Vi, Vj))
                        res += 0.25 * B(proj_comm_i_j, x) ** 2

                res -= B(self.m.project_basis(comm(z, x)), x)
                # TODO: Implement Z (Prop 5.5, page 82 Arvanitogeorgos)
                raise NotImplementedError

        self.sectional_curvature = lambda x, y: sectional_curvature(x, y).real
        self.ricci_curvature = lambda x: ricci_curvature(x).real

    @property
    def natural(self):
        return self._natural


def Ad(U):
    return lambda x: U @ x @ U.conj().T


def R_g(U):
    return lambda x: x @ U


def L_g(U):
    return lambda x: U @ x
