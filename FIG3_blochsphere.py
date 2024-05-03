import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

I = np.eye(2).astype(complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)


def bloch_sphere_coordinates(psi):
    alpha = psi[0]
    beta = psi[1]
    u = beta / alpha
    ux = u.real
    uy = u.imag
    px = 2 * ux / (1 + ux ** 2 + uy ** 2)
    py = 2 * uy / (1 + ux ** 2 + uy ** 2)
    pz = (1 - ux ** 2 - uy ** 2) / (1 + ux ** 2 + uy ** 2)
    return px, py, pz


def U(x, y):
    return scipy.linalg.expm(1j * (x * X + y * Y))


def K(t):
    return scipy.linalg.expm(1j * t * Z)


def psiU(x, y):
    return U(x, y) @ np.array([1.0, 0.], complex)


def theta_rot(t):
    return np.array([[np.cos(2 * t), -np.sin(2 * t)],
                     [np.sin(2 * t), np.cos(2 * t)]])


def get_cosets(theta_list, dt=0.1, gran=10):
    cosets = []
    for x, y in theta_list:
        cosets.append(np.stack(
            [bloch_sphere_coordinates(K(t).conj().T @ U(x, y) @ K(t) @ np.array([1.0, 0.])) for t in
             np.linspace(-dt, dt, gran)], axis=1))
    return cosets


def main():
    fig = plt.figure()
    fig.set_size_inches(4, 8)
    ax = fig.add_subplot(2, 1, 1, projection='3d')

    # Create a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    r = 1.
    x = np.outer(np.cos(u), np.sin(v)) * r
    y = np.outer(np.sin(u), np.sin(v)) * r
    z = np.outer(np.ones(np.size(u)), np.cos(v)) * r

    # Plot the sphere
    ax.plot_surface(x, y, z, color='black', alpha=0.3)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    gran = 20

    theta_1_1 = np.linspace(0, np.pi / 2, gran)
    theta_2_1 = np.zeros(gran)

    theta_1_2 = np.zeros(gran)
    theta_2_2 = np.linspace(0, np.pi / 2, gran)

    theta_1_3 = np.linspace(0, np.sqrt(1. / 2) * np.pi / 2, gran)
    theta_2_3 = np.linspace(0, np.sqrt(1. / 2) * np.pi / 2, gran)

    colors = ['red', 'green', 'blue']

    ax2d = fig.add_subplot(2, 1, 2)

    t = np.linspace(-0.1, 0.1, 10)
    for i, (t1_list, t2_list) in enumerate(zip([theta_1_1, theta_1_2, theta_1_3],
                                               [theta_2_1, theta_2_2, theta_2_3])):
        color = colors[i]
        coords = np.stack([bloch_sphere_coordinates(psiU(x, y)) for x, y in zip(t1_list, t2_list)], axis=1)
        for t1, t2 in zip(t1_list, t2_list):
            theta = np.array([t1, t2])
            theta_cosets = np.array([theta_rot(t_i) @ theta for t_i in t]).T
            ax2d.plot(theta_cosets[0], theta_cosets[1], color='gray')
            coset = np.stack([bloch_sphere_coordinates(psiU(x, y)) for x, y in theta_cosets.T], axis=1)
            ax.plot(coset[0], coset[1], coset[2], color='gray', linewidth=1)
        ax.plot(coords[0], coords[1], coords[2], color=color, linewidth=1)
        ax2d.plot(t1_list, t2_list, color=color)
    axis_len = 1.5
    ax.plot([-axis_len, axis_len], [0, 0], [0, 0], color='black')
    ax.plot([0, 0], [-axis_len, axis_len], [0, 0], color='black')
    ax.plot([0, 0], [0, 0], [-axis_len, axis_len], color='black')
    ax2d.set_xlabel(r'$\theta_1$')
    ax2d.set_ylabel(r'$\theta_2$')

    # Remove grid lines
    ax.grid(False)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.savefig('./figures/FIG4_coset_sphere.pdf')
    plt.show()


if __name__ == '__main__':
    main()
