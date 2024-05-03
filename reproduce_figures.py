import os

from FIG1_su2 import main as fig1_main
from FIG2_compare_gates import main as fig2_main
from FIG3_blochsphere import main as fig3_main

if __name__ == '__main__':
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    # Reproduce figure 1
    fig1_main(run=True, plot=True, chain_type='random_chain')
    fig1_main(run=True, plot=True, chain_type='uniform_chain')
    # Reproduce figure 2
    N = 8
    depth = 4
    if N > 8:
        print("For N>8 it may take a long time to get the data")
    for seed_i in range(100):  # Get the data for all the seeds
        fig2_main('gue', seed_i, nqubits=N, depth=depth, run=True, plot=False)
        fig2_main('goe', seed_i, nqubits=N, depth=depth, run=True, plot=False)
    fig2_main('gue', None, nqubits=N, depth=depth, run=False, plot=True)
    fig2_main('goe', None, nqubits=N, depth=depth, run=False, plot=True)
    # Reproduce figure 3
    fig3_main()
