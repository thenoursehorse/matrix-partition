
import argparse
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

from autopartition import AutoPartition
from operators import *
from hamiltonians import ising_chain


if __name__ == '__main__':
    print("Ising: N chain length, g tranvserse field. alpha power law coupling.")
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5)
    parser.add_argument('-g', type=float, default=1)
    parser.add_argument('-alpha', type=float, default=10000)
    parser.add_argument('-time', type=float, default=1)
    parser.add_argument('-with-qutip', type=int, default=0)
    parser.add_argument('-print', type=int, default=1)
    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(args)

    # Define Hamiltonian
    N = args.N
    if args.with_qutip > 0:
        H = ising_chain(N=args.N, g=args.g, alpha=args.alpha).data.todense()
    else:
        H = ising_chain(N=args.N, g=args.g, alpha=args.alpha)

    # Obtain unitary at time t
    t = np.pi/4.
    U = scipy.linalg.expm(-1j*H*args.time)

    print()
    print("Partitioning in computational basis:")

    ising = AutoPartition(U)

    if args.print > 0:
        # Weight plot of matrix elements in block-diagonal permutation
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ising.plot(ax=ax)
        ax.set_xlabel('States')
        ax.set_ylabel('States')
        plt.show()