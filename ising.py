
import argparse
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

from src.autopartition import AutoPartition
from operators import *
from hamiltonians import ising_chain, qutip_ising_chain


if __name__ == '__main__':
    print("Ising: N chain length, g tranvserse field. alpha power law coupling.")
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5)
    parser.add_argument('-g', type=float, default=1)
    parser.add_argument('-alpha', type=float, default=10000)
    parser.add_argument('-time', type=float, default=1)
    parser.add_argument('-with-qutip', type=int, default=0)
    parser.add_argument('-plot', type=int, default=1)
    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(args)
    
    if args.plot > 0:
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))

    # Define Hamiltonian
    N = args.N
    if args.with_qutip > 0:
        H = qutip_ising_chain(N=args.N, g=args.g, alpha=args.alpha).data.todense()
    else:
        H = ising_chain(N=args.N, g=args.g, alpha=args.alpha)

    # Obtain unitary at time t
    t = np.pi/4.
    U = scipy.linalg.expm(-1j*H*args.time)

    print()
    print("Partitioning in computational basis:")
    ising = AutoPartition(U)
    ising.set_labels()
    ising.print_labels()
    
    if args.plot > 0:
        ising.plot(ax=ax[0])
        ax[0].set_ylabel('States')
        ax[0].set_xlabel('States')

    print()
    print("Partitioning in parity basis:") 
    P = parity(N)
    symmetry_operators = {'P':P}
    ising.apply_symmetries(symmetry_operators=symmetry_operators)
    ising.set_labels()
    ising.print_labels()

    if args.plot > 0:
        ising.plot(ax=ax[1])
        ax[1].set_ylabel('States')
        ax[1].set_xlabel('States')
    
    if args.plot > 0:
        plt.show()