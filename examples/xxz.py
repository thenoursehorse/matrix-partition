import argparse
import textwrap
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

from matrix_partition.autopartition import AutoPartition
from operators import *
from hamiltonians import xyz_chain, qutip_xyz_chain

if __name__ == '__main__':
    description = textwrap.dedent('''\
         XXZ spin-1/2 chain:
             N     : number of sites
             Delta : Sz Sz coupling
             alpha :  power law coupling (>1e3 -> nearest-neighbour)
         ''')
    epilog = textwrap.dedent('''\
         Symmetries:
             XXX periodic even number of sites: Sz, total spin (SU(2)), Parity/reflection (P), pi rotation about x (R_x), crystal momentum (C_N)
             XXX periodic odd: Sz, SU(2), P, C_N
             XXX open even: Sz, SU(2), P, R_x
             XXX open odd: Sz, SU(2), P
             XXZ periodic even: Sz, P, R_x, C_N
             XXZ periodic odd: Sz, P, C_N
             XXZ open even: Sz, P, R_x
             XXZ open odd: Sz, P
         ''')
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description,
                                     epilog=epilog)
    print("XXZ: Delta ZZ coupling. alpha power law coupling.")
    parser.add_argument('-N', type=int, default=5)
    parser.add_argument('-Delta', type=float, default=1)
    parser.add_argument('-alpha', type=float, default=10000)
    parser.add_argument('-with-qutip', type=int, default=0)
    parser.add_argument('-plot', type=int, default=1)
    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(vars(args))

    if args.plot > 0:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Define Hamiltonian
    N = args.N
    if args.with_qutip > 0:
        H = qutip_xyz_chain(N=args.N, Delta=args.Delta, alpha=args.alpha).data.todense()
    else:
        H = xyz_chain(N=args.N, Delta=args.Delta, alpha=args.alpha)
    
    print()
    print("Partitioning in computational (Sz) basis:")
    xxz = AutoPartition(H)
    Sz = spin_sum(N=N, axis='z')
    symmetry_operators = {'Sz':Sz}
    xxz.apply_symmetries(symmetry_operators=symmetry_operators)
    xxz.set_labels()
    xxz.print_labels()
    
    if args.plot > 0:
        xxz.plot(ax=ax[0,0])
        ax[0,0].set_title(r'$S^z$ eigenbasis')

    print()
    print("Partitioning in Sz, S2 basis:")
    xxz = AutoPartition(H)
    S2 = spin_S2(N=N)
    symmetry_operators = {'Sz':Sz, 'S2':S2}
    xxz.apply_symmetries(symmetry_operators=symmetry_operators)
    xxz.set_labels()
    xxz.print_labels()
    
    if args.plot > 0:
        xxz.plot(ax=ax[0,1])
        ax[0,1].set_title(r"$S^z$, $S^2$ eigenbasis")

    print()
    print("Partitioning in Sz, P basis:")
    xxz = AutoPartition(H)
    P = parity(N)
    symmetry_operators = {'Sz':Sz, 'P':P}
    xxz.apply_symmetries(symmetry_operators=symmetry_operators)
    xxz.set_labels()
    xxz.print_labels()
    
    if args.plot > 0:
        xxz.plot(ax=ax[1,0])
        ax[1,0].set_title(r"$S^z$, P eigenbasis")

    print()
    print("Partitioning in Sz, S2, P basis:")
    xxz = AutoPartition(H)
    symmetry_operators = {'Sz':Sz, 'S2':S2, 'P':P}
    xxz.apply_symmetries(symmetry_operators=symmetry_operators)
    xxz.set_labels()
    xxz.print_labels()

    if args.plot > 0:
        xxz.plot(ax=ax[1,1])
        ax[1,1].set_title(r"$S^z$, $S^2$, P eigenbasis")
    
    if args.plot > 0:
        plt.show()