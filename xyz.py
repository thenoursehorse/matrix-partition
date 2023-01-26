import argparse
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

from autopartition import AutoPartition, rotate_operator
from operators import *
from hamiltonians import xyz_chain, qutip_xyz_chain

if __name__ == '__main__':
    print("XYZ: Delta ZZ coupling. alpha power law coupling. gamma XX,ZZ coupling difference")
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=5)
    parser.add_argument('-Delta', type=float, default=1)
    parser.add_argument('-alpha', type=float, default=10000)
    parser.add_argument('-gamma', type=float, default=0)
    parser.add_argument('-with-qutip', type=int, default=0)
    parser.add_argument('-print', type=int, default=1)
    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(args)

    print()
    print("Symmetries for N sites are:")
    print("xxx periodic even: Sz, total spin (SU(2)), Parity (P), pi rotation about x (R_x), crystal momentum (C_N)")
    print("xxx periodic odd: Sz, SU(2), P, C_N")
    print("xxx open even: Sz, SU(2), P, R_x")
    print("xxx open odd: Sz, SU(2), P")
    print("xxy periodic even: Sz, P, R_x, C_N")
    print("xxy periodic odd: Sz, P, C_N")
    print("xxy open even: Sz, P, R_x")
    print("xxy open odd: Sz, P")

    # Define Hamiltonian
    N = args.N
    if args.with_qutip > 0:
        H = qutip_xyz_chain(N=args.N, Delta=args.Delta, alpha=args.alpha, gamma=args.gamma).data.todense()
    else:
        H = xyz_chain(N=args.N, Delta=args.Delta, alpha=args.alpha, gamma=args.gamma)

    #### COMPUTATIONAL BASIS ####
    #############################

    print()
    print("Partitioning in computational (Sz) basis:")

    # Symmetry labels
    Sz = spin_sum(N=N, axis='z')
    symmetry_operators = {'Sz':Sz}

    xyz = AutoPartition(H, symmetry_operators=symmetry_operators)

    ####    Sz,S^2  BASIS    ####
    #############################

    print()
    print("Partitioning in Sz, S2 basis:")

    S2 = spin_S2(N=N)

    # For each submatrix find the basis where S2 is diagonal
    S2_S2 = xyz.get_submatrices(S2)
    S2_vecs = []
    S2_eigs = []
    for s in range(xyz.num_sectors):
        e, v = np.linalg.eigh(S2_S2[s])
        S2_eigs.append( e )
        S2_vecs.append( v )
        S2_S2[s] = np.diag(S2_eigs[s])
    S2_S2 = scipy.linalg.block_diag(*S2_S2)

    # Rotate Sz into this basis
    Sz_S2 = xyz.get_submatrices(Sz)
    for s in range(xyz.num_sectors):
        Sz_S2[s] = rotate_operator(Op=Sz_S2[s], U=S2_vecs[s])
    Sz_S2 = scipy.linalg.block_diag(*Sz_S2)

    # Rotate permuted H into S2 eigenbasis
    Hs_S2 = xyz.rotate_sectors(S2_vecs)
    H_S2 = scipy.linalg.block_diag(*Hs_S2)

    # Symmetry labels
    symmetry_operators_S2 = {'Sz':Sz_S2, 'S2':S2_S2}

    # Repartition in S2 eigenbasis
    xyz_S2 = AutoPartition(H_S2, symmetry_operators=symmetry_operators_S2)

    ####     Sz,P  BASIS     ####
    #############################

    print()
    print("Partitioning in Sz, P basis:")

    P = parity(N)

    P_P = xyz.get_submatrices(P)
    P_vecs = []
    P_eigs = []
    for s in range(xyz.num_sectors):
        e, v = np.linalg.eigh(P_P[s])
        P_eigs.append( e )
        P_vecs.append( v )
        P_P[s] = np.diag(P_eigs[s])
    P_P = scipy.linalg.block_diag(*P_P)

    # Rotate Sz into this basis
    Sz_P = xyz.get_submatrices(Sz)
    for s in range(xyz.num_sectors):
        Sz_P[s] = rotate_operator(Op=Sz_P[s], U=P_vecs[s])
    Sz_P = scipy.linalg.block_diag(*Sz_P)

    # Rotate permuted H into S2 eigenbasis
    Hs_P = xyz.rotate_sectors(P_vecs)
    H_P = scipy.linalg.block_diag(*Hs_P)

    # Symmetry labels
    symmetry_operators_P = {'Sz':Sz_P, 'P':P_P}

    # Repartition in S2 eigenbasis
    xyz_P = AutoPartition(H_P, symmetry_operators=symmetry_operators_P)

    ####   Sz,S^2,P  BASIS   ####
    #############################

    print()
    print("Partitioning in Sz, S2, P basis:")

    # Start from P eigenbasis and then rotate into S2

    # Rotate P into S2 basis
    S2_P = xyz.get_submatrices(S2)
    for s in range(xyz.num_sectors):
        S2_P[s] = rotate_operator(Op=S2_P[s], U=P_vecs[s])
    S2_P = scipy.linalg.block_diag(*S2_P)

    # Diagonalize S2 in each P block
    S2_final = xyz_P.get_submatrices(S2_P)
    S2_final_vecs = []
    S2_final_eigs = []
    for s in range(xyz_P.num_sectors):
        e, v = np.linalg.eigh(S2_final[s])
        S2_final_eigs.append( e )
        S2_final_vecs.append( v )
        S2_final[s] = np.diag(S2_final_eigs[s])
    S2_final = scipy.linalg.block_diag(*S2_final)

    # Rotate Sz and P into this basis
    Sz_final = xyz_P.get_submatrices(Sz_P)
    P_final = xyz_P.get_submatrices(P_P)
    for s in range(xyz_P.num_sectors):
        Sz_final[s] = rotate_operator(Op=Sz_final[s], U=S2_final_vecs[s])
        P_final[s] = rotate_operator(Op=P_final[s], U=S2_final_vecs[s])
    Sz_final = scipy.linalg.block_diag(*Sz_final)
    P_final = scipy.linalg.block_diag(*P_final)

    # Rotate permuted H into final eigenbasis
    Hs_final = xyz_P.rotate_sectors(S2_final_vecs)
    H_final = scipy.linalg.block_diag(*Hs_final)

    # Symmetry labels
    symmetry_operators_final = {'Sz':Sz_final, 'S2':S2_final, 'P':P_final}

    # Repartition in S2 eigenbasis
    xyz_final = AutoPartition(H_final, symmetry_operators=symmetry_operators_final)

    if args.print > 0:
        # Weight plot of matrix elements in block-diagonal permutations
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        xyz.plot(ax=ax[0,0])
        ax[0,0].set_title(r'$S^z$ eigenbasis')

        xyz_S2.plot(ax=ax[0,1])
        ax[0,1].set_title(r"$S^z$, $S^2$ eigenbasis")

        xyz_P.plot(ax=ax[1,0])
        ax[1,0].set_title(r"$S^z$, P eigenbasis")

        xyz_final.plot(ax=ax[1,1])
        ax[1,1].set_title(r"$S^z$, $S^2$, P eigenbasis")

        plt.show()