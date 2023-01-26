import numpy as np
import scipy
import scipy.linalg
from autopartition import AutoPartition, rotate_operator
from operators import *

import matplotlib.pyplot as plt

def xyz_chain(N, Delta, alpha=np.inf, gamma=0, g=0):
    J = 1
    H = 0
    for i in range(N):
        H -= g * spin_chain(N, 'z', i)

    if alpha == np.inf:
        for i in range(N):
            k = (i+1)%N # Assume periodic boundaries
            H += J * (1.0 + gamma) * spin_chain(N, 'x', i, 'x', k)
            H += J * (1.0 - gamma) * spin_chain(N, 'y', i, 'y', k)
            H += J * Delta * spin_chain(N, 'z', i, 'z', k)
    else:
        for i in range(N):
            for k in range(N):
                if i > k:
                    coupling = J / np.power( np.abs(i-k), alpha)
                    H += coupling * (1.0 + gamma) * spin_chain(N, 'x', i, 'x', k)
                    H += coupling * (1.0 - gamma) * spin_chain(N, 'y', i, 'y', k)
                    H += coupling * Delta * spin_chain(N, 'z', i, 'z', k)

    return H

# Chain length
N = 5

# Define Hamiltonian
alpha = np.inf # 1.51 # 0
Delta = 1
H = xyz_chain(N=N, Delta=Delta, alpha=alpha)

# Symmetry labels
S2 = spin_S2(N=N)
Sz = spin_sum(N=N, axis='z')
symmetry_operators = {'S2':S2, 'Sz':Sz}

print("Partitioning in computational (Sz) basis:")
xyz = AutoPartition(H, symmetry_operators=symmetry_operators)

print()
print("Partitioning in S^2, Sz basis:")

# For each submatrix find the basis where S2 is diagonal
S2_rotated = xyz.get_submatrices(S2)
S2_vecs = []
S2_eigs = []
for s in range(xyz.num_sectors):
    e, v = np.linalg.eigh(S2_rotated[s])
    S2_eigs.append( e )
    S2_vecs.append( v )
    S2_rotated[s] = np.diag(S2_eigs[s])
S2_rotated = scipy.linalg.block_diag(*S2_rotated)

# Rotate Sz into this basis
Sz_rotated = xyz.get_submatrices(Sz)
for s in range(xyz.num_sectors):
    Sz_rotated[s] = rotate_operator(Op=Sz_rotated[s], U=S2_vecs[s])
Sz_rotated = scipy.linalg.block_diag(*Sz_rotated)

# Rotate permuted H into S2 eigenbasis
Hs_rotated = xyz.rotate_sectors(S2_vecs)
H_rotated = scipy.linalg.block_diag(*Hs_rotated)

# Symmetry labels
symmetry_operators_rotated = {'S2':S2_rotated, 'Sz':Sz_rotated}

# Repartition in S2 eigenbasis
xyz_rotated = AutoPartition(H_rotated, symmetry_operators=symmetry_operators_rotated)

# Weight plot of matrix elements in block-diagonal permutations
fig, ax = plt.subplots(2, 1, figsize=(10, 20))
xyz.plot(ax=ax[0])
ax[0].set_title(r'$S^z$ eigenbasis')

xyz_rotated.plot(ax=ax[1])
ax[1].set_title(r"$S^2$, $S^z$ eigenbasis")

plt.show()