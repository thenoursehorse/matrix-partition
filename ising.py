import numpy as np
import scipy
import scipy.linalg
from autopartition import AutoPartition
from operators import *

import matplotlib.pyplot as plt

def ising_chain(N=5, g=0, alpha=np.inf):
    # Set the energy scale
    J = 1

    # construct the hamiltonian
    H = 0

    # magnetic field
    for i in range(N):
        H -= g * sigma_chain(N, 'x', i)

    # interaction terms
    if alpha == np.inf:
        for i in range(N):
            H += J * sigma_chain(N, 'z', i, 'z', (i+1)%N)

    else:
        for i in range(N):
            for j in range(N):
                if i > j:
                    coupling = J / np.power( np.abs(i-j), alpha)
                    H += coupling * sigma_chain(N, 'z', i, 'z', j)
    return H

# Chain length
N = 4

# Define Hamiltonian
g = 1
alpha = 1.51 #np.inf #1.51
H = ising_chain(N=N, g=g, alpha=alpha)

# Obtain unitary at time t
t = np.pi/4.
U = scipy.linalg.expm(-1j*H*t)

# Symmetry labels
sigmaz = sigma_sum(N=N, axis='z')
Op_dict = {'sigmaz': sigmaz}

print("Partitioning in computational basis:")
ising = AutoPartition(U, symmetry_operators=Op_dict)

# Weight plot of matrix elements in block-diagonal permutation
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ising.plot(ax=ax)
ax.set_xlabel('States')
ax.set_ylabel('States')
plt.show()