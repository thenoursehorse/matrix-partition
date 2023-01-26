import numpy as np
from operators import *

def get_sigma_ops(N, axis):
    import qutip as qt
    si = qt.qeye(2)
   
    if axis == 'x':
        s = qt.sigmax()
    elif axis == 'y':
        s = qt.sigmay()
    elif axis == 'z':
        s = qt.sigmaz()
    elif axis == '+':
        s = qt.sigmap()
    elif axis == '-':
        s = qt.sigmam()
    else:
        raise ValueError('must be x,y,z,+,- for spin operators')

    s_list = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = s
        s_list.append(qt.tensor(op_list))
    return s_list

def qutip_ising_chain(N, g=0, alpha=np.inf):

    sx_list = get_sigma_ops(N=N, axis='x')
    sz_list = get_sigma_ops(N=N, axis='z')

    # Set the energy scale
    J = 1

    # For equivalence to time crystal, Akitada would replace
    # g by g = pi/2 * J/0.03 eps

    # construct the hamiltonian
    H = 0

    # magnetic field
    for n in range(N):
        H -= g * sx_list[n]

    # interaction terms
    if alpha == np.inf:
        for i in range(N):
            H += J * sz_list[i] * sz_list[(i+1) % N]

    else:
        for i in range(N):
            for j in range(N):
                if i > j:
                    coupling = J / np.power( np.abs(i-j), alpha)
                    H += coupling * sz_list[i] * sz_list[j]
    return H

def qutip_xyz_chain(N, Delta=1, alpha=np.inf, gamma=0, g=0):
    
    sx_list = get_sigma_ops(N=N, axis='x')
    sy_list = get_sigma_ops(N=N, axis='y')
    sz_list = get_sigma_ops(N=N, axis='z')

    J = 1

    H = 0
    
    for n in range(N):
        H -= g * sz_list[n]

    if alpha == np.inf:
        for n in range(N):
            H += J * (1.0 + gamma) * sx_list[n] * sx_list[(n+1) % N]
            H += J * (1.0 - gamma) * sy_list[n] * sy_list[(n+1) % N]
            H += J * Delta * sz_list[n] * sz_list[(n+1) % N]
    else:
        for i in range(N):
            for j in range(N):
                if i > j:
                    coupling = J / np.power( np.abs(i-j), alpha)
                    H += coupling * (1.0 + gamma) * sx_list[i] * sx_list[j]
                    H += coupling * (1.0 - gamma) * sy_list[i] * sy_list[j]
                    H += coupling * Delta * sz_list[i] * sz_list[j]

    return H

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

def xyz_chain(N, Delta=1, alpha=np.inf, gamma=0, g=0):
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