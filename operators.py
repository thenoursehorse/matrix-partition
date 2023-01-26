import numpy as np
from copy import deepcopy

def sigma(axis):
    if axis == 'x':
        Op = np.array([[0,1],[1,0]], dtype=np.complex_)
    elif axis == 'y':
        Op = np.array([[0,-1j],[1j,0]], dtype=np.complex_)
    elif axis == 'z':
        Op = np.array([[1,0],[0,-1]], dtype=np.complex_)
    elif axis == '+':
        Op = np.array([[0,1],[0,0]], dtype=np.complex_)
    #elif axis == '-':
    else:
        raise ValueError('must be x,y,z,+,- for spin operators')
    return Op

def sigma_chain(N, axis1, idx1, axis2=None, idx2=None):
    '''
    Creates a one or two-body Pauli operator on a chain.

    Args:
        N : Size of chain.

        axis1 : x,y,z Pauli matrix for operator 1.

        idx1 : Index position on chain for operator 1.

        axis2 : (Default None) As axis1 for operator 2. If None
            then returns a one-body operator on the chain.

        idx2 : (Default None) As idx2
    '''
    ident = np.eye(2, dtype=np.complex_)
   
    op_list = []
    for m in range(N):
        op_list.append(ident)
    op_list[idx1] = sigma(axis1)

    if axis2 is not None:
        op_list[idx2] = sigma(axis2)
        
    op = deepcopy(op_list[0])
    for m in range(1,N):
        op = np.kron(op, op_list[m])
    return op

def sigma_sum(N, axis):
    ident = np.eye(2, dtype=np.complex_)

    Op = 0
    for m in range(N):
        Op += sigma_chain(N, axis, m)
    return Op

def spin(axis):
    return 0.5 * sigma(axis)

def spin_chain(N, axis1, idx1, axis2=None, idx2=None):
    if axis2 is None:
        return 0.5 * sigma_chain(N, axis1, idx1)
    else:
        return 0.25 * sigma_chain(N, axis1, idx1, axis2, idx2)
    
def spin_sum(N, axis):
    return 0.5 * sigma_sum(N, axis)

def spin_vec(N):
    axes = ['x','y','z']
    S_vec = [0,0,0]
    for i in range(N):
        for mu, axis in enumerate(axes):
            S_vec[mu] += spin_chain(N, axis, i)
    return S_vec

def spin_S2(N):
    S_vec = spin_vec(N)
    S2 = 0
    for i in range(3):
        S2 += S_vec[i] @ S_vec[i].conj().T
    return S2

def S2_to_S(S2):
    return 0.5*(-1 + np.sqrt(1 + 4*S2))

def permute_ij(N, i, j):
    # indexed from zero, but my equations are 1->N
    i -= 1
    j -= 1
    axes = ['x','y','z']
    P = 0
    for mu, axis in enumerate(axes):
        P += sigma_chain(N, axis, i, axis, j)
    P += np.eye(P.shape[0])
    return P / 2.0

def parity(N):
    Op = permute_ij(N, 1, N)
    
    # Even
    if N%2 == 0:
        for i in range(1, int(N/2) ):
            Op = Op @ permute_ij(N, i, N-i)
    # Odd
    else:
        for i in range(1, int( (N-1)/2) ):
            Op = Op @ permute_ij(N, i, N-i)
    
    return Op