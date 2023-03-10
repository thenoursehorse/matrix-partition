'''
Copyright (C) 2023 H. L. Nourse

This file is part of matrix-partition.

matrix-partition is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the Free 
Software Foundation, with version 3 of the License.

matrix-partition is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
more details.

You should have received a copy of the GNU General Public License along with 
matrix-partition. If not, see <https://www.gnu.org/licenses/>.

The functions merge_lists_jochen, merge_lists_howard, and plot_hinton 
are not under the GPL-3.0 license outlined above.
'''

import numpy as np

from copy import deepcopy
import itertools

import matplotlib.pyplot as plt
import matplotlib as mpl

def commutator(Op_A, Op_B):
    return Op_A @ Op_B - Op_B @ Op_A

def rotate_operator(Op, U):
    return U.conj().T @ Op @ U

def apply_mask(l, mask):
    if isinstance(l, np.ndarray):
        return l[mask]
    elif isinstance(l, list):
        return list(itertools.compress(l, mask))
    else:
        raise ValueError("Must apply mask to a list or numpy array !")

def merge_lists_jochen(l):
    '''
    Exactly copy pasted from https://stackoverflow.com/a/4843408
    NOTE: Not under GPL-3.0 license.
    '''
    import networkx 
    from networkx.algorithms.components.connected import connected_components
    
    def to_graph(l):
        G = networkx.Graph()
        for part in l:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        """ 
            treat `l` as a Graph and returns it's edges 
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)

        for current in it:
            yield last, current
            last = current
    
    G = to_graph(l)
    return list(connected_components(G))

def merge_lists_howard(l):
  '''
  Exactly copy pasted from https://stackoverflow.com/a/4842897
  NOTE: Not under GPL-3.0 license.
  '''
  out = []
  while len(l)>0:
      first, *rest = l
      first = set(first)

      lf = -1
      while len(first)>lf:
          lf = len(first)

          rest2 = []
          for r in rest:
              if len(first.intersection(set(r)))>0:
                  first |= set(r)
              else:
                  rest2.append(r)
          rest = rest2

      out.append(first)
      l = rest
  return out

def get_symmetry_sectors(basis_connections):
    '''
    Partition Hilbert space according to how basis states connect from 
    by the application of an operator (usually the Hamiltonian).

    Args:
        basis_connections : A list of lists containing the index that each 
            basis vector connects to each other basis vector.

    Returns:
        An organized list of lists of symmetry sectors. Each element of the 
        outer list is a symmetry sector. Each list within that sector is an 
        index to the basis states that belong to that symmetry sector.
    '''
    # Makes sure each connection includes itself
    for i in range(len(basis_connections)):
        basis_connections[i].append(i)
        basis_connections[i] = np.unique(basis_connections[i]).tolist()
    try:
        symmetry_sectors = merge_lists_jochen(basis_connections)
    except:
        symmetry_sectors = merge_lists_howard(basis_connections)
    return [list(e) for e in symmetry_sectors]

def get_connections(Op, tol=1e-12, weight_fnc=None):
    '''
    Find all connections an operator has between basis states.

    Args:
        Op : A Hermitian operator.

        tol : (Default e-12) The tolerance for non-zero connections.

        weight_fnc : (Default abs()) The weighting function to compare matrix 
            values.

    Returns:
        A list of size the dimension of Op, where each element indexes a basis 
        state and is another list of indices to the basis states Op connects 
        to.
    '''
    if weight_fnc is None:
        weight_fnc = np.abs

    # psi_i = \sum_j H_ij psi_j
    # psi_j = \delta_jk for a basis state
    # psi_i = H_ij
    # Could also do this with basis states and go basis_new = H * basis[i]
    # and check for non-zeros.
    hilbert_size = Op.shape[0]
    basis_connections = []
    for j in range(hilbert_size):
        basis_connections.append(np.asarray(weight_fnc(Op[:,j]) > tol).nonzero()[0].tolist())
    return basis_connections

def filter_disjointed_connections(basis_connections):
    '''
    If an operator returns zero on a basis state, that state is effectly not 
    in the Hilbert space under the action of that operator and should be 
    removed as a disjointed set.

    Args:
        basis_connections : A list of lists that indexes what each basis state
            connects to.

    Returns:
        basis_active : A mask of basis states that have non-zero action.

        basis_inactive : A mask of the basis states that have zero action.
    '''
    basis_active = [False for i in range(len(basis_connections))]
    basis_inactive = [False for i in range(len(basis_connections))]

    for i in range(len(basis_connections)):
        if len(basis_connections[i]) == 0:
            basis_inactive[i] = True
        else:
            basis_active[i] = True
    return basis_active, basis_inactive

def get_permutation(symmetry_sectors):
    '''
    Permutation list for a matrix to convert from original computational 
    basis ordering to an order that is block diagonal with respect to 
    the symmetry sectors.

    Args:
        symmetry_sectors : A list where each element is a symmetry sector, which 
            is a list of indices of the basis states that belong to it in the 
            original computational basis ordering.
    '''
    if isinstance(symmetry_sectors[0], list):
        return sum(symmetry_sectors, [])
    # FIXME assumes above is an int instead (list, not list of lists)
    else:
        return symmetry_sectors

def permute_Op(Op, P):
    '''
    Permutes an operator according to the list P, i.e., shifts rows and 
    columns such that [0, 1, 2, ..., n] -> [P_0, P_1, P_2, ..., P_n].

    Args:
        Op : A Hermitian operator.

        P : A permutation list of indices.

    Returns:
        Operator permuted by P.
    '''
    Op_P = deepcopy(Op)
    Op_P = Op_P[P,:]
    Op_P = Op_P[:,P]
    return Op_P

def commutator_list(Op_dict, Op_B):
    '''
    Checks if a dictionary containing operators commutes with another operator,
    like a Hamiltonian.

    Args:
        Op_dict : Dictionary of operators. Keys are a label and the value is 
            the operator.

        Op_B : The operator to check commutation against.

    Returns:
        A dictionary where the value is a bool for if that operator commutes.
    '''
    comm_truth_list = {key: False for key in Op_dict.keys()}
    for key, Op_A in Op_dict.items():
        comm = commutator(Op_A, Op_B)
        if len(np.asarray(np.abs(comm[:,:]) > 1e-12).nonzero()[0]) == 0:
            comm_truth_list[key] = True
    return comm_truth_list

def get_symmetry_labels(H, Op_dict, symmetry_sectors):
    '''
    Label each symmetry sector according to a dictionary of operators. Checks 
    that the operators commute with the Hamiltonian, and that the basis of H 
    is a simultaneous eigenbasis for the operator.

    Args:
        H : The Hamiltonian (or any Hermitian operator).

        Op_dict : A dictionary of operators.

        symmetry_sectors : A list where each element is a symmetry sector, which 
            is a list of indices of the basis states that belong to it in the 
            original computational basis ordering.

    Returns:
        A list of dictionarys where the index in the list is the symmetry 
        sector and the dictionary are the associated quantum numbers with that 
        symmetry sector. The dictionary also includes the size of each sub 
        Hilbert space (each symmetry sector) with key 'dim'
    '''
    labels = [dict() for _ in symmetry_sectors]
    for s in range(len(symmetry_sectors)):
        labels[s]['dim'] = len(symmetry_sectors[s])
    
    if Op_dict is not None:
        quantum_numbers = [dict() for _ in symmetry_sectors]
        for s in range(len(symmetry_sectors)):
            symm = symmetry_sectors[s]
            for key, Op in Op_dict.items():
                temp = np.empty(len(symm))
                for j in range(len(symm)):
                    temp[j] = np.real(Op[symm[j],symm[j]])
                if np.var(temp) > 1e-12:
                    # HACK it is safer to check that the application of Op does
                    # not connect to another symmetry sector, but whatever
                    print(f"{key} index {j} is not a quantum number in this basis !")
                    quantum_numbers[s][key] = np.inf
                else:
                    quantum_numbers[s][key] = np.round(temp[0], decimals=2)
            labels[s].update(quantum_numbers[s])
    return labels

def print_labels(labels):
    '''
    Print a table to stdout of each symmetry sector and its quantum numbers.

    Args:
        labels : A dictionary of quantum numbers.
    '''
    header = '{:<10}'.format("Sector")
    for key in labels[0].keys():
        header += f'{key:<10}'
    print(header)
    header_border = ''
    for i in range(len(header)):
        header_border += '-'
    print(header_border)
    for s in range(len(labels)):
        line = '{:<10}'.format(s)
        for val in labels[s].values():
            #line += f'{val:<10.2f}'
            line += f'{val:<10}'
        print(line)

def plot_hinton(matrix, max_weight=None, ax=None, cmap=None):
    """
    Draw Hinton diagram for visualizing a weight matrix.
    Copied from https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html
    Not under GPL-3.0 license.
    
    Args:
        matrix : A not necessary square matrix.

        max_weight : The maximum value to assign to a matrix element. Defaults 
            2 * ceil( max( log(|matrix|) ) )

        ax : Axis to draw on. Defaults to current canvas.

        cmap : What color map (a cmap object from matplotlib) to use for the 
            colors. Defaults to 'seismic'. If cmap = 'whiteblack' color will 
            be white if value > 0 or black if value < 0.
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    if not cmap:
        cmap = plt.get_cmap('seismic')
    
    def color(w):
        if cmap == 'whiteblack':
            color = 'white' if w > 0 else 'black'
            return color
        else:
            w = np.abs(w) * np.sign(np.real(w))
            return cmap(int((w + max_weight) * 256 / (2 * max_weight)))

    ax.patch.set_facecolor('darkgray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        if np.abs(w) < 1e-12:
            continue
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color(w), edgecolor=color(w))
        ax.add_patch(rect)

    if cmap != 'whiteblack':
        # Copied from qutip hinton function
        norm = mpl.colors.Normalize(-max_weight, max_weight)
        cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.01)
        mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=cmap)
    
    ax.autoscale_view()
    ax.invert_yaxis()