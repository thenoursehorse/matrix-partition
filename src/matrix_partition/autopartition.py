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
'''

import numpy as np
import scipy

from copy import deepcopy
import itertools

from matrix_partition.functions import *

class AutoPartition(object):
    '''
    Automatically permute a matrix into block diagonal form based on 
    off-diagonal components. Tranform matrix into additional blocks 
    with respect to a symmetry operator.
    '''
    def __init__(self, M):
        '''
        Initialize and execute run() to automatically partition a matrix.

        Args:
            M : Matrix to autopartition.
        '''
        self._M = M

        self._connections = None
        self._sectors = None
        self._symmetry_operators = None
        self._P = None
        self._Ms = None
        self._labels = None

        self.run()
    
    def run(self):
        '''
        Sort rows and columns of matrix M into block diagonal form. Sets the 
        sectors and the permutation array from the original matrix M to the
        permuted one.
        '''
        self._connections = self.get_connections(M=self._M)
        self._sectors, self._num_sectors = self.get_sectors(connections=self._connections)
        self._P = self.get_permutation(sectors=self._sectors)
        self._Ms = self.get_submatrices(M=self._M, sectors=self._sectors)
        self._M = scipy.linalg.block_diag(*self._Ms)

    def check_commutations(self, symmetry_operators, M=None):
        '''
        Determines if a list of matrices commutes [A,M] = AM - MA = 0.

        Args:
            symmetry_operators : A dictionary containing each operator that 
                should commute with M.

            M : (Default self.M) The matrix to check commutations against.

        Returns :
            Prints commutation for each operator, and returns a boolean 
            dictionary where each element indicates which operators commute.
        '''
        if M is None:
            M = self._M

        # Reorder symmetry operators into order of M
        sym_operators_rot = deepcopy(symmetry_operators)
        for key,Op in sym_operators_rot.items():
            sym_operators_rot[key] = self.permute(Op)
        
        # Check commutations
        truth_list = commutator_list(sym_operators_rot, M)
        for key,val in truth_list.items():
            print(f'[H,{key}] = 0 is {val}')
        return truth_list

    def get_connections(self, M=None):
        '''
        Determines the non-zero off-diagonal elements for each basis state 
        for the matrix M. Assumes a tolerance of 1e-12 for a zero.
        
        Args:
            M : (Default self.M) The matrix.

        Returns:
            A list of lists, where each list are the indices that the 
            basis element has non-zero connection to.
        '''
        if M is None:
            M = self._M
        return get_connections(M, tol=1e-12, weight_fnc=None)

    def get_sectors(self, connections=None):
        '''
        Determines each block-diagonal sector.

        Args:
            connections : (Default self.connections) A list of lists for each 
                basis state indicating the elements it has non-zero 
                connection to for some matrix M. See self.get_connections.

        Returns:
            sectors : A list of lists. Each element of the list is a block of 
                the matrix M. Each block is a list indicating which basis 
                vectors are in that block.

            num_sectors : The number of blocks.
        '''
        if connections is None:
            connections = self._connections
        sectors = get_symmetry_sectors(connections)
        num_sectors = len(sectors)
        return sectors, num_sectors

    def set_labels(self):
        '''
        Sets the symmetry labels (quantum numbers) of each block of matrix 
        self.M. Also sets the dimension of each block. If 
        self.symmetry_operators has not been set by self.apply_symmetries 
        then it only sets the dimension.
        '''
        self._labels = get_symmetry_labels(self._M, self._symmetry_operators, self._sectors)

    def print_labels(self):
        '''
        Print the labels in a table.
        '''
        print_labels(self._labels)

    def get_permutation(self, sectors=None):
        '''
        Retrun an array that indicates the permutation of basis states.

        Args:
            sectors : A list (or list of lists) for the permutation in each 
                symmetry sector of the matrix M.
        '''
        if sectors is None:
            sectors = self._sectors
        return get_permutation(sectors)

    def permute(self, M, P=None):
        '''
        Permute the rows and columns of a matrix.

        Args:
            M : The matrix.

            P : A permutation list indicating the new ordering.

        Returns:
            The matrix M permuted.
        '''
        if P is None:
            P = self._P
        return permute_Op(M, P)

    def get_submatrices(self, M, sectors=None):
        '''
        Return a matrix permuted by the autopartioning.

        Args:
            M : The matrix to permute.

            sectors : (Default self.sectors) A list of lists specifying the 
                permutation order of each symmetry sector.

        Returns:
            A list of matrices, where each element of the list corresponds to 
            a symmetry sector of the autopartioning.
        '''
        if sectors is None:
            sectors = self._sectors
        Ms = []
        for s in range(len(sectors)):
            Ms.append( self.permute(M, sectors[s]) )
        return Ms

    def rotate_sectors(self, rotation, Ms=None):
        '''
        Returns a list of matrices that have changed basis of each symmetry 
        sector.

        Args:
            rotation : A list of unitaries for each symmetry sector that 
                transforms into the new basis.

            Ms : (Default self.Ms) A list of matrices to transform for each
                symmetry sector.
        '''
        if Ms is None:
            Ms = self._Ms
        out = [np.empty(shape=A.shape) for A in Ms]
        for s in range(len(Ms)):
            out[s] = rotate_operator(Ms[s], rotation[s])
        return out

    def apply_symmetries(self, symmetry_operators):
        '''
        Block diagonalize self.M according to symmetry operators.

        Args:
            symmetry_operators : A dictionary of symmetry operators.

        Returns:
            If any operators in symmetry_operators do not commute with self.M, 
            returns a dictionary of booleans specifying whether the operator
            commutes.
        '''
        # Check symmetry operators commute
        truth_list = self.check_commutations(symmetry_operators=symmetry_operators)
        if not all(truth_list.values()):
            return truth_list
        else:
            self._symmetry_operators = symmetry_operators
            
        # To hold transformation information
        self._sym_Op_eigs = {key:[] for key in self._symmetry_operators.keys()}
        self._sym_Op_vecs = {key:[] for key in self._symmetry_operators.keys()}
        self._sym_Op_order = {key:[] for key in self._symmetry_operators.keys()}

        for i, (key,Op) in enumerate( self._symmetry_operators.items() ):
            self._sym_Op_order[key] = i
            
            # For each submatrix find the eigenbasis of Op
            Op_s = self.get_submatrices(Op)
            for s in range(self._num_sectors):
                e, v = np.linalg.eigh(Op_s[s])
                self._sym_Op_eigs[key].append( e )
                self._sym_Op_vecs[key].append( v )
            self._symmetry_operators[key] = np.diag( list(itertools.chain.from_iterable(self._sym_Op_eigs[key])) )

            # Rotate all other operators into this basis
            for key2, Op2 in self._symmetry_operators.items():
                if key2 != key:
                    Op2_s = self.get_submatrices(Op2)
                    for s in range(self._num_sectors):
                        Op2_s[s] = rotate_operator(Op=Op2_s[s], U=self._sym_Op_vecs[key][s])
                    # FIXME it isn't worth reconstructing this whole operator and we should instead
                    # keep a list of blocks
                    self._symmetry_operators[key2] = scipy.linalg.block_diag(*Op2_s)

            # Rotate H into this basis
            self._Ms = self.rotate_sectors(self._sym_Op_vecs[key])
    
            # FIXME better to do this for each block seperately
            self._M = scipy.linalg.block_diag(*self._Ms)
            self.run()             

    def plot(self, M=None, ax=None, cmap=None, style='hinton'):
        '''
        Plot a visual representation of a matrix.

        Args:
            M : (Default self.M) The matrix to plot.
            
            ax : (Default creates new figure) The axis to plot to.

            cmap : (Default 'seismic') A matplotlib colormap for the colorbar. 
                Can also specify 'whiteblack' for no colormap.

            style : (Default 'hinton') The type of plot to make. Only 'hinton' 
                is implemented so far.
        '''
        if M is None:
            M = self._M
        if style == 'hinton':
            plot_hinton(M, ax=ax, cmap=cmap)
        else:
            raise ValueError("Only hinton plots implemented !")

    @property
    def M(self):
        ''' 
        The autompartioned matrix.
        '''
        return self._M

    @property
    def submatrices(self):
        '''
        A list of submatrices corresponding to each symmetry block.
        '''
        return self._Ms

    @property
    def connections(self):
        '''
        A list for each basis state specifying non-zero off-diagonal elements
        of the original matrix before autopartioning.
        '''
        return self._connections

    @property
    def sectors(self):
        '''
        A list of each symmetry sector specifying which basis state belongs to 
        it before autopartioning.
        '''
        return self._sectors

    @property
    def num_sectors(self):
        '''
        The number of symmetry sectors.
        '''
        return self._num_sectors

    @property
    def labels(self):
        '''
        A dictionary of symmetry labels (quantum numbers) for each symmetry 
        sector. If no symmetries have been specifying the dictionary only 
        contains the dimension of each sector.
        '''
        return self._labels

    @property
    def P(self):
        '''
        The permutation order from the original matrix to the autopartioned 
        matrix.
        '''
        return self._P