# Copyright (c) 2023 H. L. Nourse
#
# This file is part of matrix-partition.
#
# matrix-partition is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, with version 3 of the License.
#
# matrix-partition is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
# more details.
#
# You should have received a copy of the GNU General Public License along with 
# matrix-partition. If not, see <https://www.gnu.org/licenses/>. 

import numpy as np
import scipy

from copy import deepcopy
import itertools

from .functions import *

class AutoPartition(object):
    def __init__(self, M):
        self._M = M

        self._connections = None
        self._sectors = None
        self._symmetry_operators = None
        self._P = None
        self._Ms = None
        self._labels = None

        self.run()
    
    def run(self):
        self._connections = self.get_connections(M=self._M)
        self._sectors, self._num_sectors = self.get_sectors(connections=self._connections)
        self._P = self.get_permutation(sectors=self._sectors)
        self._Ms = self.get_submatrices(M=self._M, sectors=self._sectors)
        self._M = scipy.linalg.block_diag(*self._Ms)

    def check_commutations(self, symmetry_operators, M=None):
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
        if M is None:
            M = self._M
        return get_connections(M, tol=1e-12, weight_fnc=None)

    def get_sectors(self, connections=None):
        if connections is None:
            connections = self._connections
        sectors = get_symmetry_sectors(connections)
        num_sectors = len(sectors)
        return sectors, num_sectors

    def set_labels(self):
        self._labels = get_symmetry_labels(self._M, self._symmetry_operators, self._sectors)

    def print_labels(self):
        print_labels(self._labels)

    def get_permutation(self, sectors=None):
        if sectors is None:
            sectors = self._sectors
        return get_permutation(sectors)

    def permute(self, M, P=None):
        if P is None:
            P = self._P
        return permute_Op(M, P)

    def get_submatrices(self, M, sectors=None):
        if sectors is None:
            sectors = self._sectors
        Ms = []
        for s in range(len(sectors)):
            Ms.append( self.permute(M, sectors[s]) )
        return Ms

    def rotate_sectors(self, rotation, Ms=None):
        if Ms is None:
            Ms = self._Ms
        out = [np.empty(shape=A.shape) for A in Ms]
        for s in range(len(Ms)):
            out[s] = rotate_operator(Ms[s], rotation[s])
        return out

    def apply_symmetries(self, symmetry_operators):        
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
        if M is None:
            M = self._M
        if style == 'hinton':
            plot_hinton(M, ax=ax, cmap=cmap)
        else:
            raise ValueError("Only hinton plots implemented !")

    @property
    def M(self):
        return self._M

    @property
    def submatrices(self):
        return self._Ms

    def submatrix(self, s):
        return self._Ms[s]

    @property
    def connections(self):
        return self._connections

    @property
    def sectors(self):
        return self._sectors

    @property
    def num_sectors(self):
        return self._num_sectors

    @property
    def labels(self):
        return self._labels

    @property
    def P(self):
        return self._P