
'''
REF : https://github.com/wutobias/r2z
'''

import itertools
from rdkit import Chem
import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

get_element = Chem.GetPeriodicTable().GetElementSymbol

def pts_to_bond(A, B):
    AB   = (B-A).to(ureg.nanometer).magnitude
    dist = np.linalg.norm(AB)
    return dist*ureg.nanometer

def pts_to_angle(A, B, C):
    BA  = (A-B).to(ureg.nanometer).magnitude
    BC  = (C-B).to(ureg.nanometer).magnitude
    BA /= np.linalg.norm(BA)
    BC /= np.linalg.norm(BC)
    ang = np.arccos(np.dot(BA,BC))
    return ang*ureg.radian

def pts_to_dihedral(A, B, C, D):
    BA  = (A-B).to(ureg.nanometer).magnitude
    BC  = (C-B).to(ureg.nanometer).magnitude
    CD  = (C-D).to(ureg.nanometer).magnitude
    CB  = (C-B).to(ureg.nanometer).magnitude
    BA /= np.linalg.norm(BA)
    BC /= np.linalg.norm(BC)
    CD /= np.linalg.norm(CD)
    CB /= np.linalg.norm(CB)
    n1  = np.cross(BC,BA)
    n2  = np.cross(CD,CB)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    dih = np.arccos(np.dot(n1,n2))
    sign = np.dot(np.cross(n1,n2), BC)
    if sign < 0.:
        dih = -dih
    return dih*ureg.radian

class ZMatrix(object):

    ### Note, internally units are nanometer for length and cart coordinates
    ### and radians for angles. However, the output of the zmatrix is degree
    ### instead of radian, since most QC programs use it.

    def __init__(self, rdmol, root_atm_idx=0):

        if not root_atm_idx < rdmol.GetNumAtoms():
            raise ValueError("root_atm_idx must be 0<root_atm_idx<N_atms")

        self.rdmol             = rdmol
        self.ordered_atom_list = [None]*rdmol.GetNumAtoms()
        self.z                 = dict()
        self.N_atms            = 0
        self.rank              = list(Chem.CanonicalRankAtoms(rdmol, breakTies=False))
        self.n_non_deadends    = 0

        self.add_atom(root_atm_idx)
        self.order_atoms(root_atm_idx)
        self.zzit()

    def z2a(self, z_idx):
        return self.ordered_atom_list[z_idx]

    def a2z(self, atm_idx):
        return self.ordered_atom_list.index(atm_idx)

    def zzit(self):

        self.zz = dict()
        for z_idx, atm_idxs in self.z.items():
            self.zz[z_idx] = [self.a2z(atm_idx) for atm_idx in atm_idxs]
        return True

    def get_neighbor_idxs(self, atm_idx):

        atm            = self.rdmol.GetAtomWithIdx(atm_idx)
        idx_rank_list  = list()
        for atm_nghbr in atm.GetNeighbors():
            idx_rank_list.append([self.rank[atm_nghbr.GetIdx()],
                                   atm_nghbr.GetIdx()])
        for idx_rank in sorted(idx_rank_list, key=lambda idx: idx[0]):
            yield idx_rank[1]

    def get_path_length(self, atm_idx1, atm_idx2, maxlength=100):

        if maxlength < 0:
            raise ValueError("maxlength must >0")

        path_length = -1
        if atm_idx1 == atm_idx2:
            path_length = 0
        elif self.is_neighbor_of(atm_idx1, atm_idx2):
            path_length = 1
        else:
            for k in range(2, maxlength):
                neighbor_list = self.get_k_nearest_neighbors(atm_idx1, k)
                if atm_idx2 in neighbor_list:
                    path_length = k
                    break

        return path_length

    def get_shortest_paths(self, atm_idx1, atm_idx2, query_pool=list(), maxattempts=100):

        if maxattempts < 0:
            raise ValueError("maxattempts must >0")

        if atm_idx1 == atm_idx2:
            shortest_paths = [[atm_idx1]]
        elif self.is_neighbor_of(atm_idx1, atm_idx2):
            shortest_paths = [[atm_idx1, atm_idx2]]
        else:
            shortest_paths = list()
            path_length    = self.get_path_length(atm_idx1, atm_idx2)
            nearest1       = self.get_k_nearest_neighbors(atm_idx1, path_length)
            nearest2       = self.get_k_nearest_neighbors(atm_idx2, path_length)
            intersect      = list(set(nearest1).intersection(set(nearest2)))
            if len(query_pool) > 0:
                intersect = list(set(intersect).intersection(set(query_pool)))
            for intersect_1 in itertools.permutations(intersect, path_length-1):
                if maxattempts < 1:
                    break
                kpaths  = [atm_idx1]
                kpaths += [None]*(path_length-1)
                kpaths += [atm_idx2]
                for l in range(1,path_length):
                    for atm_intersect in intersect_1:
                        l1 = self.get_path_length(atm_intersect, atm_idx1)
                        l2 = self.get_path_length(atm_intersect, atm_idx2)
                        if not (l1 == l and l2 == (path_length-l)):
                            continue
                        if self.is_neighbor_of(atm_intersect, kpaths[l-1]):
                            kpaths[l] = atm_intersect
                maxattempts -= 1
                if kpaths in shortest_paths:
                    continue
                if None in kpaths:
                    continue
                shortest_paths += [kpaths]

        return shortest_paths


    def get_k_nearest_neighbors(self, atm_idx, k=3):

        if k < 0:
            raise ValueError("k must be >0")
        if k == 0:
            neighbor_list = [atm_idx]
        else:
            neighbor_list = list(self.get_neighbor_idxs(atm_idx))
        while (k>1):
            klist = list()
            for atm_nghbr_idx1 in neighbor_list:
                for atm_nghbr_idx2 in list(self.get_neighbor_idxs(atm_nghbr_idx1)):
                    if atm_idx == atm_nghbr_idx2:
                        continue
                    if atm_nghbr_idx1 == atm_nghbr_idx2:
                        continue
                    if atm_nghbr_idx2 in neighbor_list:
                        continue
                    if atm_nghbr_idx2 in klist:
                        continue
                    klist.append(atm_nghbr_idx2)
            neighbor_list += klist
            k -= 1
        return neighbor_list

    def add_atom(self, atm_idx):

        ### Check if we can add atom
        if atm_idx in self.ordered_atom_list:
            return False
        else:
            self.ordered_atom_list[self.N_atms] = atm_idx
            self.N_atms += 1
            if not self.is_dead_end(atm_idx):
                self.n_non_deadends += 1
        ### Build the z matrix
        ### The first three atoms are added 'manually'
        if self.N_atms == 1:
            self.z[0] = [self.ordered_atom_list[0]]
            return True
        elif self.N_atms == 2:
            self.z[1] = [self.ordered_atom_list[1],
                         self.ordered_atom_list[0]]
            return True
        elif self.N_atms == 3:
            if self.get_path_length(self.ordered_atom_list[2],
                                    self.ordered_atom_list[0]) == 2:
                self.z[2] = [self.ordered_atom_list[2],
                             self.ordered_atom_list[1],
                             self.ordered_atom_list[0]]
            else:
                self.z[2] = [self.ordered_atom_list[2],
                             self.ordered_atom_list[0],
                             self.ordered_atom_list[1]]
            return True
        else:
            ### First try to find a chemically identical atom
            ### which alrady has been defined
            for query_atm_idx in self.ordered_atom_list[:self.N_atms-1]:
                if query_atm_idx == None:
                    continue
                if self.rank[query_atm_idx] == self.rank[atm_idx]:
                    idx = self.ordered_atom_list.index(query_atm_idx)
                    if self.is_neighbor_of(query_atm_idx, atm_idx) and idx > 2:
                        self.z[self.N_atms-1] = [atm_idx,
                                                 self.z[idx][1],
                                                 self.z[idx][2],
                                                 self.z[idx][3]]
                        return True
            ### If not, try to find a chemically reasonable path
            for query_atm_idx in self.ordered_atom_list[:self.N_atms-1]:
                if query_atm_idx == None:
                    continue
                if self.get_path_length(atm_idx, query_atm_idx) == 3:
                    zlist               = self.get_shortest_paths(atm_idx,
                                                                  query_atm_idx,
                                                                  self.ordered_atom_list[:self.N_atms])
                    if len(zlist) > 0:
                        self.z[self.N_atms-1] = [atm_idx,
                                                 zlist[0][1],
                                                 zlist[0][2],
                                                 zlist[0][3]]
                        return True
            ### If this didn't work, find another path of length 2
            ### Happens in very small molecules like CH4, where the
            ### the longest path is 2
            for query_atm_idx in self.ordered_atom_list[:self.N_atms-1]:
                if query_atm_idx == None:
                    continue
                if self.get_path_length(atm_idx, query_atm_idx) == 2:
                    zlist               = self.get_shortest_paths(atm_idx,
                                                                  query_atm_idx,
                                                                  self.ordered_atom_list[:self.N_atms])
                    if len(zlist) > 0:
                        for atm_idx_tmp in self.ordered_atom_list[:self.N_atms]:
                            if not atm_idx_tmp in zlist[0]:
                                self.z[self.N_atms-1] = [atm_idx,
                                                         zlist[0][1],
                                                         zlist[0][2],
                                                         atm_idx_tmp]
                                return True
            ### If after all, we still don't have a z matrix entry
            ### for the current atom, just build something that is valid.
            for query_atm_idx in self.ordered_atom_list[:self.N_atms-1]:
                if query_atm_idx == None:
                    continue
                if self.get_path_length(atm_idx, query_atm_idx) == 1:
                    self.z[self.N_atms-1] = list()
                    self.z[self.N_atms-1].append(atm_idx)
                    self.z[self.N_atms-1].append(query_atm_idx)
                    for atm_idx_tmp in self.ordered_atom_list[:self.N_atms]:
                        if not atm_idx_tmp in self.z[self.N_atms-1]:
                            self.z[self.N_atms-1].append(atm_idx_tmp)
                        if len(self.z[self.N_atms-1]) == 4:
                            return True
        return False

    def order_atoms(self, atm_idx):

        add_later = list()
        for atm_nghbr_idx in self.get_neighbor_idxs(atm_idx):
            ### We don't want dead ends in the first
            ### 4 atoms. So add them later!
            if self.is_dead_end(atm_nghbr_idx):
                if self.N_atms < 4:
                    add_later.append(atm_nghbr_idx)
                else:
                    self.add_atom(atm_nghbr_idx)
            elif self.add_atom(atm_nghbr_idx):
                self.order_atoms(atm_nghbr_idx)

        for atm_nghbr_idx in add_later:
            self.add_atom(atm_nghbr_idx)

    def is_dead_end(self, atm_idx):

        if len(list(self.get_neighbor_idxs(atm_idx))) < 2:
            return True
        else:
            return False

    def is_neighbor_of(self, atm_idx1, atm_idx2):

        for atm_nghbr_idx in self.get_neighbor_idxs(atm_idx1):
            if atm_nghbr_idx == atm_idx2:
                return True
        return False

    def build_cart_crds(self, z_crds, virtual_bond=None, virtual_angles=None,
                                      virtual_dihedrals=None, attach_crds=None,
                                      z_order=False):

        ### We use the Natural Extension Reference Frame algorithm.
        ### See DOI 10.1002/jcc.20237 and 10.1002/jcc.25772
        if isinstance(virtual_bond, type(None)):
            virtual_bond = 1.*ureg.nanometer
        if isinstance(virtual_angles, type(None)):
            virtual_angles = np.array([np.pi/2., np.pi/2.])*ureg.radian
        if isinstance(virtual_dihedrals, type(None)):
            virtual_dihedrals = np.array([np.pi/2., np.pi/2., np.pi/3.])*ureg.radian
        if isinstance(attach_crds, type(None)):
            attach_crds = np.eye(3, dtype=float)*ureg.nanometer
            attach_crds[:,0] = np.array([1,0,0])*ureg.nanometer
            attach_crds[:,1] = np.array([0,1,0])*ureg.nanometer
            attach_crds[:,2] = np.array([1,1,0])*ureg.nanometer
        atm_idx_check            = np.zeros(len(self.z), dtype=bool)
        atm_idx_check[self.z[0]] = True
        cart_crds                = np.zeros((len(self.z),3)) * ureg.nanometer
        for z_idx in range(len(self.z)):
            atm_idxs = self.z[z_idx]
            if not np.all(atm_idx_check[atm_idxs[1:]]):
                raise Exception(f"Not all atoms for row {z_idx} properly defined.")
            if z_idx == 0:
                A        = attach_crds[:,0].to(ureg.nanometer)
                B        = attach_crds[:,1].to(ureg.nanometer)
                C        = attach_crds[:,2].to(ureg.nanometer)
                bond     = virtual_bond.to(ureg.nanometer)
                angle    = virtual_angles[0].to(ureg.radian)
                dihedral = virtual_dihedrals[0].to(ureg.radian)
            elif z_idx == 1:
                A        = attach_crds[:,1].to(ureg.nanometer)
                B        = attach_crds[:,2].to(ureg.nanometer)
                C        = cart_crds[atm_idxs[1]].to(ureg.nanometer)
                bond     = z_crds[z_idx][0].to(ureg.nanometer)
                angle    = virtual_angles[1].to(ureg.radian)
                dihedral = virtual_dihedrals[1].to(ureg.radian)
            elif z_idx == 2:
                A        = attach_crds[:,2].to(ureg.nanometer)
                B        = cart_crds[atm_idxs[2]].to(ureg.nanometer)
                C        = cart_crds[atm_idxs[1]].to(ureg.nanometer)
                bond     = z_crds[z_idx][0].to(ureg.nanometer)
                angle    = z_crds[z_idx][1].to(ureg.radian)
                dihedral = virtual_dihedrals[2].to(ureg.radian)
            else:
                A        = cart_crds[atm_idxs[3]].to(ureg.nanometer)
                B        = cart_crds[atm_idxs[2]].to(ureg.nanometer)
                C        = cart_crds[atm_idxs[1]].to(ureg.nanometer)
                bond     = z_crds[z_idx][0].to(ureg.nanometer)
                angle    = z_crds[z_idx][1].to(ureg.radian)
                dihedral = z_crds[z_idx][2].to(ureg.radian)

            r_cos_angle = np.cos(np.pi-angle.magnitude)*bond
            r_sin_angle = np.sin(np.pi-angle.magnitude)*bond

            cart_crds[atm_idxs[0]][0] = r_cos_angle
            cart_crds[atm_idxs[0]][1] = np.cos(dihedral.magnitude)*r_sin_angle
            cart_crds[atm_idxs[0]][2] = np.sin(dihedral.magnitude)*r_sin_angle
            BC  = (C-B).to(ureg.nanometer).magnitude
            BC /= np.linalg.norm(BC)
            AB  = (B-A).to(ureg.nanometer).magnitude
            AB /= np.linalg.norm(AB)
            N   = np.cross(AB,BC)
            N  /= np.linalg.norm(N)
            rot       = np.zeros((3,3), dtype=float)
            rot[:,0]  = BC
            rot[:,1]  = np.cross(N,BC)
            rot[:,1] /= np.linalg.norm(rot[:,1])
            rot[:,2]  = N
            cart_crds[atm_idxs[0]]     = np.dot(rot, cart_crds[atm_idxs[0]].magnitude)*ureg.nanometer
            cart_crds[atm_idxs[0]]    += C
            atm_idx_check[atm_idxs[0]] = True

        _cart_crds = np.zeros((len(self.z),3))*ureg.nanometer
        if z_order:
            for z_idx in range(len(self.z)):
                atm_idxs = self.z[z_idx]
                _cart_crds[z_idx] = cart_crds[atm_idxs[0]]
            cart_crds = _cart_crds

        return cart_crds

    def build_pretty_zcrds(self, crds):

        z_crds_dict = self.build_z_crds(crds)
        z_string    = []
        for z_idx, atm_idxs in self.z.items():
            atm      = self.rdmol.GetAtomWithIdx(atm_idxs[0])
            number   = atm.GetAtomicNum()
            element  = get_element(number)
            z_row    = [f"{element} "]
            if z_idx > 0:
                for i, z_idx2 in enumerate(self.zz[z_idx][1:]):
                    if i == 0:
                        value = z_crds_dict[z_idx][i].to(ureg.angstrom)
                    else:
                        value = z_crds_dict[z_idx][i].to(ureg.degree)
                    z_row.append(f"{z_idx2+1} {value.magnitude:6.4f} ")
            z_string.append("".join(z_row))
        return "\n".join(z_string)

    def build_z_crds(self, crds):

        z_crds_dict = dict()
        for z_idx, atm_idxs in self.z.items():
            z_crds_dict[z_idx] = list()
            if z_idx == 0:
                z_crds_dict[z_idx].append(crds[atm_idxs[0]].to(ureg.nanometer))
            if z_idx > 0:
                dist = pts_to_bond(crds[atm_idxs[0]],
                                   crds[atm_idxs[1]])
                z_crds_dict[z_idx].append(dist)
            if z_idx > 1:
                ang = pts_to_angle(crds[atm_idxs[0]],
                                   crds[atm_idxs[1]],
                                   crds[atm_idxs[2]])
                z_crds_dict[z_idx].append(ang.to(ureg.degree))
            if z_idx > 2:
                dih = pts_to_dihedral(crds[atm_idxs[0]],
                                      crds[atm_idxs[1]],
                                      crds[atm_idxs[2]],
                                      crds[atm_idxs[3]])
                z_crds_dict[z_idx].append(dih.to(ureg.degree))
        return z_crds_dict
    
