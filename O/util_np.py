from . import DIR_main

''' util_np.py

saving/loading python variables and objects:
    f : save_pickle_
    f : load_pickle_

numpy array reshaping (single component systems only):
    f : reshape_to_molecules_np_
    f : reshape_to_atoms_np_
    f : reshape_to_flat_np_

misc:
    f : cumulative_average_
    f : sta_array_
    f : half_way_
    f : take_random_
    f : joint_grid_from_marginal_grids_

no-jump (molecules already whole):
    f : tidy_crystal_xyz_

'''
import sys

import os
import re
import copy
import time

from pathlib import Path
import subprocess
#import glob

import numpy as np
import scipy as sp

from rdkit import Chem
import mdtraj

import pickle

## ## 

def inject_methods_from_another_class_(target_instance, source_class, include_properties=False):
    import types
    for name, item in source_class.__dict__.items():
        
        if name.startswith("__"): continue
        else: pass
            
        if callable(item):
            setattr(target_instance, name, types.MethodType(item, target_instance))
        else: pass
            
        if include_properties and isinstance(item, (property, types.GetSetDescriptorType, types.MemberDescriptorType)):
            setattr(target_instance.__class__, name, item)
        else: pass

## ## 

def save_pickle_(x, name, verbose=True):
    ''' save any python variable, or instance of an object as a pickled file with name
    '''
    with open(name, "wb") as f: pickle.dump(x, f)
    if verbose: print('saved',name)
    else: pass
    
def load_pickle_(name):
    ''' load the pickled file with name back into python
    '''
    with open(name, "rb") as f: x = pickle.load(f) ; return x

## ## 

def reshape_to_molecules_np_(r, n_molecules, n_atoms_in_molecule):
    '''
    Output: (m, n_mol, n_atoms_mol, 3) array 
    '''
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules, n_atoms_in_molecule, 3])

def reshape_to_atoms_np_(r, n_molecules, n_atoms_in_molecule):
    '''
    Output: (m, n_mol * n_atoms_mol, 3) = (m,N,3) array 
    '''
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules*n_atoms_in_molecule, 3])
    
def reshape_to_flat_np_(r, n_molecules, n_atoms_in_molecule):
    '''
    Output: (m, n_mol * n_atoms_mol * 3) array 
    '''
    n_frames = r.shape[0]
    return r.reshape([n_frames, n_molecules*n_atoms_in_molecule*3])

## ## 

cumulative_average_ = lambda x,axis=None : np.cumsum(x,axis=axis) / np.cumsum(np.ones_like(x),axis=axis)

sta_array_ = lambda x : (x-x.min())/(x.max()-x.min())

cdist_ = sp.spatial.distance.cdist

def half_way_(a,c):
    '''
    output: number between a and c
    '''
    ac = sorted([a,c])
    b = min(ac) + 0.5*(max(ac) - min(ac))
    return b

def take_random_(x, m=20000):
    '''
    output: uniformly taken random m values from first axis of x (len(x) >= m)
    '''
    return x[np.random.choice(x.shape[0],min([m,x.shape[0]]),replace=False)]

def find_split_indices_(u, split_where:int, tol=0.00001, verbose=True):
    ''' training : validation split where both sets have same average potential energy within tol
    Inputs:
        u : (m,1) array of potential energies during MD sampling
        split_where : int, how many samples wanted in the training set
        tol : how similar should average energy of training set be to the average energy of the validation set
    Outputs:
        inds_rand or None: run multiple times until returns not None, or increase tol
            inds_rand : permutation of u (i.e., u[inds_rand]),
            where the first split_where points/samples are belong to the training set,
            and the rest of the array (i.e., u[inds_rand][split_where:]) validation set.
            Use this permuation on any other array relevant for training: r, u, w, b
    '''
    u = np.array(u)
    n = u.shape[0]
    target = u.mean()
    for i in range(1000):
        inds_rand = np.random.choice(n,n,replace=False)
        randomised = np.array(u[inds_rand])
        if np.abs(randomised[:split_where].mean() - target) < tol and np.abs(randomised[split_where:].mean() - target) < tol:
            if verbose: print('found !')
            else: pass
            return inds_rand
        else: pass
    if verbose: print('! not found')
    else: pass
    return None

def joint_grid_from_marginal_grids_(*marginal_grids, flatten_output=True):
    
    ''' like np.meshgrid but easier to use 
    Inputs:
        *marginal_grids : more than one flat arrays, these are usually grids made by np.linspace
            dim = number of input grids
        flatten_output : bool affecting shape of the output array
    Outputs:
        if flatten_output:
            joint_grid : (N, dim) ; N = bins[1]*...*bins[dim]
        else:
            joint_grid : (dim, bins[1], ..., bins[dim])
    '''

    list_marginal_grids = list(marginal_grids)
    letters = 'jklmnopqrst'
    dim = len(list_marginal_grids)
    bins = [len(x) for x in list_marginal_grids]

    Xs = []
    string_input = 'io,'
    string_output = 'oi'
    for i in range(dim):
        X = np.ones([bins[i],dim])
        X[:,i] = np.array(list_marginal_grids[i])
        Xs.append(X)
        if i > 0:
            string_input += letters[i]+'o,'
            string_output += letters[i]
        else: pass

    string = string_input[:-1]+'->'+string_output #; print(string)
    
    joint_grid = np.einsum(string,*Xs)

    if flatten_output:
        joint_grid = joint_grid.T.reshape(-1, dim)
    else: pass

    return joint_grid

def tidy_crystal_xyz_(r, b, n_atoms_mol, ind_rO, batch_size=1000):
    ''' makes molecules not jump in Cartesian space

    !! may not work well in unstable systems such as very small cells

    Inputs:
        r : (n_frames, N_atoms, 3) 
            array of coordinates (must be a single component system)
            molecules must be already whole (true by default in any openmm trajectories)
            (if molecules not whole see the method in SC_helper.unwrap_molecule, run that first)

        b : (n_frames, 3, 3)
            array of simulation boxes
        ind_rO : int
            index of any atom in a molecule that has slow dynamics relative to the cell
        batch_size : int
            to reuduce memory cost when running on large trajectory
    Outputs:
        r : (n_frames, N_atoms, 3)
            array of coordinates with PBC wrapping where molecules are not jumping
            Importantly: outputs are expected to evaluate to the same energy as the inputs (same packing as input)

    '''
    def check_shape_(x):
        x = np.array(x)
        shape = len(x.shape)
        assert shape in [2,3]
        if len(x.shape) == 3: pass
        else: x = x[np.newaxis,...]
        return x

    r = check_shape_(r)
    n_frames = r.shape[0]
    batch_size = min([batch_size, n_frames])
    
    if len(b.shape) == 2: b = np.array([b]*n_frames)
    else: assert b.shape[0] == n_frames
    def wrap_points_(R, box):
        # R   : (... 3), shaped as molecules
        # box : (...,3, 3) # rows
        st = 'oabi,oij->oabj'
        return np.einsum(st, np.mod(np.einsum(st, R, np.linalg.inv(box)), 1.0), box)
    
    N = r.shape[1]
    n_mol = N // n_atoms_mol
    assert n_mol == N / n_atoms_mol
    '''
    # step 1 : put atoms with index rO into box (and bring whole molecule with it)
    '''
    r = reshape_to_molecules_np_(r, n_atoms_in_molecule=n_atoms_mol, n_molecules=n_mol)
    for i in range(n_frames//batch_size):
        _from = i*batch_size
        _to = (i+1)*batch_size
        rO = r[_from:_to,:,ind_rO:ind_rO+1]
        r[_from:_to] = r[_from:_to] - rO + wrap_points_(rO,b[_from:_to])

    if n_frames - _to > 0:
        _from = _to
        rO = r[_from:,:,ind_rO:ind_rO+1]
        r[_from:] = r[_from:] - rO + wrap_points_(rO,b[_from:])
    else: pass
    '''
    # step 2: bring any atoms with index rO that are still jumping to pre-jump position (and bring whole molecule with it)
    using method copied from: https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/transformations/nojump.py
    this should give lattice looking like the first frame throughout a crystaline trajectory
    '''
    def dot_(Ri, mat):
        st = 'abi,ij->abj'
        return np.einsum(st, Ri, mat)

    rO = np.array(r[:,:,ind_rO:ind_rO+1])
    b_inv = np.linalg.inv(b)
    
    rO_revised = np.zeros_like(rO)
    rO_revised[0] = rO[0]
    rO_0 = dot_(rO[0], b_inv[0])
    for i in range(1,n_frames):
        rO_1 = dot_(rO[i], b_inv[i])
        rO_1 -= np.round( rO_1 - rO_0 )
        rO_revised[i] = dot_(rO_1, b[i])

    r = r - rO + rO_revised
    # if remove_COM:
    r -= r[:,:,ind_rO:ind_rO+1].mean(1, keepdims=True)
    # else: pass
    r = reshape_to_atoms_np_(r, n_atoms_in_molecule=n_atoms_mol, n_molecules=n_mol)
    return r

## ## 

def get_torsion_np_(r, inds_4_atoms):
    ''' REF: https://github.com/noegroup/bgflow '''
    # r            : (..., # atoms, 3)
    # inds_4_atoms : (4,)
    
    A,B,C,D = inds_4_atoms
    rA = r[...,A,:] # (...,3)
    rB = r[...,B,:] # (...,3)
    rC = r[...,C,:] # (...,3)
    rD = r[...,D,:] # (...,3)
    
    vBA = rA - rB   # (...,3)
    vBC = rC - rB   # (...,3)
    vCD = rD - rC   # (...,3)

    _clip_low_at_ = 1e-8
    _clip_high_at_ = 1e+18
    clip_positive_ = lambda x : np.clip(x, _clip_low_at_, _clip_high_at_) 
    norm_clipped_ = lambda x : clip_positive_(np.linalg.norm(x,axis=-1,keepdims=True))
    unit_clipped_ = lambda x : x / norm_clipped_(x)
    
    uBC = unit_clipped_(vBC) # (...,3)

    w = vCD - np.sum(vCD*uBC, axis=-1, keepdims=True)*uBC # (...,3)
    v = vBA - np.sum(vBA*uBC, axis=-1, keepdims=True)*uBC # (...,3)
    
    uBC1 = uBC[...,0] # (...,)
    uBC2 = uBC[...,1] # (...,)
    uBC3 = uBC[...,2] # (...,)
    
    zero = np.zeros_like(uBC1) # (...,)
    S = np.stack([np.stack([ zero, uBC3,-uBC2],axis=-1),
                np.stack([-uBC3, zero, uBC1],axis=-1),
                np.stack([ uBC2,-uBC1, zero],axis=-1)],axis=-1) # (...,3,3)
    
    y = np.expand_dims(np.einsum('...j,...jk,...k->...',w,S,v), axis=-1) # (...,1)
    x = np.expand_dims(np.einsum('...j,...j->...',w,v), axis=-1)         # (...,1)
    
    phi = np.arctan2(y,x) # (...,1)

    return phi # (...,1)

def get_angle_np_(R, inds_3_atoms):
    ''' bond angle '''
    # R            : (..., # atoms, 3)
    # inds_3_atoms : (3,)

    A,B,C = inds_3_atoms
    rA = R[...,A,:] # (...,3)
    rB = R[...,B,:] # (...,3)
    rC = R[...,C,:] # (...,3)

    _clip_low_at_ = 1e-8
    _clip_high_at_ = 1e+18
    clip_positive_ = lambda x : np.clip(x, _clip_low_at_, _clip_high_at_) 
    norm_clipped_ = lambda x : clip_positive_(np.linalg.norm(x,axis=-1,keepdims=True))
    unit_clipped_ = lambda x : x / norm_clipped_(x)

    uBA = unit_clipped_(rA - rB) # (...,3)
    uBC = unit_clipped_(rC - rB) # (...,3)

    dot = np.sum(uBA*uBC, axis=-1, keepdims=True)             # (...,1)
    dot = np.clip(dot, -1.0, 1.0)                           # (...,1)
    
    theta = np.arccos(dot) # (...,1)
    theta = np.clip(theta, _clip_low_at_, np.pi-_clip_low_at_) # (...,1)
 
    return theta # (...,1)

def get_distance_np_(R, inds_2_atoms):
    ''' bond distance '''
    # R            : (..., # atoms, 3)
    # inds_2_atoms : (2,)
    A,B = inds_2_atoms
    rA = R[...,A,:]  # (...,3)
    rB = R[...,B,:]  # (...,3)
    vBA = rA - rB    # (...,3)

    _clip_low_at_ = 1e-8
    _clip_high_at_ = 1e+18
    clip_positive_ = lambda x : np.clip(x, _clip_low_at_, _clip_high_at_) 
    norm_clipped_ = lambda x : clip_positive_(np.linalg.norm(x,axis=-1,keepdims=True))

    return norm_clipped_(vBA) # (...,1)

## ## 

def color_text_(text, p='_R'):
    # REF: https://stackoverflow.com/questions/8924173/how-can-i-print-bold-text-in-python
    selection = ''
    if '_' in p: selection += '\033[4m'
    else: pass
    if p.isupper(): selection += '\033[1m'
    else: pass
    color = {   'p'   : '\033[95m', 'c'   : '\033[96m', 'dc'  : '\033[36m',
                'b'   : '\033[94m', 'g'   : '\033[92m', 'y'   : '\033[93m',
                'r'   : '\033[91m', 'o'   : '\033[38;5;208m', 'i' : '',
    }[p.replace('_', '').lower()]
    selection += color
    return selection + str(text) + '\033[0m'

## ## 

class TestConverged_1D:
    def __init__(self,
                 x,
                 tol = 0.2,
                 verbose = True,
                ):
        self.tol = tol
        
        x = np.array(x).flatten()
        MU = cumulative_average_(x)
        VAR = cumulative_average_((x-MU)**2)
        err = np.abs(MU - cumulative_average_(MU))**2
        err = np.ma.divide(err,VAR)**0.5
        err *= 10.0
        self.err = np.array(err)
        
        if verbose:
            gR_ = lambda _bool : ['R','g'][np.array(_bool).astype(np.int32)]
            b = self.__call__()
            print(f'with tol = {self.tol}, is converged: {color_text_(b, gR_(b))}')
        else: pass
        
        self.MU = MU
        self.x = np.array(x)

    def __call__(self):
        return self.err[-1] <= self.tol

    @property
    def where(self):
        return np.where(self.err <= self.tol)[0]
    
    @property
    def recommend_cut_from(self,):
        # index of frame after which the quantity may be converged
        idx = len(self.x) - len(TestConverged_1D(np.flip(self.x), tol=self.tol, verbose=False).where)
        if TestConverged_1D(self.x[idx:], tol=self.tol, verbose=False)():
            print('the quantity might be converged after frame with index:', idx)
        else: print('!!')
        return idx
    
    def show_(self, window=1, centre=False, show_x = True, color='black'):
        # scatter is faster than plot
        import matplotlib.pyplot as plt
        mean = self.MU[-1]
        m = len(self.x)
        t = np.arange(m)
        
        if centre:
            if show_x: plt.scatter(t, self.x-mean, alpha=0.5,s=0.01, color=color)
            plt.scatter(t, self.MU - mean, alpha=1, s=1, color=color)
            plt.plot([0,m], [0]*2, color=color, linestyle='--')
            plt.ylim(-window, window)
        else:
            if show_x: plt.scatter(t, self.x, alpha=0.5,s=0.01, color=color)
            plt.scatter(t, self.MU, alpha=1, s=1, color=color)
            plt.plot([0,m], [mean]*2, color=color, linestyle='--')
            plt.ylim(mean-window, mean+window)
        
## ## 

def K_to_C_(K):
    return K - 273.15

def C_to_K_(C):
    return C + 273.15

## ## 











