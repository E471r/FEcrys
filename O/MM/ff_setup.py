from .mm_helper import *

## ## ## ## ## ## ## ## ## ## ## ##
## moved from sc_system to here:

def change_charges_itp_top_(path_top_or_itp_file_in : str,
                            path_top_or_itp_file_out : str,
                            n_atoms_mol : int,
                            replacement_charges : np.ndarray = None, #! correct permuation of atoms
                            neutralise_charge : bool = True,
                            ):
    print('')
    print('__ changing charges in top/itp: __________________________')
    file_in_path = str(path_top_or_itp_file_in)
    file_out_path = str(path_top_or_itp_file_out)
    
    file_in = open(file_in_path, 'r')
    lines_in = []
    for line in file_in:
        lines_in.append(line)
    file_in.close()
    
    indx_header = [i for i in range(len(lines_in)) if '[ atoms ]' in lines_in[i]][0]
    
    inds_lines_replace = []
    charges_in_str = []
    i = indx_header
    while len(charges_in_str) < n_atoms_mol:
        split = re.split('\s+',lines_in[i])

        #if len(split) == 13 and split[0] != ';':
        #    inds_lines_replace.append(i)
        #    charges_in_str.append(split[7])
        #else: pass
        try:
            if '.' in split[7] and '.' in split[8]: # charge (7), mass (8) parts have a dot..
                inds_lines_replace.append(i)
                charges_in_str.append(split[7])
            else: pass
        except: pass

        i +=1
        
    charges_in = np.array([float(x) for x in charges_in_str])
    if replacement_charges is None:
        if neutralise_charge:
            charges_out = charges_in - charges_in.mean()
            print('average charge neutralised from',charges_in.mean(),'to',charges_out.mean())
        else:
            charges_out = charges_in
            print('charges with average',charges_in.mean(),'were unchanged')
    else:
        assert len(replacement_charges) == n_atoms_mol
        charges_out = replacement_charges
        print('charges with average',charges_in.mean(),'were repalced with custom charges with average of',replacement_charges.mean())
        
    charges_out_str = [str(x) for x in charges_out]
    
    lines_out = dict(zip(np.arange(len(lines_in)), lines_in))
    lines_replaced = []
    for i, charge_in_str, charge_out_str in zip(inds_lines_replace, charges_in_str ,charges_out_str):
        lines_out[i] = lines_out[i].replace(charge_in_str, charge_out_str)
        lines_replaced.append(i)
    
    print('replaced',len(lines_replaced),'lines from',lines_replaced[0],'to',lines_replaced[-1],'in',file_in_path)

    file_out = open(file_out_path, 'w')
    for i in range(len(lines_out)):
        file_out.write(lines_out[i])
    file_out.close()

    print('these changes were written into file:',file_out_path)

    print('__________________________________________________________\n')
    return charges_out

def change_n_mol_top_(path_top_file_in : str,
                      path_top_file_out : str,
                      replace_n_mol : int,
                      verbose = True,
                     ):
    if verbose: print_ = lambda *x : print(*x)
    else: print_ = lambda *x : None
    print_('')
    print_('__ changing n_mol in top: ________________________________')
    file_in_path = str(path_top_file_in)
    file_out_path = str(path_top_file_out)

    file_in = open(file_in_path, 'r')
    lines_in = []
    for line in file_in:
        lines_in.append(line)
    file_in.close()

    indx_header = [i for i in range(len(lines_in)) if '[ molecules ]' in lines_in[i]][0]

    inds_lines_replace = []
    n_mol_in_str = []
    i = indx_header
    while len(n_mol_in_str) < 1:
        split = re.split('\s+',lines_in[i])
        if len(split) == 3 and split[0] != ';':
            inds_lines_replace.append(i)
            n_mol_in_str.append(split[1])
        else: pass
        i +=1
    assert len(n_mol_in_str) == 1
    assert len(inds_lines_replace) == 1

    lines_out = dict(zip(np.arange(len(lines_in)), lines_in))
    i = inds_lines_replace[0]
    lines_out[i] = lines_out[i].replace(n_mol_in_str[0], str(replace_n_mol))

    print_('in the',file_in_path)
    print_('replaced 1 line (',i,') \n from: \n    ',lines_in[i] ,'to \n    ',lines_out[i][:-1])

    file_out = open(file_out_path, 'w')
    for i in range(len(lines_out)):
        file_out.write(lines_out[i])
    file_out.close()

    print_('these changes were written into file:',file_out_path)
    print_('__________________________________________________________\n')

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class methods_for_permutation:

    ## ## ## ## ## ## ## ## ## ## ## ##
    ## moved from OPLS:

    @property
    def _current_r_(self,):
        # positions
        # unit = nanometer
        return self._unpermute_(self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value)

    @property
    def _current_v_(self,):
        # velocities
        # unit = nanometer/picosecond
        return self._unpermute_(self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)._value)
    
    @property
    def _current_F_(self,):
        # forces
        # unit = kilojoule/(nanometer*mole) ; [F = -∇U]
        return self._unpermute_(self.simulation.context.getState(getForces=True).getForces(asNumpy=True)._value)
    
    @property
    def _system_mass_(self,):
        return self._unpermute_(np.array([self.system.getParticleMass(i)._value for i in range(self.system.getNumParticles())]),
                                axis=0)

    def _set_r_(self, r):
        assert r.shape == (self.N,3)
        self.simulation.context.setPositions(self._permute_(r))

    def _set_v_(self, v):
        assert v.shape == (self.N,3)
        self.simulation.context.setVelocities(self._permute_(v))

    def forward_atom_index_(self, inds):
        # select atom in openmm using this layer that converts to the intended index
        return self._unpermute_crystal_from_top_[inds]
    
    def inverse_atom_index_(self, inds):
        return self._permute_crystal_to_top_[inds]

    ## ## ## ## ## ## ## ## ## ## ## ##
    ## moved from COST_FIX_permute_xyz_after_a_trajectory:

    def set_arrays_blank_(self,):
        self._xyz_top_ = []
        self._xyz = []
        self._u = []
        self._temperature = []
        self._boxes = []
        self._COMs = []
        self.n_frames_saved = 0
        # self._Ps = []
        # self._v = []

    def save_frame_(self,):
        self._xyz_top_.append( self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value ) # nm
        self._u.append( self._current_u_ )            #  kT
        self._temperature.append( self._current_T_  ) # K
        self._boxes.append( self._current_b_ )        # nm
        self._COMs.append( self._current_COM_ )       # nm
        self.n_frames_saved += 1                      # frames
        # self._Ps.append( self._current_P_ )
        # self._v.append(self._current_v_)

    def run_simulation_(self, n_saves, stride_save_frame:int=100, verbose_info : str = ''):
        self.stride_save_frame = stride_save_frame
        for i in range(n_saves):
            self.simulation.step(stride_save_frame)
            self.save_frame_()
            info = 'frame: '+str(self.n_frames_saved)+' T sampled:'+str(self.temperature.mean().round(3))+' T expected:'+str(self.T)+verbose_info
            print(info, end='\r')

        self._xyz += [x for x in self._unpermute_(np.array(self._xyz_top_), axis=-2)] # permute after
        self._xyz_top_ = [] # missing this line was not an error when running one simulation all in one go
        # ! interupting run_simulation_ will not save any xyz data. 

    def run_simulation_w_(self, n_saves, stride_save_frame:int=100, verbose_info : str = ''):
        self.stride_save_frame = stride_save_frame
        for i in range(n_saves):
            self.simulation.step(stride_save_frame)
            state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.simulation.context.setPositions(state.getPositions())
            self.save_frame_()
            info = 'frame: '+str(self.n_frames_saved)+' T sampled:'+str(self.temperature.mean().round(3))+' T expected:'+str(self.T)+verbose_info
            print(info, end='\r')

        self._xyz += [x for x in self._unpermute_(np.array(self._xyz_top_), axis=-2)] # permute after
        self._xyz_top_ = [] # missing this line was not an error when running one simulation all in one go
        # ! interupting run_simulation_ will not save any xyz data. 
  
    def u_(self, r, b=None):
        '''
        speed up evaluation also # ..
        '''
        n_frames = r.shape[0]
        r = np.array(self._permute_(r)) # permute before 
        
        _r = np.array(self._current_r_)
        _b = np.array(self._current_b_)
        
        U = np.zeros([n_frames,1])
        if b is None:
            for i in range(n_frames):
                self.simulation.context.setPositions(r[i])
                U[i,0] = self._current_U_
        else:
            for i in range(n_frames):
                self.simulation.context.setPositions(r[i])
                self._set_b_(b[i])
                U[i,0] = self._current_U_

        self._set_r_(_r)
        self._set_b_(_b)

        return U*self.beta

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class itp2FF(MM_system_helper):
    def __init__(self,):
        super().__init__()
        self.using_gmx_loader = True

    @property
    def itp_mol(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_{self.name}.itp"
    
    @property
    def itp_mol_adjusted_charges(self) -> Path:
        # default : not written or used
        return self.misc_dir / f"{self._FF_name_}_{self.name}_adjusted_charges.itp"
    
    @property
    def top_crys(self) -> Path:
        return self.misc_dir / f"x_x_{self._FF_name_}_{self.name}.top"

    @property
    def gro_mol(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_{self.name}.gro"

    @property
    def pdb_mol(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_{self.name}.pdb"

    @property
    def single_mol_pdb(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_single_mol_{self.name}.pdb" 
    @property
    def _single_mol_pdb_file_(self):
        return self.misc_dir / f"{self._FF_name_}_single_mol_{self.name}.pdb" 
    
    @property
    def single_mol_pdb_permuted(self) -> Path:
        return self.misc_dir / f"{self._FF_name_}_single_mol_permuted_{self.name}.pdb" # to match itp
    
    @property
    def single_mol_permutations(self,) -> Path:
        return self.misc_dir / f"{self._FF_name_}_single_mol_permutations_{self.name}"

    def set_pemutation_to_match_topology_(self,):
        try: 
            self._permute_molecule_to_top_, self._unpermute_molecule_from_top_ = load_pickle_(str(self.single_mol_permutations.absolute()))
        except:
            ' explained in detail in OPLS class with same method '
            print(f'first time running {self._FF_name_}_general for this moelcule? setting permuation in case needed:')

            pdb = str(self.single_mol_pdb.absolute())
            gro = str(self.gro_mol.absolute())

            pdb_from_gro = str(self.pdb_mol.absolute())
            pdb_reordered = str(self.single_mol_pdb_permuted.absolute())

            if os.path.exists(self.pdb_mol.absolute()): pass 
            else:
                import MDAnalysis as mda # # conda install -c conda-forge mdanalysis
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    universe = mda.Universe(gro)
                    with mda.Writer(pdb_from_gro) as x:
                        x.write(universe)

            reorder_atoms_mol_(mol_pdb_fname=pdb, template_pdb_fname=pdb_from_gro, output_pdb_fname=pdb_reordered)

            r_gro = mdtraj.load(pdb_reordered, pdb_reordered).xyz[0]
            r_pdb = mdtraj.load(pdb, pdb).xyz[0]

            D = cdist_(r_pdb, r_gro, metric='euclidean') # (n_atoms_mol,n_atoms_mol)
            self._permute_molecule_to_top_     = D.argmin(0) # (n_atoms_mol,1)
            self._unpermute_molecule_from_top_ = D.argmin(1) # (n_atoms_mol,1)

            assert np.abs(self._permute_molecule_to_top_[self._unpermute_molecule_from_top_] - np.arange(self.n_atoms_mol)).sum() == 0

            print('forward permutation works:', np.abs(r_pdb[self._permute_molecule_to_top_] - r_gro).max()     == 0.0)
            print('inverse permatation works:', np.abs(r_pdb - r_gro[self._unpermute_molecule_from_top_]).max() == 0.0)

            save_pickle_([self._permute_molecule_to_top_, self._unpermute_molecule_from_top_], str(self.single_mol_permutations.absolute()))
            
        self._permute_crystal_to_top_ = np.concatenate([self._permute_molecule_to_top_ + i*self.n_atoms_mol for i in range(self.n_mol)], axis=0)
        self._unpermute_crystal_from_top_ = np.concatenate([self._unpermute_molecule_from_top_ + i*self.n_atoms_mol for i in range(self.n_mol)], axis=0)
        assert np.abs(self._permute_crystal_to_top_[self._unpermute_crystal_from_top_] - np.arange(self.N)).sum() == 0

        if np.sum(np.abs(self._permute_crystal_to_top_ - np.arange(self.N))) == 0:
            self.atom_order_PDB_match_itp = True
            print('permutation not used, because not needed')
        else:
            self.atom_order_PDB_match_itp = False
            self._permute_   = lambda r, axis=-2 : np.take(r, self._permute_crystal_to_top_, axis=axis)
            self._unpermute_ = lambda r, axis=-2 : np.take(r, self._unpermute_crystal_from_top_, axis=axis)
            print(f'! using {self._FF_name_} ff with permuation of atoms turned ON.')
            if self.n_mol > 1: print('this assumes all molecules in the input PDB have the same permutation as the first molecule')
            else: pass
            print("'permuation of atoms turned ON' -> to reduce cost during run_simulation_ the method for saving xyz frames is slightly adjusted")
            self.inject_methods_from_another_class_(methods_for_permutation, include_properties=True)

    def a_step_after_initialise_(self,):
        pass

    def initialise_FF_(self, neuralise_net_charge=False, replacement_charges=None):
        ''' run this only after (n_mol and n_atoms_mol) defined in __init__ of SingleComponent '''
        if os.path.exists(self.single_mol_pdb.absolute()): pass
        else: process_mercury_output_(self.PDB, self.n_atoms_mol, single_mol = True, custom_path_name=str(self.single_mol_pdb.absolute()))
            
        if os.path.exists(self.itp_mol.absolute()): pass
        else: print('!! expected file not found:',self.itp_mol.absolute(),)

        self.set_pemutation_to_match_topology_()

        if neuralise_net_charge or replacement_charges is not None:
            itp_mol_adjusted_charges = self.itp_mol_adjusted_charges.absolute()
            if not os.path.exists(itp_mol_adjusted_charges) or replacement_charges is not None:
                change_charges_itp_top_(path_top_or_itp_file_in = str(self.itp_mol.absolute()),
                                        path_top_or_itp_file_out = str(itp_mol_adjusted_charges),
                                        n_atoms_mol= self.n_atoms_mol,
                                        replacement_charges = replacement_charges,
                                        neutralise_charge = neuralise_net_charge,
                                        )
            else: pass
            self.adjusted_charges = True
        else: self.adjusted_charges = False

        self.a_step_after_initialise_()

    def set_FF_(self,):
        ''' run this just before self.system initialisation  '''
        self.reset_n_mol_top_()
        self.ff = parmed.gromacs.GromacsTopologyFile(str(self.top_crys.absolute())) # better
        # self.ff = mm.app.GromacsTopFile(self.top_crys.absolute(), periodicBoxVectors=self.b0)
    
    def reset_n_mol_top_(self,):
        if self.adjusted_charges: line0 = f'#include "{self._FF_name_}_{self.name}_adjusted_charges.itp"'
        else:                     line0 = f'#include "{self._FF_name_}_{self.name}.itp"'

        lines_to_add = [
            line0,
            '\n',
            '[ defaults ]',
            '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
            self._FF_name_defaults_line_,
            '\n',
            '[ system ]',
            '; Name',
            self._system_name_,
            '\n',
            '[ molecules ]',
            '; Compound          #mols',
            f'{self._compound_name_}               {self.n_mol}',
            '\n',
            ]
        file_top = open(str(self.top_crys.absolute()), 'w')
        for line in lines_to_add:
            file_top.write(line + "\n")
        file_top.close()


class OPLS_general(itp2FF):
    def __init__(self,):
        super().__init__()
        self._FF_name_ = 'OPLS'
        #                           '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
        self._FF_name_defaults_line_ = '1               3               yes             0.5          0.5  '
        self._system_name_ = 'Generic title'
        self._compound_name_ = 'UNK'
        
    @classmethod
    @property
    def FF_name(self,):
        return 'OPLS'

class GAFF_general(itp2FF):
    def __init__(self,):
        super().__init__()
        self._FF_name_ = 'GAFF'
        #                           '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
        self._FF_name_defaults_line_ = '1               2               yes             0.5          0.83333333  '
        self._system_name_ = 'Generic title'
        self._compound_name_ = 'UNK'
        
    @classmethod
    @property
    def FF_name(self,):
        return 'GAFF'

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def remove_force_by_names_(system, names:list, verbose=True):
    def remove_force_by_name_(_name):
        index = 0
        for force in system.getForces():
            if force.getName() == _name: 
                system.removeForce(index)
                return [_name]
            else: index += 1
        return []
        
    removed = []
    for name in names:
        while name in [force.getName() for force in system.getForces()]:
            removed += remove_force_by_name_(name)
    if verbose: print(f'removed {len(removed)} forces from the system: {removed}')
    else: pass

def _get_pairs_mol_inner_(single_mol_pdb_file, n=3):
    '''
    n atoms in a row is n-1 bonds

        n = 3 for this velff because nrexcl = 2
            within 2 bonds away removed
            within 3 bonds away kept (1-2-3-4 ; 1-4 kept)
    '''
    # important that this is the pdb taken from the .gro file
    # in velff no permutaion was used, but in general should not use self.mol here.
    mol = Chem.MolFromPDBFile(str(single_mol_pdb_file), removeHs=False)
    
    AM = Chem.rdmolops.GetAdjacencyMatrix( mol )
    n_atoms_mol = len(AM)

    import networkx as nx
    '''
    gr = nx.Graph()
    for i in range(n_atoms_mol):
        for j in range(n_atoms_mol):
            if AM[i,j]>0.5:
                gr.add_edge(i,j)
            else: pass 
    '''
    gr = nx.from_numpy_array(AM, create_using = nx.DiGraph)
    assert np.abs(nx.adjacency_matrix(gr).toarray() - AM).sum() == 0
    
    within_n_bonds_away = np.eye(n_atoms_mol)*0
    for i in range(n_atoms_mol):
        for j in range(n_atoms_mol):
            n_atoms_in_a_row = len(nx.shortest_path(gr, i,j))
            if n_atoms_in_a_row <= n: # n-1 bonds away
                within_n_bonds_away[i,j] = 1
            else: pass
                
    return within_n_bonds_away # 1 : inner, 0 : outer

def _get_pairs_(remove_mol_ij, n_mol):

    n_atoms_mol = len(remove_mol_ij)

    include_ij = np.eye(n_atoms_mol*n_mol)*0 + 1 # 1 : include
    for i in range(n_mol):
        a = n_atoms_mol*i
        b = n_atoms_mol*(i+1)
        include_ij[a:b,a:b] -= remove_mol_ij # remove 1-1, 1-2, 1-3
        
    return include_ij # 1 : include, 0 : dont include

def custom_LJ_force_(sc, C6_C12_types_dictionary):
    ''' LJ : all Lennard-Jones '''

    include_ij = _get_pairs_(_get_pairs_mol_inner_(sc.pdb_mol, n=3), sc.n_mol)
    atom_types = [x.type for x in sc.ff.atoms]
    
    _table_C6 = np.eye(sc.N)*0.0
    _table_C12 = np.eye(sc.N)*0.0
    
    for i in range(sc.N):
        for j in range(sc.N):
            if i >= j:
                type_A = atom_types[i]
                type_B = atom_types[j]
                try:  C6, C12 = C6_C12_types_dictionary[(type_A,type_B)]
                except: C6, C12 = C6_C12_types_dictionary[(type_B,type_A)]
                
                ''' testing > 1-4 earlier
                sig_i, eps_i = opls_q_sig_eps[atom_types[i]][1:]
                sig_j, eps_j = opls_q_sig_eps[atom_types[j]][1:]
                
                eps_ij = np.sqrt(eps_i*eps_j)
                sig_ij = np.sqrt(sig_i*sig_j)
    
                C6  = 4.0 * eps_ij * (sig_ij**6)
                C12 = 4.0 * eps_ij * (sig_ij**12)
                '''
                # can add if needed: filter for fudge factor (multiply it to both C6 and C12 )
                
                _table_C6[i, j] = C6
                _table_C12[i, j] = C12
                _table_C6[j, i] = _table_C6[i, j]
                _table_C12[j, i] = _table_C12[i, j]
            else: pass
    
    table_C6 = mm.Discrete2DFunction(sc.N, sc.N, _table_C6.flatten().tolist())
    table_C12 = mm.Discrete2DFunction(sc.N, sc.N, _table_C12.flatten().tolist())
    
    force = mm.CustomNonbondedForce('ecm_lambda * ((1/r^12)*C12 - (1/r^6)*C6) ; C6 = table_C6(p1,p2) ; C12 = table_C12(p1,p2)')
    force.addGlobalParameter('ecm_lambda', 1.0)
    
    force.addPerParticleParameter('p')
    force.addTabulatedFunction('table_C6', table_C6)
    force.addTabulatedFunction('table_C12', table_C12)

    for i in range(sc.N):
        force.addParticle([i])

    for i in range(sc.N):
        for j in range(sc.N):
            if i > j: # !! not i>=j
                if include_ij[i,j] < 0.5:
                    force.addExclusion(i,j)
                else: pass
            else: pass
    
    if sc.n_mol > 1:
        nb_method = mm.CustomNonbondedForce.CutoffPeriodic
        force.setNonbondedMethod(nb_method)
        force.setCutoffDistance(sc.PME_cutoff * mm.unit.nanometers)
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(sc.SwitchingFunction_factor * sc.PME_cutoff * mm.unit.nanometers)
        force.setUseLongRangeCorrection(True)
    else:
        nb_method = mm.CustomNonbondedForce.NoCutoff
        force.setNonbondedMethod(nb_method)

    return [force]

def custom_C_force_(sc):
    ''' C : all Coulombic '''
    include_ij = _get_pairs_(_get_pairs_mol_inner_(sc.pdb_mol, n=3), sc.n_mol)

    q = np.concatenate([sc.partial_charges_mol]*sc.n_mol,axis=0) # (N,)

    force = mm.NonbondedForce()
    force.setNonbondedMethod( get_force_by_name_(sc.system, 'NonbondedForce').getNonbondedMethod() ) # 4 here ; 5 (LJ-PME), but cannot use it with velff
    force.setEwaldErrorTolerance(sc.custom_EwaldErrorTolerance)
    
    if sc.n_mol > 1:
        force.setCutoffDistance(sc.PME_cutoff * mm.unit.nanometers)
        force.setIncludeDirectSpace(True)
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(sc.SwitchingFunction_factor * sc.PME_cutoff * mm.unit.nanometers)
        force.setUseDispersionCorrection(True)
    else: pass
    
    for i in range(sc.N):
        force.addParticle(*[q[i] * unit.elementary_charge, 0.0, 0.0, ])

    for i in range(sc.N):
        for j in range(sc.N):
            if i > j: # !! not i>=j
                if include_ij[i,j] < 0.5:
                    force.addException(*[i, j, 0.0, 0.0, 0.0])
                else: pass # can add if needed: filter for fudge factor (multiply it to qq_ij)
            else: pass

    return [force]

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

''' notes (tmFF):

the following self._FF_name_defaults_line_ gives warnings that are dealt with by running self.recast_NB_() as soon as self.system

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    
nbfunc : same as OPLS (1) ; similar to this py file was tested for OPLS that already works by default.

gen-pairs : 1-4 interactions are active for all non-bonded (NB) interactions

nrexcl = 2  : non-bonded interactions are excluded only for atoms separated by 0,1,2 bonds
    ! default gmx parsers (parmed or openmm) : this is currently not supported (because kept for >=3 bonds apart)

comb-rule 1 : geometric mean for getting both eps_ij, sig_ij (or C6_ij from C6_i, C6_j, C12_ij from ...)
    this is supported only if combination rules are standard (e.g., OPLS), but not the case here, where for OPLS:
            NonbondedForce : some LJ, no Coulombic       # defined automatically
                < 1-5  : exceptions : qq_ij = 0 ; LJ = 0
                    set all zero
                = 1-5  : exceptions : qq_ij = fudgeQQ * qi * qj ; LJ = fudgeLJ * combine(i, j, LJ_params)
                    combined manually as exceptions
                > 1-5  : qi * qj ; LJ = 0
                    all remaining electrostatic (without any fudge)
            CustomNonbondedForce : most LJ, no Coulombic # defined automatically
                > 1-5  : LJ = customizable_energy_function(i, j, LJ_param, combination_rule)
                <= 1-5 : exclusions for 1-5, ..., 1-1 interactions

atoms types : itp file has LJ parameters already as combined as C6_ij and C12_ij for different atom type pairs
    ! default gmx parsers (parmed or openmm) : this is currently not supported

    the way this is dealt with in this ff (where fudgeQQ = fudgeLJ = 1):
        NonbondedForce : only Coulombic # defined manually below (to replace all automatically defined Coulombic)
            < 1-4  : exceptions : qq_ij = 0 ; LJ = 0
            >= 1-4 : qi * qj ; LJ = 0
        CustomNonbondedForce : only LJ  # defined manually below (to replace all automatically defined LJ)
            >= 1-4 : LJ = LJ_function(i, j, tabulated_LJ_params)
                tabulated_LJ_params is a function of i and j 
            < 1-4  : exclusions for 1-3, 1-2, 1-1
            
            LJ_energy = dispersion_correction(switching_function(LJ_function(...,r_cut),r_switch))
            dispersion_correction : x -> x + const*n_mol*n_mol / V
                the exactness of the const matters for delta_u when volumes are different
                the const takes into account the shapes of the smooth 1D functions > r_switch
'''

class tmFF(itp2FF):
    ''' tailor-made FF '''
    def __init__(self,):
        super().__init__()
        self._FF_name_ = 'tmFF'
        #                           '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
        self._FF_name_defaults_line_ = '1               1               yes             1.0          1.0  '
        self._system_name_ = 'organic'
        self._compound_name_ = 'molecule'
    
    @classmethod
    @property
    def FF_name(self,):
        return 'tmFF'
    
    def a_step_after_initialise_(self,):
        file_name = str(self.itp_mol.absolute())
        self.nonbond_params = {}
        start = False

        file = open(file_name,'r')
        for line in file:
            if '[ nonbond_params ]' in line:
                start = True
            else: pass
            if start:
                try:
                    line_split = line.split()
                    if line_split[0] != ';':
                        try:
                            typeA, typeB, _, C6, C12 = line_split
                            self.nonbond_params[(typeA, typeB)] = [float(C6), float(C12)]
                        except: pass # print(f'skipped line {line}')
                    else: pass
                except: pass
            else: pass
                
        file.close()
        keys = self.nonbond_params.keys()
        self.n_atom_types = len(set([x[0] for x in keys]))
        check = 0.5 * (self.n_atom_types**2 + self.n_atom_types)
        assert str(check).split('.')[-1] == '0'
        assert len(keys) == int(check)
        
    def recast_NB_(self, verbose=True):

        forces_add = custom_LJ_force_(self, C6_C12_types_dictionary=self.nonbond_params) + custom_C_force_(self) 

        remove_force_by_names_(self.system, 
                               ['CustomNonbondedForce','NonbondedForce'], # ,'HarmonicBondForce','HarmonicAngleForce','PeriodicTorsionForce','RBTorsionForce']
                               verbose=verbose,
                               )
        
        for force in forces_add:
            self.system.addForce(force)

        if verbose: print(f'added {len(forces_add)} forces to the system: {[x.getName() for x in forces_add]}')
        else: pass

    def corrections_to_ff_(self, verbos=True):
        self.recast_NB_(verbose=verbos)

class velff(tmFF, itp2FF):
    ''' keeping seperate to load pickled files '''
    def __init__(self,):
        super().__init__()
        self._FF_name_ = 'velff'
        #                           '; nbfunc        comb-rule       gen-pairs       fudgeLJ      fudgeQQ',
        self._FF_name_defaults_line_ = '1               1               yes             1.0          1.0  '
        self._system_name_ = 'veliparib'
        self._compound_name_ = 'vel'
    
    @classmethod
    @property
    def FF_name(self,):
        return 'velff'
  
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
