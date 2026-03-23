from .mm_helper import *

from .ff_setup import *
OPLS = OPLS_general

## ## ## ## ## ## ## ## ## ## ## ##

''' TODO: explain how to use in a notebook '''

class SingleComponent:
    def __new__(cls,
                PDB,
                n_atoms_mol,
                name,
                FF_name = 'GAFF',
                atom_order_PDB_match_itp = False, # depreceated
                FF_class = None,
                ):
        
        if FF_class is None: mixin = {'GAFF': GAFF, 'OPLS': OPLS, 'TIP4P':TIP4P}[FF_name]
        else:                mixin = FF_class

        cls.FF_name = mixin.FF_name

        cls = type(cls.__name__ + '+' + mixin.__name__, (cls, mixin), {})
        return super(SingleComponent, cls).__new__(cls)
    
    def __init__(self,
                 PDB : str,                        # PDB of a crystal, or one molecule in vacuum, including all hydrogens
                 n_atoms_mol : int,                # number of atoms in one molecule (including hydrogens)
                 name : str,                       # name of the molecule (i.e., name of the single component)
                 FF_name : str = 'GAFF',           # FF_name in ['GAFF', 'OPLS', 'TIP4P'], ! this will be ignored if FF_class not None.
                 atom_order_PDB_match_itp = False, # not used, assumed False and if True that is detected later
                 FF_class = None,                  # last minute update: a more general way of using any  simple class that has methods (initialise_FF_ and set_FF_)
                ):
        super(SingleComponent, self).__init__()
        '''
        folders necesary: (/name1/misc/, /name1/data/) for any FF, and /name1/temp/
        if using OPLS; run LigParGen on the single molecule first and put both outputs in /name1/misc/
        These two files need be named as: LigParGen_output_{name}.gro, LigParGen_output_{name}.itp
        ; where name = name above (usually name = name1)
        '''
        self.using_gmx_loader = True # would not work if False.
        self.verbose = True          # can be modified externally once SingleComponent initialised
        self.PDB = str(Path(PDB).absolute())
        self.PDB = DIR_main+'/MM/'+'/'.join(self.PDB.split('/')[-3:])
        assert self.PDB[-4:] == '.pdb'
        self.misc_dir =  Path('/'.join(self.PDB.split('/')[:-1]) + '/misc')
        self.n_atoms_mol = n_atoms_mol
        self.name = str(name)

        self.print('# initialise_object (SingleComponent) with '+self.FF_name+' FF, from the input file (PDB):\n', self.PDB)

        if self.FF_name == 'TIP4P': assert self.n_atoms_mol == 3
        else: pass
        #if self.FF_name == 'OPLS': self.atom_order_PDB_match_itp = atom_order_PDB_match_itp
        #else: 
        #    # if not self.atom_order_PDB_match_itp: self.print('! not implemented warnining : atom_order_PDB_match_itp parameter has no effect in this FF')
        #    # else: pass
        #    pass

        self.traj = mdtraj.load(self.PDB,self.PDB) # box info needed for mdtraj to read a file in this way
        self.r0 = self.traj.xyz[0]
        self.N = self.traj.n_atoms
        self.n_molecules = self.N // self.n_atoms_mol
        assert self.n_molecules == self.N / self.n_atoms_mol
        self.n_mol = self.n_molecules
        self.print('n_molecules:', self.n_molecules)

        try:
            self.b0 = self.traj.unitcell_vectors[0]
        except:
            if self.n_molecules == 1:
                print('!! even a single molecule needs a box in the input file (PDB)')
                print('(this can be any large box)')
            else:
                print('!! required box information missing in the input file (PDB)')
                print('(this must be a correct box)')

        if not box_in_reduced_form_(self.b0):
            self.print('! box not in reduced form. Writing a new input PDB file:')
            self.traj, self.PDB = change_box_(self.PDB, n_atoms_mol, make_orthorhombic=False)
            self.b0 = self.traj.unitcell_vectors[0]
            self.print('The input PDB file being used:',self.PDB)
        else: pass

        self.initialise_FF_()
        self.set_rdkit_mol_()
        self.set_arrays_blank_()

        self.args_initialise_object = {'PDB'         : self.PDB,
                                       'n_atoms_mol' : self.n_atoms_mol,
                                       'name'        : self.name,
                                       'FF_name'     : self.FF_name,
                                       'atom_order_PDB_match_itp' : atom_order_PDB_match_itp,
                                       'FF_class'    : FF_class, # will be used later.
                                       }
        self.print('')

    def set_rdkit_mol_(self):
        ' not used in this class, just to check atom indices '
        self.mol = Chem.MolFromPDBFile(str(self._single_mol_pdb_file_.absolute()),
                                       removeHs = False)
        for i, a in enumerate(self.mol.GetAtoms()): a.SetAtomMapNum(i)
        self.mol.RemoveConformer(0)

    def print(self, *text, verbose=False):
        if self.verbose: print(*text)
        elif verbose: print(*text)
        else: pass

    def initialise_system_(self,
                           PME_cutoff : float = None,
                           removeCMMotion : bool = True,
                           nonbondedMethod = app.PME,
                           custom_EwaldErrorTolerance = 0.0001, #1e-5 too slow in skewed cell
                           # 0.0005 default : fast
                           constraints = None, # can use # app.HBonds
                           SwitchingFunction_factor = 0.95,
                           ):
        self.print('# initialise_system:')

        self.args_initialise_system = {'PME_cutoff'                 : PME_cutoff,
                                       'removeCMMotion'             : removeCMMotion,
                                       'nonbondedMethod'            : nonbondedMethod,
                                       'custom_EwaldErrorTolerance' : custom_EwaldErrorTolerance,
                                       'constraints'                : constraints,
                                       'SwitchingFunction_factor'   : SwitchingFunction_factor,
                                       }

        ''' draw PME_cutoff in VMD:
        (VMD) 56 % draw color white
        0
        (VMD) 57 % draw sphere {0 0 0} radius *PME_cutoff* resolution 25
        1
        (VMD) 58 % draw material Transparent
        '''

        if PME_cutoff is None:
            PME_cutoff = np.diag(self.b0).min() * 0.5 * 0.95
        else: pass
        self.PME_cutoff = PME_cutoff
        if self.n_molecules == 1 and nonbondedMethod != app.NoCutoff:
            self.print('! only one molecule in the system, setting non-bonded method to app.NoCutoff')
            nonbondedMethod  = app.NoCutoff 
        else:
            self.print('set '+str(nonbondedMethod)+' cutoff to:',PME_cutoff,'nm')

        self.removeCMMotion = removeCMMotion 
        self.print('removeCMMotion active:', removeCMMotion is True)

        ### ### ### ###
        self.set_FF_() # creates self.ff
        ### ### ### ###

        if self.FF_name == 'GAFF' and not self.using_gmx_loader:
            print("!!! depreciated; GAFF in openmm using the 'amber loader' is not set up here")
            print('self.using_gmx_loader = True is required')
            ''' 
            method depreciated (NB: the box in prmtop file is abitrary, so would need to be inserted below).
            self.system = self.ff.createSystem( nonbondedMethod = nonbondedMethod,
                                                nonbondedCutoff = PME_cutoff * unit.nanometers,
                                                constraints     = constraints,
                                                removeCMMotion  = self.removeCMMotion)
            '''
        else:
            # self.ff already defined, but needs updating 
            abc_ABG = np.concatenate([self.traj.unitcell_lengths*10.0,self.traj.unitcell_angles],axis=1)[0]
            self.ff.box = abc_ABG + 1e-5 # > 1e-6  # .astype(np.float64) does not help
            # an error inside opnemm, because it can be shown that adding 1e-5 can have a huge effect on energy.
            # this seems to occur when initial box has angles 90 exactly
            # ^ this is TODO; report the issue using a simple minimal example.

            # top.topology.setPeriodicBoxVectors(self.b0) # not using this anymore because of above
            self.topology = self.ff.topology
            self.system = self.ff.createSystem( nonbondedMethod = nonbondedMethod,
                                                nonbondedCutoff = PME_cutoff * unit.nanometers,
                                                constraints     = constraints,
                                                removeCMMotion  = self.removeCMMotion)
            
            # self.system.setDefaultPeriodicBoxVectors(*self.b0)
            
        self.SwitchingFunction_factor = SwitchingFunction_factor
        if self.n_molecules > 1:
            self.turn_ON_nonbonded_SwitchingFunction(factor=self.SwitchingFunction_factor)
            self.print('set SwitchingFunction to',self.SwitchingFunction_factor,'* PME_cutoff =',
                        self.PME_cutoff*self.SwitchingFunction_factor, 'nm')
        else: pass
        self.custom_EwaldErrorTolerance = custom_EwaldErrorTolerance 
        self.adjust_EwaldErrorTolerance(self.custom_EwaldErrorTolerance, verbose = self.n_molecules>1 and self.verbose)

        ##
        self.define_mu_()
        ##
        
        self.n_constraints = self.system.getNumConstraints()
        if self.removeCMMotion: self.n_constraints += 3
        else: pass
        self.n_DOF = self.system.getNumParticles()*3 - self.n_constraints
        if constraints is None:
            if self.removeCMMotion: assert self.n_DOF == 3*(self.N-1)
            else:                   assert self.n_DOF == 3*self.N
        else: pass

        self.corrections_to_ff_(self.verbose)

        self.print('n_mol = ', str(self.n_mol)+',',
                   'n_atoms_mol =', str(self.n_atoms_mol)+',',
                   'N =', str(self.N)+',',
                   'n_DOF =', str(self.n_DOF),
                   '(n_constraints =', str(self.n_constraints)+')',
                   '\n'
                   )

    def initialise_simulation_(self,
                               rbv = None,
                               minimise = True,
                               #
                               T = 300.0,
                               timestep_ps = 0.0005,
                               collision_rate = 1,
                               #
                               P = None,
                               barostat_type = 2,
                               barostat_1_scaling = [True,True,True],
                               stride_barostat = 25,
                               custom_integrator = None, # lambda function with other parameters specified
                              ):
        self.args_initialise_simulation = { 'rbv'                : rbv,
                                            'minimise'           : minimise,
                                            'T'                  : T,
                                            'timestep_ps'        : timestep_ps,
                                            'collision_rate'     : collision_rate,
                                            'P'                  : P,
                                            'barostat_type'      : barostat_type,
                                            'barostat_1_scaling' : barostat_1_scaling,
                                            'stride_barostat'    : stride_barostat,
                                            'custom_integrator'  : custom_integrator,
                                        }
        self.print('# initialise_simulation:')

        self.T = T
        self.kT = CONST_kB * self.T  
        self.beta = 1.0 / self.kT
        self.print('set temperature:',self.T,'Kelvin')
        self.timestep_ps = timestep_ps
        self.print('set integration timestep:',self.timestep_ps,'ps')
        self.collision_rate = collision_rate
        self.print('set collision rate (friction ceofficent):',self.collision_rate,'/ps')

        if custom_integrator is not None:
            integrator = custom_integrator(self.T*unit.kelvin,
                                           1.0/unit.picosecond,
                                           self.timestep_ps*unit.picoseconds)
        else:
            integrator = mm.LangevinMiddleIntegrator(self.T*unit.kelvin,
                                                    1.0/unit.picosecond,
                                                    self.timestep_ps*unit.picoseconds)

        self.P = P ; self.barostat_type = barostat_type ; self.barostat_1_scaling = barostat_1_scaling ; self.stride_barostat = stride_barostat
        
        self._remove_barostat_from_system_()
        if self.P is not None: self._add_barostat_to_system_()
        else: pass

        self.simulation = app.Simulation(self.topology, self.system, integrator)

        if rbv is None:
            #self.system.setDefaultPeriodicBoxVectors(*self.b0)
            #self.simulation.system.setDefaultPeriodicBoxVectors(*self.b0)

            self._set_b_(self.b0)
            self._set_r_(self.r0)
            if custom_integrator is not None:
                v0 = np.random.randn(self.N,3) * ((self._mass_*self.beta)**(-0.5))
                self._set_v_(v0)
        else:
            r,b,v = rbv
            #self.system.setDefaultPeriodicBoxVectors(*self.b0)
            #self.simulation.system.setDefaultPeriodicBoxVectors(*b)

            self._set_b_(b)
            self._set_r_(r)
            if v is not None: self._set_v_(v)
            else: pass

        if minimise:
            self.print('minimise = True, minimising potential energy (u):')
            self.minimise_()
        else: pass

        if self.removeCMMotion is True:
            self._set_r_(self._current_r_ - self._current_COM_)
        else: pass

        if hasattr(self, 'u_'): pass
        else: self.u_ = self.u_GPU_

        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        self.print('')

    def save_simulation_data_(self, path_and_name:str):
        '''
        simulation timescale = len(u) * stride_save_frame * timestep_ps
        '''
        # print(f'Saving MD data. This took {self.time_elapsed} seconds to sample.')
        dataset = {'xyz':self.xyz,
                   'COMs':self.COMs,
                   'b':self.boxes,
                   'u':self.u,
                   'T':self.temperature,
                   'rbv':[self._current_r_, self._current_b_, self._current_v_], # to resume
                   'stride_save_frame':self.stride_save_frame,
                   'time_elapsed':self.time_elapsed,
                   }
        assert hasattr(self, 'simulation')
        simulation_data = {'MD dataset':dataset,
                           'args_initialise_object': self.args_initialise_object,
                           'args_initialise_system': self.args_initialise_system,
                           'args_initialise_simulation': self.args_initialise_simulation,
                          }
        save_pickle_(simulation_data, path_and_name)

    @staticmethod
    def load_simulation_data_(path_and_name):
        return load_pickle_(path_and_name)

    @staticmethod
    def initialise_from_save_(path_and_name:str, resume_simulation=True, verbose=True):

        if type(path_and_name) is str:
            dataset = SingleComponent.load_simulation_data_(path_and_name=path_and_name)
            print(f'initialising from saved dataset: {path_and_name}')
        else:
            dataset = path_and_name

        sc = SingleComponent(**dataset['args_initialise_object'])
        sc.verbose = verbose
        sc.initialise_system_(**dataset['args_initialise_system'])
        if resume_simulation:
            dataset['args_initialise_simulation']['rbv'] = dataset['MD dataset']['rbv']
        else: pass
        sc.initialise_simulation_(**dataset['args_initialise_simulation'])
        sc._xyz = [x for x in dataset['MD dataset']['xyz']]
        sc._COMs = [x for x in dataset['MD dataset']['COMs']]
        sc._boxes = [x for x in dataset['MD dataset']['b']]
        sc._u = [x for x in dataset['MD dataset']['u'][:,0]]
        sc._temperature = [x for x in dataset['MD dataset']['T']]
        sc.n_frames_saved = len(dataset['MD dataset']['u'])
        sc.stride_save_frame = dataset['MD dataset']['stride_save_frame']
        return sc

def concatenate_datasets_(paths_datasets : list, remove_warmup=None, stride=1):
    dataset_concat = {}
    a = 0
    keys = ['xyz', 'COMs', 'b', 'u', 'T']
    for path in paths_datasets:
        dataset = load_pickle_(path)
        if a > 0:
            assert dataset['args_initialise_object'] == dataset_concat['args_initialise_object']
            assert dataset['args_initialise_system'] == dataset_concat['args_initialise_system']
            assert dataset['args_initialise_simulation'] == dataset_concat['args_initialise_simulation']
            assert dataset['MD dataset']['stride_save_frame'] == dataset_concat['MD dataset']['stride_save_frame']

            for key in keys:
                dataset_concat['MD dataset'][key] = np.concatenate([dataset_concat['MD dataset'][key],
                                                                    dataset['MD dataset'][key][remove_warmup:][::stride],
                                                                   ], axis=0)
        else:
            dataset_concat['args_initialise_object'] = dataset['args_initialise_object']
            dataset_concat['args_initialise_system'] = dataset['args_initialise_system']
            dataset_concat['args_initialise_simulation'] = dataset['args_initialise_simulation']
            dataset_concat['MD dataset'] = {'xyz':dataset['MD dataset']['xyz'][remove_warmup:][::stride],
                                            'COMs':dataset['MD dataset']['COMs'][remove_warmup:][::stride],
                                            'b':dataset['MD dataset']['b'][remove_warmup:][::stride],
                                            'u':dataset['MD dataset']['u'][remove_warmup:][::stride],
                                            'T':dataset['MD dataset']['T'][remove_warmup:][::stride],
                                            'rbv':dataset['MD dataset']['rbv'],
                                            'stride_save_frame':dataset['MD dataset']['stride_save_frame'],
                                            }
        a += 1
    return dataset_concat

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## ADD openmm FFs below ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

class LJ(MM_system_helper):

    def __init__(self,):
        super().__init__()

    @classmethod
    @property
    def FF_name(self,):
        return 'LJ'
    
    @property
    def _single_mol_pdb_file_(self) -> Path:
        return self.misc_dir / f"{self.name}_single_mol.pdb" 

    def initialise_FF_(self,):
        assert self.n_mol == self.N
        assert self.n_atoms_mol == 1

        if os.path.exists(self._single_mol_pdb_file_.absolute()): pass
        else: process_mercury_output_(self.PDB, self.n_atoms_mol, single_mol = True, custom_path_name=str(self._single_mol_pdb_file_.absolute()),
                                     ) # make *single_mol.pdb
            
    @property
    def top_file_gmx_crys(self) -> Path:
        return self.misc_dir / f"x_x_{self.name}_gmx.top"

    def reset_n_mol_top_(self,):

        lines_top = [f'#include "{self.name}.itp"',
                     '',
                     '[system]',
                     'LJ',
                     '',
                     '[molecules]',
                     f'{self.name}  ' + str(self.n_mol),
                     ]
        file_top = open(str(self.top_file_gmx_crys.absolute()), 'w')
        for line in lines_top:
            file_top.write(line + "\n")
        file_top.close()

    def set_FF_(self,):

        self.reset_n_mol_top_()
        self.ff = parmed.gromacs.GromacsTopologyFile(str(self.top_file_gmx_crys.absolute()))

##

class TIP4P(MM_system_helper):
    ''' mixin class for SingleComponent. Methods relevant only for using TIP4P are here.
    '''

    def __init__(self,):
        super().__init__()

    @classmethod
    @property
    def FF_name(self,):
        return 'TIP4P'
    
    @property
    def _single_mol_pdb_file_(self) -> Path:
        return self.misc_dir / f"{self.name}_single_mol.pdb" 

    def initialise_FF_(self):
        '''
        run this after (n_mol and n_atoms_mol) defined in __init__ of SingleComponent
        '''
        if os.path.exists(self._single_mol_pdb_file_.absolute()):
            pass
        else:
            # just for self.mol (rdkit just for consistency)
            assert self.n_atoms_mol == 3
            process_mercury_output_(self.PDB,
                                    self.n_atoms_mol,
                                    single_mol = True,
                                    custom_path_name=str(self._single_mol_pdb_file_.absolute()),
                                    ) # make *single_mol.pdb

    @property
    def top_file_gmx_crys(self) -> Path:
        return self.misc_dir / f"x_x_{self.name}_gmx.top"

    def reset_n_mol_top_(self,):

        lines_top = ['#include "forcefield.itp"',
                     '#include "tip4p-ice.itp"',
                     '',
                     '[system]',
                     'ICE',
                     '',
                     '[molecules]',
                     'water  ' + str(self.n_mol),
                     ]
        file_top = open(str(self.top_file_gmx_crys.absolute()), 'w')
        for line in lines_top:
            file_top.write(line + "\n")
        file_top.close()

    def set_FF_(self,):

        self.reset_n_mol_top_()
        self.ff = parmed.gromacs.GromacsTopologyFile(str(self.top_file_gmx_crys.absolute()))

    ## ##

    def add_v_sites_(self, r):

        if len(r.shape) == 2: r = r[np.newaxis,...] ; remove_batch_axis = True
        else: remove_batch_axis = False

        r = reshape_to_molecules_np_(r, n_atoms_in_molecule=3, n_molecules=self.n_mol)
        O  = r[...,0,:]
        H1 = r[...,1,:]
        H2 = r[...,2,:]
        M  = O + 0.13458335*(H1 + H2 - O - O)
        r  = np.stack([O,H1,H2,M], axis=-2)
        r  = reshape_to_atoms_np_(r, n_atoms_in_molecule=4, n_molecules=self.n_mol)
        
        if remove_batch_axis: return r[0]
        else: return r

    def remove_v_sites_(self, r):

        if len(r.shape) == 2: r = r[np.newaxis,...] ; remove_batch_axis = True
        else: remove_batch_axis = False

        r = reshape_to_molecules_np_(r, n_atoms_in_molecule=4, n_molecules=self.n_mol)
        r  = r[...,:3,:]
        r  = reshape_to_atoms_np_(r, n_atoms_in_molecule=3, n_molecules=self.n_mol)

        if remove_batch_axis: return r[0]
        else: return r

    '''
    The following methods override some of the methods in mm_helper.
    '''

    @property
    def _current_r_(self,):
        # positions
        # unit = nanometer
        return self.remove_v_sites_(self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value)

    @property
    def _current_v_(self,):
        # velocities
        # unit = nanometer/picosecond
        return self.remove_v_sites_(self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)._value)
    
    @property
    def _current_F_(self,):
        # forces
        # unit = kilojoule/(nanometer*mole) ; [F = -∇U]
        return self.remove_v_sites_(self.simulation.context.getState(getForces=True).getForces(asNumpy=True)._value)

    def _set_r_(self, r):
        assert r.shape == (self.N,3)
        self.simulation.context.setPositions(self.add_v_sites_(r))

    def _set_v_(self, v):
        assert v.shape == (self.N,3)
        self.simulation.context.setVelocities(self.add_v_sites_(v))

    def forward_atom_index_(self, inds):
        forward_mask = np.concatenate([np.arange(3) + i*4 for i in range(self.n_mol)],axis=0)
        return forward_mask[inds]
    
    @property
    def _system_mass_(self,):
        mass_with_v_sites =  np.array([self.system.getParticleMass(i)._value for i in range(self.system.getNumParticles())])
        return mass_with_v_sites[self.forward_atom_index_(np.arange(self.N))]

    def inverse_atom_index_(self, inds):
        print('!! inverse_atom_index_ : undefined')

##

class GAFF(MM_system_helper):
    ''' mixin class for SingleComponent. Methods relevant only for using GAFF are here.

    attribes needed before running self.initialise_FF_

    self.misc_dir
    self.name
    .. add

    '''
    def __init__(self,):
        super().__init__()

    @classmethod
    @property
    def FF_name(self,):
        return 'GAFF'
    
    def initialise_FF_(self,):
        '''
        run this after (n_mol and n_atoms_mol) defined near the end of __init__ of SingleComponent
        this is at the end of __init__
        '''

        if self.can_run_tleap: pass
        else: 
            print('generating tleap inputs for this molecule:')
            self.first_time_molecule_()
            assert self.can_run_tleap
        # if later using set_FF_amber : needs to be ran (to set n_mol in prmtop)
        # if later using set_FF_gmx   : running again in case set_FF_amber was ran earlier (n_mol in prmtop should be 1)
        self.create_prmtop_()

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    # first time molecule:

    @property
    def _single_mol_pdb_file_(self) -> Path:
        return self.misc_dir / f"{self.name}_single_mol.pdb" 
    @property
    def _single_mol_prepi_file_(self) -> Path:
        return self.misc_dir / f"{self.name}_single_mol.prepi"
    @property
    def _single_mol_frcmod_file_(self) -> Path:
        return self.misc_dir / f"{self.name}_single_mol.frcmod"
    @property
    def _single_mol_mol2_file_(self) -> Path:
        return self.misc_dir / f"{self.name}_single_mol.mol2"

    def first_time_molecule_(self,):
        # make *single_mol.pdb
        # first single-molecule conformer from crystal PDB is extracted and saved as *single_mol.pdb :
        process_mercury_output_(self.PDB,
                                self.n_atoms_mol,
                                single_mol = True,
                                custom_path_name=str(self._single_mol_pdb_file_.absolute()),
                                )
        
        '''
        # Prompt: "should antechamber only be ran on energy minimised molecules?"

        # NB: this question is also relevant for LigParGen input (*single_mol.pdb)

        # Response from Gemini 2.0 Flash:
        "
        When You Might Skip Energy Minimization (with Caution):

        Already Reasonable Geometry: If you have a high-quality 3D structure 
        obtained from reliable experimental data (e.g., X-ray crystallography) 
        or a thorough quantum chemical optimization, the geometry might already 
        be sufficiently good for Antechamber. However, it's still generally a 
        good practice to perform a quick minimization to ensure no severe clashes 
        or issues exist.

        Specific Workflows: In some automated workflows where many molecules 
        are processed, a pre-minimization step might be skipped for speed. 
        However, this increases the risk of errors and should be done with 
        careful consideration and potentially followed by quality checks.
        "

        # first point true when using SingleComponent for crystal (on Mercury(Form.cif).pdb)
        # however, small differences in energy were seen later depending on the Form

        # second point true when using SingleComponent for single molecule because 
        # SMILES processed by rdkit a priori to make a reasonable conformer.pdb

        '''

        # make *.prepi
        subprocess.run(     ["antechamber", "-i", 
                             self._single_mol_pdb_file_.absolute(), 
                             "-fi", "pdb", "-o",
                             self._single_mol_prepi_file_.absolute(),
                             "-fo", "prepi", "-c", "bcc", "-pf","y",]
                        ) # check bcc vs other options, or use HF
        # make *.mol
        subprocess.run(     ["antechamber", "-i", 
                             self._single_mol_pdb_file_.absolute(), 
                             "-fi", "pdb", "-o",
                             self._single_mol_mol2_file_.absolute(),
                             "-fo", "mol2", "-c", "bcc", "-pf","y",]
                        )
        # make *.fcmod
        subprocess.run(     ["parmchk2", "-f", "prepi", "-i",
                             self._single_mol_prepi_file_.absolute(), 
                             "-o",
                             self._single_mol_frcmod_file_.absolute(),]
                        )

    @property
    def can_run_tleap(self,):
        paths_needed = [self._single_mol_prepi_file_.absolute(),
                        self._single_mol_frcmod_file_.absolute(),
                        self._single_mol_mol2_file_.absolute(),
                       ]
        return all([os.path.exists(x) for x in paths_needed])
    
    ##

    @property
    def tleap_file(self) -> Path:
        return self.misc_dir / f"{self.name}.tleap"
    @property
    def prmtop_file(self) -> Path: # the main file of interest
        return self.misc_dir / f"{self.name}.prmtop"
    @property
    def _inpcrd_file_(self) -> Path:
        return self.misc_dir / f"{self.name}.inpcrd"
    
    def create_prmtop_(self,):
        if self.using_gmx_loader: n = 1
        else: n = self.n_molecules
        # 
        tleap_file = """
        source leaprc.gaff
        loadamberprep {prepi_file}
        loadamberparams {frcmod_file}
        mol = loadmol2 {mol2_file}
        aggregate = combine {copies_of_molecule}
        set aggregate box {any_cell}
        saveamberparm aggregate {top_file} {inpcrd_file}
        quit
        """.format(
            prepi_file=self._single_mol_prepi_file_.absolute(),
            frcmod_file=self._single_mol_frcmod_file_.absolute(),
            mol2_file=self._single_mol_mol2_file_.absolute(),
            copies_of_molecule ='{'+' '.join(['mol']*n)+'}',
            any_cell = '{'+' '.join(['100.0']*3)+'}',
            top_file=self.prmtop_file.absolute(),
            inpcrd_file=self._inpcrd_file_.absolute(),
        )
        self.tleap_file.write_text(tleap_file)
        subprocess.run(["tleap", "-s", "-f",
                        self.tleap_file.absolute()],
                        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        )
        self.prmtop = str(self.prmtop_file.absolute())


    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    # adjust top file:

    @property
    def top_file_gmx(self) -> Path:
        return self.misc_dir / f"{self.name+'_gmx'}.top"
    
    @property
    def top_file_gmx_adjusted_charges(self) -> Path:
        return self.misc_dir / f"x_{self.name}_gmx.top"

    @property
    def top_file_gmx_crys(self) -> Path:
        return self.misc_dir / f"x_x_{self.name}_gmx.top"

    def reset_n_mol_top_(self,):
        change_n_mol_top_(path_top_file_in = str(self.top_file_gmx_adjusted_charges.absolute()),
                          path_top_file_out = str(self.top_file_gmx_crys.absolute()),
                          replace_n_mol = self.n_mol,
                          verbose = self.verbose,
                         )
        
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    # set_FF (GAFF):

    def set_FF_amber_(self,):
        print('!!! using amber loader for GAFF; this may have problems')
        self.ff = app.AmberPrmtopFile(self.prmtop)
        
    def set_FF_gmx_(self,):
        
        if os.path.exists(self.top_file_gmx.absolute()): pass
        else:
            # attaching the relevant information for the amber FF
            gaff = parmed.load_file(str(self.prmtop_file.absolute()), str(self._single_mol_pdb_file_.absolute()))
            gaff.save(str(self.top_file_gmx.absolute()))

        if os.path.exists(self.top_file_gmx_adjusted_charges.absolute()): pass
        else:
            # self.top_file_gmx_adjusted_charges file for a single molecule ; this was used by default (neutralised charge)
            change_charges_itp_top_(path_top_or_itp_file_in = str(self.top_file_gmx.absolute()),
                                    path_top_or_itp_file_out = str(self.top_file_gmx_adjusted_charges.absolute()),
                                    n_atoms_mol= self.n_atoms_mol,
                                    replacement_charges = None,
                                    neutralise_charge = True,
                                    )
            
        # self.n_mol needs to be defined before this step:
        self.reset_n_mol_top_() # this step creates self.top_file_gmx_crys file with currect number of molecules 

        # the self.top_file_gmx_crys file is then used to initialise the relevant FF:
        # if n_mol = 1 this file also used because it has other adjustments like charges
        self.ff = parmed.gromacs.GromacsTopologyFile(str(self.top_file_gmx_crys.absolute()))

    def set_FF_(self,):
        if self.using_gmx_loader: self.set_FF_gmx_()
        else: self.set_FF_amber_()


