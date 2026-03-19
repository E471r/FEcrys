''' ecm_basic.py

ECM:
    f : FE_lambda0_
    c : LAMBDA_0
    c : LAMBDA_SYSTEM
    c : ECM_basic

simulation parameters for specific systems:
    f : succinic_acid_ARGS_oss
    f : veliparib_ARGS_oss
    f : mivebresib_ARGS_oss

'''

from .sc_system import *
from pymbar import MBAR

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

def FE_lambda0_(n_bodies, N, V, T, k, mu):
    ''' REF: https://doi.org/10.1063/5.0044833

    Inputs:
        n_bodies : number of molecules in supercell 
        N        : number of atoms in supercell (NB: virtual atoms are not atoms)
        V        : volume of the supercell
        T        : temperature of the canonical ensemble 
        k        : scalar string constant of the harmonic potential (Einstein crystal)
        mu       : two options:
            COM-free simulations         : normalised masses of the atoms := m_{i} / ( \sum_{j}^{N} m_{j} ) ; m_i = mass of i'th atom
            simulations with fixed atoms : any 'one-hot' vector, can be np.array([1])

    Output:
        f_0 = f_C_minus_f_C_CM + f_EC_CM_minus_f_EC + f_EC

            f_EC               : FE of Einstein crystal (EC)
            f_EC_CM_minus_f_EC : FE difference associated with removing COM from the EC
            f_C_minus_f_C_CM   : FE difference associated with removing COM unperturbed crystal (C)
                COM = centres of mass, or one atom

            f_C = (f_C - f_C_CM) + [f_C_CM - f_EC_CM] + (f_EC_CM - f_EC) + f_EC
                = f_0 + [f_C_CM - f_EC_CM] ; [...] via FEP
    '''

    mu = np.array(mu).flatten()
    mu /= mu.sum()

    CONST_kB =  1e-3*8.31446261815324
    kT = CONST_kB *T
    beta = 1.0 / kT
       
    f_EC = - 0.5*3*N*np.log((2.0*np.pi)/(beta*k))

    f_EC_CM_minus_f_EC = - 0.5*3*np.log((beta*k)/(2.0*np.pi*np.sum(mu**2)))
    
    f_C_minus_f_C_CM = - np.log(V/n_bodies) # not sure for fixed atom

    contributions = np.array([f_EC, f_EC_CM_minus_f_EC, f_C_minus_f_C_CM])
    f_0 = contributions.sum()

    return f_0, contributions # units : kT

class LAMBDA_0:
    def __init__(self,
                 n_bodies : int,
                 mu : np.ndarray,         # (N,)
                 V : float,
                 T : float,
                 k : float,
                 R0 : np.ndarray,         # (N,3)
                 inds_valid : np.ndarray, # (N,)
                 n_atoms_in_molecule : int,
                 ):
        '''
        Inputs:
            (n_bodies, N, V, T, k, mu) : same as in FE_lambda0_ above
            R0 : centroid of the Einestein crystal state, not used here 
            inds_valid : was only relevant when v-sites are present (TIP4P in the old version)
            n_atoms_in_molecule : number of atoms in a single molecule
        '''
        self.n_bodies = n_bodies
        self.V = V
        self.T = T ; self.kT = CONST_kB * self.T ; self.beta = 1.0 /self.kT
        self.k = k
        self.mu = mu
        self.inds_valid = inds_valid
        self.N = len(self.inds_valid)
        self.n_atoms_in_molecule = n_atoms_in_molecule
        assert self.N == self.n_atoms_in_molecule * self.n_bodies
        self.R0 = R0[self.inds_valid][np.newaxis,...] # (1,N,3)
        f0, f0_parts = FE_lambda0_(n_bodies=self.n_bodies, N=self.N, V=self.V, T=self.T, k=self.k, mu=self.mu)
        self.f0 = f0
        self.f0_parts = f0_parts

        #self.sigma = (self.beta*self.k)**(-0.5)

    #def sample0_(self, n_samples):
    #    # can be used but was not used
    #    return np.random.randn(n_samples, self.N, 3) * self.sigma + self.R0 # (m,n,3)
    
    def energy0_(self, r):
        '''
        also was not used, but can compare with openMM (self.lambda_systems[0.0].u_ should evalaute to the same values as this function;
                                                        the input (r) should have the relevant COM already removed if checking this)
        '''
        if len(r.shape) == 2: r = r[np.newaxis,...]
        else: pass
        return 0.5*self.k*((r[:,self.inds_valid,:] - self.R0)**2).sum(-1).sum(-1, keepdims=True) * self.beta # (m,1)

class LAMBDA_SYSTEM:
    def __init__(self,

                 args_initialise_object,
                 args_initialise_system,
                 args_initialise_simulation,

                 COM_removal_by_fixing_one_atom_index_of_this_atom : int = None,

                 lam = 1.,
                 k_EC = 6000., # ok.
                 stride_save_frame = 50,
                 remove_warmup = 200,
                 ):

        args_initialise_simulation['minimise'] = False
            
        self.args_initialise_object = args_initialise_object
        self.args_initialise_system = args_initialise_system
        self.args_initialise_simulation = args_initialise_simulation
        lam = float(lam)
        assert 0.0 <= lam <=1.0
        self.lam = lam
        self.k_EC = k_EC
        self.stride_save_frame = stride_save_frame
        self.remove_warmup = remove_warmup
        self.warmup_removed = False
        ##

        if COM_removal_by_fixing_one_atom_index_of_this_atom is not None:
            self.args_initialise_system['removeCMMotion'] = False
            self.args_initialise_simulation['rbv'] = None
            self.ind_atom_fixed = COM_removal_by_fixing_one_atom_index_of_this_atom
            if type(self.ind_atom_fixed) is int:
                self.fixed_atom = True
            else:
                self.fixed_atom = False
        else:
            self.args_initialise_system['removeCMMotion'] = True
            self.ind_atom_fixed = None
            self.fixed_atom = False

        self.sc = SingleComponent(**self.args_initialise_object)
        self.sc.initialise_system_(**self.args_initialise_system)
        self.sc.initialise_simulation_(**self.args_initialise_simulation)
        
        # ! 
        #if self.ind_atom_fixed is not None: self.ind_atom_fixed = int(self.sc.forward_atom_index_(self.ind_atom_fixed))
        #else: pass

        if self.fixed_atom:
            self.inds_true_atoms = np.array(list(set(np.arange(self.sc.N).tolist()) - set([self.ind_atom_fixed])))
            self.sc.system.setParticleMass(self.ind_atom_fixed,0.0)
        else:
            self.inds_true_atoms = np.arange(self.sc.N)

        '''
            if self.ind_atom_fixed is None : centre of mass removed from all atoms (self.sc._mu_ defined by self.sc.define_mu_ is not sparse)
            if self.ind_atom_fixed is an integer : atom with this index in the first molecule is fixed (first this defines self.sc._mu_ to be a one hot vector)
        '''
        self.sc.define_mu_(self.ind_atom_fixed)
        self.mu = np.array(self.sc._mu_[np.newaxis,...]) # (1,N,1)
        ' minimise the system first (starts with args_initialise_object[PDB] and will minimise the same way every time) '
        # self.sc.minimise_() ! maybe generally not good
        ' recenter the system using self.sc._mu_ defined earlier '
        self.sc._recenter_simulation_()
        ' this is now the centroid of the EC system with COM missing (the same no matter which lambda state this is)'
        self.r0 = np.array(self.sc._current_r_)
        self.sc.r0 = self.r0

        if COM_removal_by_fixing_one_atom_index_of_this_atom is not None and not self.fixed_atom:
            custom_integrator = lambda T, f, dt : CustomIntegrator_(T, f, dt, [self.sc._mu_, self.sc._mass_])
        else:
            custom_integrator = None

        if lam < 1.0:
            ' system modified by adding the harmonic EC potential scaled by (1-lam), and scaling the physical system by lam'
            put_lambda_into_system_(self.sc.system, 
                                    lam = lam,
                                    R0 = self.r0,
                                    k_EC = self.k_EC,
                                    inds_true_atoms = self.inds_true_atoms,
                                    verbose = False,
                                    )
        else: pass

        self.sc.initialise_simulation_(**self.args_initialise_simulation, custom_integrator=custom_integrator)
        self.sc.n_DOF = self.sc.N*3 - 3 # for temperature

        self.u_ = self.sc.u_GPU_

        if self.lam == 0.0:
            self.lambda_0 = LAMBDA_0(n_bodies = self.sc.n_mol,
                                     mu = self.sc._mu_, # (N,1)
                                     inds_valid = np.arange(self.sc.N),
                                     V = self.sc._current_V_,
                                     T = self.sc.T,
                                     k = self.k_EC,
                                     R0 = self.r0,
                                     n_atoms_in_molecule = self.sc.n_atoms_mol,
                                    )
                                     #fixed_atom = self.fixed_atom)
        else: pass
        # assert LAMBDA_SYSTEM(lam=0).__call__(r) == LAMBDA_SYSTEM(lam=0).lambda_0.energy0_(r) agree for any r.
        
        self.sc.minimise_()
        self.sc._recenter_simulation_()
                     
        print('ready to run_simulation_(n_saves) at lam =',
                '{:5.5s}'.format(str(self.lam)),'T =',
                self.sc.T,'K, of',
                self.sc.N,'atoms.',
            )

    @property
    def plot_check_harmonic(self):
            k = float(self.k_EC)
            PME_cutoff = float(self.args_initialise_system['PME_cutoff'])
            gaus_ = lambda x : np.exp(-0.5*k*(x**2))
            x = np.linspace(-PME_cutoff, PME_cutoff, 500)
            plt.plot(x, gaus_(x))
            plt.plot([-PME_cutoff]*2,[0,1], color='black')
            plt.plot([PME_cutoff]*2,[0,1], color='black')

    def reinitialise_simulation_(self,):
        self.sc.initialise_simulation_(**self.args_initialise_simulation)
        self.sc.simulation.context.setPositions(self.r0)

    @property
    def simulation_timescale(self,):
        'not used here'
        _time = self.stride_save_frame*(self.sc.n_frames_saved-1)*self.timestep_ps/1000.
        print(_time, 'ns')
        return _time

    def set_arrays_blank_(self,):
        self.sc.set_arrays_blank_()

    def run_simulation_(self, n_saves, verbose_info : str = ''):
        
        if self.fixed_atom: pass 
        else: self.sc._recenter_simulation_()

        if not self.warmup_removed and self.remove_warmup > 0:
            #self.sc._recenter_simulation_()
            self.sc.run_simulation_(self.remove_warmup, self.stride_save_frame,
                                    verbose_info = verbose_info+'_equilibration')
            if self.fixed_atom: pass 
            else: self.sc._recenter_simulation_()
            self.set_arrays_blank_()
            self.warmup_removed = True
        else: pass

        self.sc.run_simulation_(n_saves, self.stride_save_frame,
                                verbose_info = verbose_info+'                  ')

        '''
        if self.lam < 1.0:
            for i in range(n_saves):
                self.sc.run_simulation_(1, self.stride_save_frame)
                self.sc._recenter_simulation_() 
        else:
            self.sc.run_simulation_(n_saves, self.stride_save_frame)
        '''

    @property
    def xyz(self,):
        return self.sc.xyz

    @property
    def u(self,):
        return self.sc.u

    @property
    def temperature(self,):
        return self.sc.temperature[:,np.newaxis]

##

class ECM_basic:
    def __init__(self,
                 name,
                 working_dir_folder_name : str, # 
                 ARGS_oss : list,
                 k_EC = 6000.,
                 COM_removal_by_fixing_one_atom_index_of_this_atom = None,
                 overwrite = False,
                 path_lambda_1_dataset = None
                ):
        self.generic_name = name+'_'
        self.working_dir_folder_name = working_dir_folder_name
        self.ARGS_oss = ARGS_oss
        self.k_EC = k_EC

        self.ind_atom_fixed = COM_removal_by_fixing_one_atom_index_of_this_atom 
        if type(self.ind_atom_fixed) is int:
            self.COM_remover_ = self.atom_based_COM_remover_
            self.fixed_atom = True
        else:
            self.COM_remover_ = self.global_COM_remover_
            self.fixed_atom = False

        self._lambdas = []
        self.lambda_systems = {}
        self.lambda_coordinates = {}
        self.simulation_info_u = {} 
        self.simulation_info_T = {}
        self.lambda_colors = {}
        self.lambda_Deltaf = {}
        self.lambda_dDeltaf = {}
        self.Ns_used_for_BAR = {}

        ##
        self.lambda_evaluations = {} # ij for BAR
        self.log_BAR_results = []

        # for unsupervised
        self.lambda_pairs_SE_BAR = {}  ; self.SE_BAR = 100.
        self.lambda_pairs_SE_mBAR = {} ; self.SE_mBAR = 100.
        self.lambda_dataset_converged = {}
        #

        self.path_lambda_1_dataset = path_lambda_1_dataset
        if not overwrite:
            ''' # if RAM overflow 
            if self.path_lambda_1_dataset is not None:
                self.add_lambda_(1.0)
                self.load_lambda1_dataset_seperately_(60000)
            else: pass
            '''
            self.import_data_(self.which_lambdas_exist_in_folder)

            try: self.import_lambda_evaluations_()
            except: print('no previously saved lambda_evaluations were found')
            try:  self.import_BAR_results_()
            except: print('no previously saved BAR_results were found')
            try:  self.import_mBAR_results_()
            except: print('no previously saved mBAR_results were found')
        else: 
            print('ECM : note; previous files with same name will be overwritten')
            
    ##

    def save_lambda_evaluations_(self,):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        save_pickle_(self.lambda_evaluations, self.working_dir_folder_name+'/'+x+'_lambda_evaluations')
 
    def import_lambda_evaluations_(self,):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        self.lambda_evaluations = load_pickle_(self.working_dir_folder_name+'/'+x+'_lambda_evaluations')
        
    def save_BAR_results_(self,):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        save_pickle_(self.log_BAR_results, self.working_dir_folder_name+'/'+x+'_log_of_BAR_results')

    def import_BAR_results_(self,):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        self.log_BAR_results = load_pickle_(self.working_dir_folder_name+'/'+x+'_log_of_BAR_results')

    def save_mBAR_results_(self,):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        save_pickle_(self.log_mBAR_results, self.working_dir_folder_name+'/'+x+'_log_of_mBAR_results')

    def import_mBAR_results_(self,):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        self.log_mBAR_results = load_pickle_(self.working_dir_folder_name+'/'+x+'_log_of_mBAR_results')
         
    def save_usupervised_sample_sizes_(self,):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        save_pickle_(self.lambda_sample_sizes, self.working_dir_folder_name+'/'+x+'_usupervised_sample_sizes')

    def import_usupervised_sample_sizes_(self,):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        self.lambda_sample_sizes = load_pickle_(self.working_dir_folder_name+'/'+x+'_usupervised_sample_sizes')
         
    def save_inds_rand_lambda1_(self, m):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        save_pickle_(self.inds_rand_lambda1, self.working_dir_folder_name+'/'+x+'_inds_rand_lambda1_'+str(m))

    def import_inds_rand_lambda1_(self, m):
        x = self.generic_name[:3]+'_'+self.generic_name[3:]
        self.inds_rand_lambda1 = load_pickle_(self.working_dir_folder_name+'/'+x+'_inds_rand_lambda1_'+str(m))
         
    ##

    @property
    def lambdas(self,):
        return np.sort(self._lambdas).tolist()
    
    @property
    def n_lambdas(self,):
        return len(self._lambdas)
    
    def lambda_exists_(self, lam):
        return lam in self._lambdas

    def add_lambda_(self, lam, verbose=True):
        lam = float(lam)
        if lam not in self._lambdas:
            self.lambda_systems[lam] = LAMBDA_SYSTEM(
                                                    args_initialise_object = self.ARGS_oss[0],
                                                    args_initialise_system = self.ARGS_oss[1],
                                                    args_initialise_simulation = self.ARGS_oss[2],
                                                    lam = lam,
                                                    k_EC = self.k_EC,
                                                    stride_save_frame = 0.1 / self.ARGS_oss[2]['timestep_ps'],
                                                    remove_warmup = 200,
                                                    COM_removal_by_fixing_one_atom_index_of_this_atom = self.ind_atom_fixed,
                                                    )
            self._lambdas.append(lam)
            self.lambda_coordinates[lam] = None
            self.simulation_info_u[lam] = None
            self.simulation_info_T[lam] = None
            self.lambda_colors[lam] = plt.cm.jet(1.0 - lam)
            print('adding system at lambda:',lam,)
        else:
            if verbose:
                print('! this system was already added,')
                print('the current systems have the following lambdas:')
                print(self.lambdas)
            else: pass

    def remove_lambda_(self, lam, verbose=True):
        lam = float(lam)
        if lam in self._lambdas:

            self._lambdas = [ _lam for _lam in self._lambdas if _lam != lam ]
            del self.lambda_coordinates[lam]
            del self.simulation_info_u[lam]
            del self.simulation_info_T[lam]
            del self.lambda_colors[lam]

            x = self.generic_name[:3]+'_'+self.generic_name[3:]
            name_lambda_evaluations = self.working_dir_folder_name+'/'+x+'_lambda_evaluations'
            try:
                remove_lambda_from_lambda_evaluations_(name_lambda_evaluations, lam)
                self.import_lambda_evaluations_()
            except:
                print(f'could not delete evaluations involving lambda {lam} from {name_lambda_evaluations}')

        else:
            if verbose:
                print(f'the lambda state {lam} is already absent from the current list of systems')
                print('the current systems have the following lambdas:')
                print(self.lambdas)
            else: pass

    def is_converged_else_remove_all_unconverged_datasets_(self, remove=True):
        converged = []
        for lam in self.lambdas:
            uii = self.lambda_evaluations[(lam, lam)]
            converged.append(TestConverged_1D(uii, verbose=False)())
        inds_remove = np.where(np.array(converged) == False)[0]
        if len(inds_remove) > 0:
            lambdas_remove = np.array(self.lambdas)[inds_remove].tolist()
            print(f'lambda states {lambdas_remove} may not be well converged')
            if remove:
                if self.path_lambda_1_dataset is not None: assert 1.0 not in lambdas_remove 
                    # the samples of the physical state provided must be converged.
                else: pass
                print(f'lambda states {lambdas_remove} are being removed !')
                for lam in lambdas_remove:
                    self.remove_lambda_(lam)
            else: pass 

            return False
        else:
            return True

    ##
    
    def run_(self, lam : float, n_saves : int):
        lam = float(lam)
        if lam not in self._lambdas: self.add_lambda_(lam)
        else: pass
        self.lambda_systems[lam].run_simulation_(n_saves, verbose_info=' lambda='+str(lam))
        self.save_simulations_([lam])
        self.import_data_([lam], verbose=False)

    ##
    def last_m_frames_converged_(self,
                                 lam,
                                 m,
                                 average_temperature_error_allowed=2.0, # K
                                 ):
        if self.lambda_exists_(lam):
            try:
                averages = cumulative_average_(self.simulation_info_T[lam])
                T_target = self.lambda_systems[lam].sc.T
                if np.abs(averages[-m:] - T_target).max() < average_temperature_error_allowed and len(averages[-m:]) == m:
                    return True
                else: return False
            except: return False
        else: return False

    def run_to_get_m_coverged_(self, lam, m=5000, average_temperature_error_allowed=2.0):
        self.add_lambda_(lam, verbose=False)
        increment_size = min([500,m])
        a = 0
        converged = True
        while not self.last_m_frames_converged_(lam, m, average_temperature_error_allowed=average_temperature_error_allowed):
            self.run_(lam, increment_size)
            a +=1
            if a*increment_size >= m*1:
                print('!! simulation at lambda ',lam,' was not converged within temperature tolerance, this information was saved')
                converged = False
                break
            else: pass
        self.lambda_dataset_converged[lam] = converged

    def lambda_sample_sizes_(self, lam=None, m=None):
        if m is None: return self.lambda_sample_sizes[lam]
        else:
            if lam is None:
                for lam in self.lambdas:
                    self.lambda_sample_sizes[lam] = m
            else: 
                self.lambda_sample_sizes[lam] = m

    def which_lambda_to_add_next_(self, max_dataset_size_per_lambda):

        if len(self.lambda_pairs_SE_BAR) > 0 or len(self.lambda_pairs_SE_mBAR) > 0:
            if len(self.lambda_pairs_SE_BAR) > 0:
                lambda_pairs_SE = self.lambda_pairs_SE_BAR
            else:
                lambda_pairs_SE = self.lambda_pairs_SE_mBAR

            values = [1e6 if not np.isfinite(x) else x for x in lambda_pairs_SE.values()]
            lam_lower, lam_higher = [x for x in lambda_pairs_SE.keys()][np.argmax([x for x in values])]
            
            self.stat_amount_of_data(verbose=False)
            ''' 
            lambda_pairs_ascending_error = np.array([x for x in lambda_pairs_SE.keys()])[np.argsort([-x for x in values])]

            lambdas_priority = []
            for x in lambda_pairs_ascending_error:
                for lam in x:
                    if self.lambda_numbers_of_samples[lam] < max_dataset_size_per_lambda:
                        lambdas_priority.append(lam)
                    else: pass
            '''
            lambda_SE = dict(zip(self.lambdas, np.zeros(self.n_lambdas)))
            for i in range(self.n_lambdas-1):
                lamA = self.lambdas[i]
                lamB = self.lambdas[i+1]
                lambda_SE[lamA] += lambda_pairs_SE[(lamA, lamB)]
                lambda_SE[lamB] += lambda_pairs_SE[(lamA, lamB)]
            values = [1e6 if not np.isfinite(x) else x for x in lambda_SE.values()]
            lambda_ascending_error = np.array([x for x in lambda_SE.keys()])[np.argsort([-x for x in values])]
            lambdas_priority = []
            for lam in lambda_ascending_error:
                if self.lambda_numbers_of_samples[lam] < max_dataset_size_per_lambda:
                    lambdas_priority.append(lam)
                else: pass

            return [half_way_(lam_lower, lam_higher)], [lam_lower, lam_higher], lambdas_priority
        else: 
            print('! BAR or mBAR was not yet ran')
            return None

    def unsupervised_FE_(self,
                        batch_size_increments = 10000,
                        max_dataset_size_per_lambda = 50000,
                        max_n_lambdas = 30,
                        SE_tol_per_molecule=0.03125,
                        re_evaluate = False,
                        rerun_questionable_data = False,
                        ):
        ################ BAR

        FE_method_ = self.rerun_cumulative_BAR_result_ #  arg: re_evaluate
        current_error_ = lambda : [1e6 if not np.isfinite(self.SE_BAR) else self.SE_BAR][0]

        ##
        if self.lambda_exists_(1.0): pass
        else: self.add_lambda_(1.0)
        try:
            if len(self.simulation_info_u[1.0]) == max_dataset_size_per_lambda: pass
            else:
                assert 1==0
        except:
                if self.path_lambda_1_dataset is not None: self.load_lambda1_dataset_seperately_(max_dataset_size_per_lambda)
                else: self.run_to_get_m_coverged_(1.0, m=max_dataset_size_per_lambda)
        ##

        if self.lambda_exists_(0.0) and len(self.simulation_info_u[1.0]) ==  max_dataset_size_per_lambda: pass
        else:
            self.run_to_get_m_coverged_(0.0, m=max_dataset_size_per_lambda)
            
        SE_tol = self.lambda_systems[0.0].sc.n_mol*SE_tol_per_molecule

        def main_():
            
            FE_method_(re_evaluate=re_evaluate)

            while self.n_lambdas < max_n_lambdas and current_error_() > SE_tol:
                self.run_to_get_m_coverged_(self.which_lambda_to_add_next_(max_dataset_size_per_lambda)[0][0],
                                            m=batch_size_increments)
                FE_method_(re_evaluate=re_evaluate)

            # improving existing data

            self.stat_amount_of_data(verbose=False)
            sample_sizes = dict(self.lambda_numbers_of_samples)

            while current_error_() > SE_tol:
                
                self.stat_amount_of_data(verbose=False)
                
                if np.mean([x for x in self.lambda_numbers_of_samples.values()]) >= max_dataset_size_per_lambda:
                    print('!! the process was complete but FE did not yet converge within tolerance')
                    break
                else: pass

                lambdas_priority = self.which_lambda_to_add_next_(max_dataset_size_per_lambda)[-1]
                print('lambdas_priority:', lambdas_priority)
                lambda_run_next = lambdas_priority[0]
                sample_sizes[lambda_run_next] = sample_sizes[lambda_run_next] + batch_size_increments
                
                self.run_to_get_m_coverged_(lambda_run_next, m=sample_sizes[lambda_run_next])
                
                FE_method_(re_evaluate=re_evaluate)
        
        main_()

        if rerun_questionable_data:
            fail = 0
            while not self.is_converged_else_remove_all_unconverged_datasets_(remove=True):
                print('some datasets were not converged in terms of average potential energy, trying to sample them again:')
                batch_size_increments = max_dataset_size_per_lambda # only a few involved, can sample properly
                main_()
                fail += 1
                if fail == 3: 
                    print('\n!!! the system in too unstable with this FF, that may not work with the current version of ECM\n')
                    break
                else: pass
        else: 
            if not self.is_converged_else_remove_all_unconverged_datasets_(remove=False):
                print('! double check sampled data is converged in terms of average potential energy')
            else: print('good, add data seems converged')

    def save_simulations_(self, which_lambdas : list = None):
        if which_lambdas is None: which_lambdas = self.lambdas
        else: pass
        for lam in which_lambdas:
            xyz = self.lambda_systems[lam].xyz
            if xyz is None: print('! nothing to save for simulation at lam = '+str(lam))
            else:
                name = self.working_dir_folder_name+'/'+self.generic_name + str(lam) + '_'
                u = self.lambda_systems[lam].u
                T = self.lambda_systems[lam].temperature

                xyz_existing = self.lambda_coordinates[lam]
                if xyz_existing is None:
                    save_pickle_(xyz, name+'xyz', verbose=False)
                    save_pickle_(u, name+'u', verbose=False)
                    save_pickle_(T, name+'T', verbose=False)
                else:
                    u_existing = self.simulation_info_u[lam]
                    T_existing = self.simulation_info_T[lam]
                    save_pickle_(np.concatenate([xyz_existing,xyz]), name+'xyz', verbose=False)
                    save_pickle_(np.concatenate([u_existing,u]), name+'u', verbose=False)
                    save_pickle_(np.concatenate([T_existing,T]), name+'T', verbose=False)                 

                self.lambda_systems[lam].set_arrays_blank_()

    @property
    def which_lambdas_exist_in_folder(self,):
        lambdas_exist_in_folder = []
        folder = self.working_dir_folder_name
        # try find datasets with generic_name part in file name in the working directory.
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for file in files:
            if self.generic_name in file:
                lam = float(file.split('_')[-2])
                if lam not in lambdas_exist_in_folder:
                    lambdas_exist_in_folder.append(lam)
                else: pass
            else: pass
        if len(lambdas_exist_in_folder)>0: return lambdas_exist_in_folder
        else: return None

    def load_dataset_(self, lam : float, verbose=False, custom_generic_name=None):
        lam = float(lam)
        if lam not in self._lambdas:
            self.add_lambda_(lam)
        else: pass
        folder = self.working_dir_folder_name

        if custom_generic_name is not None:
            name = custom_generic_name + str(lam) + '_'
            print('trying to import', name, 'rather than',self.generic_name+str(lam)+'_')
        else:
            name = self.generic_name + str(lam) + '_'

        # try find dataset with name& in directory.
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        could_find = False
        for file in files:
            if name in file:
                xyz = load_pickle_(folder+'/'+name+'xyz')
                u = load_pickle_(folder+'/'+name+'u')
                T = load_pickle_(folder+'/'+name+'T')
                could_find = True
            else: pass
        if could_find:
            return xyz, u, T
        else:
            if verbose: print('! could not find the dataset at lambda',lam)
            else: pass
            return [None]*3

    def load_lambda1_dataset_seperately_(self, m=None):
        dataset = load_pickle_(self.path_lambda_1_dataset)
        r = dataset['MD dataset']['xyz']
        u = dataset['MD dataset']['u']
        T = dataset['MD dataset']['T']

        if m is not None:
            assert m <= len(u)
            try:
                self.import_inds_rand_lambda1_(m)
            except:
                def find_split_indices_(u, split_where, tol=0.00001):
                    u = np.array(u)
                    n = u.shape[0]
                    target = u.mean()
                    for i in range(1000):
                        inds_rand = np.random.choice(n,n,replace=False)
                        randomised = np.array(u[inds_rand])
                        if np.abs(randomised[:split_where].mean() - target) < tol and np.abs(randomised[split_where:].mean() - target) < tol:
                            print('found !')
                            return inds_rand
                        else: pass
                    print('! not found')
                    return None
                inds_rand = None ; a=0
                while inds_rand is None:
                    a+=1
                    print('inds_rand attempt:',a)
                    inds_rand = find_split_indices_(u,
                                                    split_where=m,
                                                    tol=0.0001)
                self.inds_rand_lambda1 = inds_rand
                self.save_inds_rand_lambda1_(m)

            r = np.array(r[self.inds_rand_lambda1][:m])
            u = np.array(u[self.inds_rand_lambda1][:m])
            T = np.array(T[self.inds_rand_lambda1][:m])
        else: pass

        self.lambda_coordinates[1.0] = r
        self.simulation_info_u[1.0] = u
        self.simulation_info_T[1.0] = T

    def import_data_(self,
                     which_lambdas : list,
                     verbose=True,
                     custom_generic_name = None,
                     ):
        if which_lambdas is None: which_lambdas = self.lambdas
        else: pass
        n_found = 0
        for lam in which_lambdas:
            xyz_imported, u_imported, T_imported = self.load_dataset_(lam, verbose=verbose, custom_generic_name=custom_generic_name)
            if xyz_imported is not None:
                self.lambda_coordinates[lam] = xyz_imported
                self.simulation_info_u[lam] = u_imported
                self.simulation_info_T[lam] = T_imported
                n_found += 1
            else: pass
        if verbose:
            print(str(n_found)+'/'+str(len(which_lambdas))+' datasets were found for system '+self.generic_name)
            if n_found > 0: print('these dataset were imported.',np.sort(which_lambdas).tolist())
            else: print('run .run_(lam, n_saves) to generate datasets')
        else: pass

    def stat_amount_of_data(self, verbose=True):
        self.lambda_numbers_of_samples = {}
        for lam in self.lambdas:
            self.lambda_numbers_of_samples[lam] = self.lambda_coordinates[lam].shape[0]
        if verbose:
            for lam in self.lambdas:
                print('lam:',lam,self.lambda_coordinates[lam].shape)
        else: pass

    def global_COM_remover_(self, r):
        mu = np.array(self.lambda_systems[self._lambdas[0]].mu)
        return r - (r*mu).sum(-2, keepdims=True)

    def atom_based_COM_remover_(self, r):
        # one atom with mass of 0
        return r - r[...,self.ind_atom_fixed:self.ind_atom_fixed+1,:]

    def u_(self, r, lam):
        if self.lambda_exists_(lam):
            return self.lambda_systems[lam].u_(self.COM_remover_(r))
        else: return None

    def u_on_r_faster_helper_(self, lam_u, lam_r,  _from):
        return self.u_(self.lambda_coordinates[lam_r][_from:],
                       lam_u,
                       )

    def u_on_r_faster_(self, lam_u, lam_r, m=None):
        # logging all u_{i}(r^{[j]}) evaluations so dont need to evaluate same structures twice
        N_AB = self.lambda_coordinates[lam_r].shape[0]

        try:    ur_A = self.lambda_evaluations[(lam_u, lam_r)]
        except: ur_A = np.array([]).reshape([0,1])

        N_A = ur_A.shape[0]
        if N_A < N_AB:
            ur_B = self.u_on_r_faster_helper_(lam_u, lam_r, _from=N_A)
            ur = np.concatenate([ur_A,ur_B],axis=0)
        else:
            ur = ur_A

        self.lambda_evaluations[(lam_u, lam_r)] = ur

        if m is not None:
            return ur[:m]
        else:
            return ur

    ##

    def plot_overlaps_(self, lam_i, lam_j,
                       m=None,
                       figsize=(10,1.5),
                       separate=False):
        
        # potential i, structures sampled from potentials i and j respectively
        uii = self.u_on_r_faster_(lam_r=lam_i, lam_u=lam_i, m=m)
        uij = self.u_on_r_faster_(lam_r=lam_j, lam_u=lam_i, m=m)
        # potential j, structures sampled from potentials j and i respectively
        ujj = self.u_on_r_faster_(lam_r=lam_j, lam_u=lam_j, m=m)
        uji = self.u_on_r_faster_(lam_r=lam_i, lam_u=lam_j, m=m)

        fig = plt.figure(figsize=figsize)
        plot_1D_histogram_(uii, range=None, color=self.lambda_colors[lam_i])
        plot_1D_histogram_(uij, range=None, color=self.lambda_colors[lam_i], linestyle='dotted')
        if separate:
            plt.show()
            fig = plt.figure(figsize=figsize)
        else: pass
        plot_1D_histogram_(ujj, range=None, color=self.lambda_colors[lam_j])
        plot_1D_histogram_(uji, range=None, color=self.lambda_colors[lam_j], linestyle='dotted')
        plt.show()

    def estimate_local_FE_difference_(self,
                                      lam_i, lam_j,
                                      m_i: int = None, m_j: int = None,
                                      verbose = True,
                                      ):
        # BAR 
        lam_lower = min([lam_i, lam_j])
        lam_higher = max([lam_i, lam_j])
        m_lower = [m_i, m_j][np.argmin([lam_i, lam_j])]
        m_higher = [m_i, m_j][np.argmax([lam_i, lam_j])]


        u_lower_on_lower = self.u_on_r_faster_(lam_r=lam_lower,
                                                lam_u=lam_lower,
                                                m=m_lower)
                   
        u_lower_on_higher = self.u_on_r_faster_(lam_r=lam_higher,
                                                lam_u=lam_lower,
                                                m=m_higher)
        
        u_higher_on_lower = self.u_on_r_faster_(lam_r=lam_lower,
                                                lam_u=lam_higher,
                                                m=m_lower)

        u_higher_on_higher = self.u_on_r_faster_(lam_r=lam_higher,
                                                 lam_u=lam_higher,
                                                 m=m_higher)
        
        u_lower_on_data = [u_lower_on_lower, u_lower_on_higher]
        u_higher_on_data = [u_higher_on_lower, u_higher_on_higher]

        Ns = np.array([x.shape[0] for x in u_lower_on_data])
        _Ns = np.array([x.shape[0] for x in u_higher_on_data])
        assert ((Ns-_Ns)**2).sum() == 0

        Q = np.zeros([2,Ns.sum()])

        Q[0,:] = np.concatenate(u_lower_on_data,  axis=0)[:,0]
        Q[1,:] = np.concatenate(u_higher_on_data, axis=0)[:,0]

        mbar_res = MBAR(Q, Ns, solver_protocol="robust").compute_free_energy_differences()
        Delta_f  = mbar_res['Delta_f'][0,-1]
        dDelta_f = mbar_res['dDelta_f'][0,-1]

        self.lambda_Deltaf[lam_lower] = Delta_f
        self.lambda_dDeltaf[lam_lower] = dDelta_f

        self.lambda_pairs_SE_BAR[lam_lower,lam_higher] = dDelta_f

        self.Ns_used_for_BAR[lam_lower] = Ns[0]
        self.Ns_used_for_BAR[lam_higher] = Ns[1]

        #self.EXP0[lower] = - np.log(np.exp(-(Q[1,:Ns[0]] - Q[0,:Ns[0]])).mean()) 
        #self.EXP1[lower] =   np.log(np.exp( (Q[1,Ns[0]:] - Q[0,Ns[0]:])).mean())

        if verbose:
            print('estimated FE difference between lambdas '+
                  '{:5.7s}'.format(str(lam_lower))+' and '+'{:5.7s}'.format(str(lam_higher))+
                  ' : '+'{:10.20s}'.format(str(Delta_f))+' +/- '+'{:10.20s}'.format(str(dDelta_f))+' kT')
        else: pass

        return Delta_f, dDelta_f
    
    def estimate_FE_using_mBAR_(self, re_evaluate=False):
        assert self.lambda_exists_(0.0)
        assert self.lambda_exists_(1.0)

        if re_evaluate:
            print('all previous energy evaluations were deleted, because re_evaluate=True')
            self.lambda_evaluations = {}
        else: pass
        
        lambdas = self.lambdas
        self.stat_amount_of_data(verbose=False)
        Ns = np.array([self.lambda_numbers_of_samples[lam] for lam in lambdas])
        N = Ns.sum()

        Q = np.zeros([self.n_lambdas, N])
        a = 0
        for lam_u in lambdas:
            print('evaluating all data ('+str(N)+' datapoints) on potential with lambda = '+str(lam_u)) # end='\r'
            Q[a] = np.concatenate([self.u_on_r_faster_(lam_u=lam_u, lam_r=lam_r) for lam_r in lambdas], axis=0).flatten()
            self.save_lambda_evaluations_()
            a += 1

        mbar_res = MBAR(Q, Ns).compute_free_energy_differences()
        self.FE_ij_mBAR  = mbar_res['Delta_f']
        self.SE_ij_mBAR  = mbar_res['dDelta_f']

        self.lambda_pairs_SE_mBAR = {}
        for i in range(self.n_lambdas-1):
            self.lambda_pairs_SE_mBAR[lambdas[i],lambdas[i+1]] = self.SE_ij_mBAR[i,i+1]

        self.FE_mBAR = self.FE_ij_mBAR[0,-1]
        self.SE_mBAR = self.SE_ij_mBAR[0,-1]

        f0 = self.lambda_systems[0.0].lambda_0.f0

        print('FE of system was estimated using mBAR:',self.FE_mBAR+f0,'+/-',self.SE_mBAR,'kT')

        self.log_mBAR_results = [f0, lambdas, 
                                 Ns, self.lambda_numbers_of_samples,
                                 self.lambda_dataset_converged,
                                 self.FE_mBAR, self.SE_mBAR, 
                                 self.FE_ij_mBAR, self.SE_ij_mBAR,
                                 ]
        self.save_mBAR_results_()

    def estimate_FE_(self,
                     m : int=None,
                     verbose = True,
                     ):
        assert self.lambda_exists_(0.0)
        assert self.lambda_exists_(1.0)

        self.lambda_pairs_SE_BAR = {}

        if verbose: print('FE estimate for system with name :',self.generic_name)
        else: pass

        lambdas = self.lambdas
        for i in range(self.n_lambdas-1):
            self.estimate_local_FE_difference_( lam_i=lambdas[i], lam_j=lambdas[i+1],
                                                m_i=m, m_j=m,
                                                verbose=verbose)

        self.FE_BAR = self.lambda_systems[0.0].lambda_0.f0
        self.SE_BAR = 0.0
        #self.FE_EXP0 = 0.0
        #self.FE_EXP1 = 0.0
        for lam in self.lambda_Deltaf.keys():
            self.FE_BAR += self.lambda_Deltaf[lam]
            self.SE_BAR += self.lambda_dDeltaf[lam]
            #self.FE_EXP0 += self.EXP0[lam]
            #self.FE_EXP1 += self.EXP1[lam]

        if verbose:
            print('FE of system was estimated using BAR:',self.FE_BAR,'+/-',self.SE_BAR,'kT')
            print('current mean temperature of lambdas 0 and 1:\n', self.simulation_info_T[0.0].mean(), self.simulation_info_T[1.0].mean(),'\n T (expected):',self.lambda_systems[1.0].sc.T)
            print('u1_mean:', self.simulation_info_u[1.0].mean())
        else: pass

        self.log_BAR_results += [self.gather_BAR_info_()]
        self.save_BAR_results_()
        print('this result was added to log_of_BAR_results and saved')
        self.save_lambda_evaluations_()

    def gather_BAR_info_(self, m_lambdas=None):
        lambdas = self.lambdas
        deltaFEs = np.zeros([self.n_lambdas])
        SEs = np.zeros([self.n_lambdas])
        numbers_of_samples = np.zeros([self.n_lambdas])
        for i in range(self.n_lambdas-1):
            lower_lambda = lambdas[i]
            deltaFEs[i] = self.lambda_Deltaf[lower_lambda]
            SEs[i] = self.lambda_dDeltaf[lower_lambda]
            numbers_of_samples[i] = self.Ns_used_for_BAR[lower_lambda]
        numbers_of_samples[-1] = self.Ns_used_for_BAR[1.0]
        delta_f = deltaFEs.sum()
        f0 = self.lambda_systems[0.0].lambda_0.f0
        f0_parts = self.lambda_systems[0.0].lambda_0.f0_parts
        SE = SEs.sum()
        FE = f0 + delta_f
        u_mean = np.zeros([self.n_lambdas])
        T_mean = np.zeros([self.n_lambdas])
        if m_lambdas is None:
            for i in range(self.n_lambdas):
                u_mean[i] = self.simulation_info_u[lambdas[i]].mean()
                T_mean[i] = self.simulation_info_T[lambdas[i]].mean()
        else:
            for i in range(self.n_lambdas):
                u_mean[i] = self.simulation_info_u[lambdas[i]][:m_lambdas[lambdas[i]]].mean()
                T_mean[i] = self.simulation_info_T[lambdas[i]][:m_lambdas[lambdas[i]]].mean()
        return [FE, SE,
                delta_f, f0, f0_parts,
                deltaFEs, SEs,
                np.stack([np.array(lambdas), u_mean, T_mean],axis=-1), # (n_lam, 3)
                numbers_of_samples]

    def rerun_cumulative_BAR_result_(self,
                                     n_windows : int = 1, # choose such that windows are of equal time
                                     re_evaluate = False,
                                     reruning_logs = False,
                                     save_evaluations = True,
                                     ):
        
        assert self.lambda_exists_(0.0)
        assert self.lambda_exists_(1.0)
        self.stat_amount_of_data(verbose=False)

        if re_evaluate:
            print('all previous energy evaluations were deleted, because re_evaluate=True')
            self.lambda_evaluations = {}
        else: pass

        print('FE estimate for system with name :',self.generic_name)

        if reruning_logs:
            self.log_BAR_results = []
        else: pass
        
        self.lambda_pairs_SE_BAR = {}

        lambdas = self.lambdas
        window_sizes = []
        for lam in lambdas:
            window_sizes.append(self.lambda_numbers_of_samples[lam]//n_windows)

        for k in range(1, n_windows+1):
            m_lambdas = {}
            for i in range(self.n_lambdas-1):
                
                m_i = k*window_sizes[i]
                m_j = k*window_sizes[i+1]

                m_lambdas[lambdas[i]] = m_i
                m_lambdas[lambdas[i+1]] = m_j

                self.estimate_local_FE_difference_( lam_i=lambdas[i], lam_j=lambdas[i+1],
                                                    m_i=m_i, m_j=m_j,
                                                    verbose=False,
                                                    )

            self.FE_BAR = self.lambda_systems[0.0].lambda_0.f0
            self.SE_BAR = 0.0
            #self.FE_EXP0 = 0.0
            #self.FE_EXP1 = 0.0
            for lam in self.lambda_Deltaf.keys():
                self.FE_BAR += self.lambda_Deltaf[lam]
                self.SE_BAR += self.lambda_dDeltaf[lam]
                #self.FE_EXP0 += self.EXP0[lam]
                #self.FE_EXP1 += self.EXP1[lam]

            print('FE of system was estimated using BAR:',self.FE_BAR,'+/-',self.SE_BAR,'kT')

            self.log_BAR_results += [self.gather_BAR_info_(m_lambdas=m_lambdas)]
        self.save_BAR_results_()
        
        if save_evaluations: self.save_lambda_evaluations_()
        else: pass

"""
P3 settings:
"""

def succinic_acid_ARGS_oss(form, cell, key='_'):
    args_initialise_object = {
        'PDB' : DIR_main+'/MM/GAFF_sc/succinic_acid/succinic_acid_'+form+'_equilibrated'+key+'cell_'+cell+'.pdb',
        'n_atoms_mol': 14,
        'name': 'succinic_acid',
        'FF_name' : 'GAFF',
    }
    args_initialise_system = {
        'PME_cutoff': 0.36,
        'removeCMMotion': True,
        'nonbondedMethod': app.PME,
        'custom_EwaldErrorTolerance': 0.0001,
        'constraints': None,   
    }
    args_initialise_simulation = {
        'rbv': None,
        'minimise': False,
        'T': 300,
        'timestep_ps': 0.002,
        'collision_rate': 1,
        'P': None, # NVT
        'barostat_type': 1,
        'barostat_1_scaling': [True, True, True],
        'stride_barostat': 25,
    }
    ARGS_oss = [args_initialise_object, args_initialise_system, args_initialise_simulation]
    return ARGS_oss

def veliparib_ARGS_oss(form, cell, key='_'):
    args_initialise_object = {
        'PDB' : DIR_main+'/MM/GAFF_sc/veliparib/veliparib_'+form+'_equilibrated'+key+'cell_'+cell+'.pdb',
        'n_atoms_mol': 34,
        'name': 'veliparib',
        'FF_name' : 'GAFF',
    }
    args_initialise_system = {
        'PME_cutoff': 0.36,
        'removeCMMotion': True,
        'nonbondedMethod': app.PME,
        'custom_EwaldErrorTolerance': 0.0001,
        'constraints': None,   
    }
    args_initialise_simulation = {
        'rbv': None,
        'minimise': False,
        'T': 300,
        'timestep_ps': 0.0005, # !
        'collision_rate': 1,
        'P': None, # NVT
        'barostat_type': 1,
        'barostat_1_scaling': [True, True, True],
        'stride_barostat': 25,
    }
    ARGS_oss = [args_initialise_object, args_initialise_system, args_initialise_simulation]
    return ARGS_oss

def mivebresib_ARGS_oss(form, cell, key='_'):
    args_initialise_object = {
        'PDB' : DIR_main+'/MM/GAFF_sc/mivebresib/mivebresib_'+form+'_equilibrated'+key+'cell_'+cell+'.pdb',
        'n_atoms_mol': 51,
        'name': 'mivebresib',
        'FF_name' : 'GAFF',
    }
    args_initialise_system = {
        'PME_cutoff': 0.36,
        'removeCMMotion': True,
        'nonbondedMethod': app.PME,
        'custom_EwaldErrorTolerance': 0.0001,
        'constraints': None,   
    }
    args_initialise_simulation = {
        'rbv': None,
        'minimise': False,
        'T': 300,
        'timestep_ps': 0.0005, # !
        'collision_rate': 1,
        'P': None, # NVT
        'barostat_type': 1,
        'barostat_1_scaling': [True, True, True],
        'stride_barostat': 25,
    }
    ARGS_oss = [args_initialise_object, args_initialise_system, args_initialise_simulation]
    return ARGS_oss

## ##

def remove_lambda_from_lambda_evaluations_(name_old:str, lam, name_new=None):
    ' 1/2 parts of patch breakage '
    old_dict = load_pickle_(name_old)
    new_dict = dict(old_dict)
    if name_new is None: name_new = name_old
    else: pass
    for key in old_dict.keys():
        if lam in key:
            del new_dict[key]
        else: pass
    save_pickle_(new_dict, name_new)

## ##
