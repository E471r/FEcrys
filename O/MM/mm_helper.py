''' mm_helper.py
ECM:
    f : update_HarmonicBondForce_
    f : update_HarmonicAngleForce_
    f : update_PeriodicTorsionForce_
    f : update_RBTorsionForce_ ; not tested yet, because not in GAFF
    f : update_NonbondedForce_
    f : update_CustomNonbondedForce_ ; carefull with param1 vs 2, in GAFF was ok
    f : put_lambda_into_system_ ; ! add method for update_RBTorsionForce_

MM:
    c : MM_system_helper
    f : plot_simulation_info_
    f : cell_lengths_and_angles_ ; mdtraj also has this method

make supercells:
    f : get_unitcell_stack_order_
    f : supercell_from_unitcell_

check box for opnemm:
    f : box_in_reduced_form_
    f : reducePeriodicBoxVectors_

fix ordering of atoms in a molecule:
    f : reorder_atoms_mol_
    f : validate_reorder_atoms_mol_
    f : reorder_atoms_unitcell_

misc:
    f : vectors_between_atoms_
    f : change_box_
    f : remove_clashes_
    f : rename_atoms_ ; not used, blank
    f : process_mercury_output_ ; not used?
    
not used:
    f : extract_subcell_from_supercell_
    f : CustomIntegrator_
    
'''

from ..util_np import *
from ..plotting import *

try:
    import openmm as mm
    import openmm.unit as unit
    import openmm.app as app
except ImportError:
    from simtk import openmm as mm
    from simtk import unit as unit
    from simtk.openmm import app as app
import parmed

# from multicontext_openmm import MultiContext
# import openmmtools as mm_tools

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

CONST_kB = 1e-3*8.31446261815324 # kilojoule/(kelvin*mole)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

def get_force_by_name_(system, name:str):
    forces = {system.getForce(index).__class__.__name__: system.getForce(
        index) for index in range(system.getNumForces())}
    return forces[name]

def update_HarmonicBondForce_(_force, _lam, deepcopy=True):
    if deepcopy: force = copy.deepcopy(_force)
    else: force = _force
    for i in range(force.getNumBonds()):
        old_params = force.getBondParameters(i)
        p1,p2,r0,k = old_params
        assert k.unit == unit.kilojoule/(unit.nanometer**2*unit.mole) ; new_k = k*_lam
        new_params = [i,p1,p2,r0,new_k]
        force.setBondParameters(*new_params)
    return force

def update_HarmonicAngleForce_(_force, _lam, deepcopy=True):
    if deepcopy: force = copy.deepcopy(_force)
    else: force = _force
    for i in range(force.getNumAngles()):
        old_params = force.getAngleParameters(i)
        p1,p2,p3,theta,k = old_params
        assert k.unit == unit.kilojoule/(unit.mole * unit.radian**2) ; new_k = k*_lam
        new_params = [i,p1,p2,p3,theta,new_k]
        force.setAngleParameters(*new_params)
    return force

def update_PeriodicTorsionForce_(_force, _lam, deepcopy=True):
    if deepcopy: force = copy.deepcopy(_force)
    else: force = _force
    unit_of_item_to_scale = unit.kilojoule/unit.mole
    for i in range(force.getNumTorsions()):
        old_params = force.getTorsionParameters(i)
        p1,p2,p3,p4,L,phi,k = old_params
        assert k.unit == unit.kilojoule/unit.mole ; new_k = k*_lam
        new_params = [i,p1,p2,p3,p4,L,phi,new_k]
        force.setTorsionParameters(*new_params)   
    return force

def update_RBTorsionForce_(_force, _lam, deepcopy=False):

    ''' only in OPLS, not in GAFF or TIP4P '''
    
    if deepcopy: force = copy.deepcopy(_force)
    else: force = _force
    for i in range(force.getNumTorsions()):
        old_params = force.getTorsionParameters(i)
        p1,p2,p3,p4, c0,c1,c2,c3,c4,c5 = old_params
        #assert c.unit == unit.kilojoule/unit.mole
        new_c0 = c0*_lam
        new_c1 = c1*_lam
        new_c2 = c2*_lam
        new_c3 = c3*_lam
        new_c4 = c4*_lam
        new_c5 = c5*_lam
        new_params = [i, p1,p2,p3,p4, new_c0,new_c1,new_c2,new_c3,new_c4,new_c5]
        force.setTorsionParameters(*new_params)   
    return force

def update_NonbondedForce_(_force, _lam, deepcopy=True):
    if deepcopy: force = copy.deepcopy(_force)
    else: force = _force
        
    # LJ : in both comb. rules 2 and 3 epsilon is treated the same:
    # epsilon_ij = (eps_i * eps_j)**0.5      # _lam * epsilon_ij = ((_lam*eps_i) * (_lam*eps_j))**0.5

    # Electrostatic :
    # qiqj = qi * qj                         # _lam * qiqj = qi*(_lam**0.5) * qj*(_lam**0.5) 

    for i in range(force.getNumParticles()):
        ## LJ + PME
        old_params = force.getParticleParameters(i)
        qi,sigmai,epsi = old_params
        assert epsi.unit == unit.kilojoule/unit.mole ; new_epsi = epsi*_lam
        assert qi.unit == unit.elementary_charge ; new_qi = qi*(_lam**0.5)
        new_params = [i,new_qi,sigmai,new_epsi]
        force.setParticleParameters(*new_params)
        
    for i in range(force.getNumExceptions()):
        ## LJ + PME (exception)
        old_params = force.getExceptionParameters(i)
        pi,pj,qiqj,sigmaij,epsij = old_params
        assert epsij.unit == unit.kilojoule/unit.mole ; new_epsij = epsij*_lam
        assert qiqj.unit == unit.elementary_charge**2 ; new_qiqj = qiqj*_lam
        new_params = [i,pi,pj,new_qiqj,sigmaij,new_epsij]
        force.setExceptionParameters(*new_params)
    return force

def update_CustomNonbondedForce_(_force, _lam, deepcopy=True):
    if deepcopy: force = copy.deepcopy(_force)
    else: force = _force

    # epsilon_ij = (eps_i * eps_j)**0.5      # _lam * epsilon_ij = ((_lam*eps_i) * (_lam*eps_j))**0.5

    # [force.getPerParticleParameterName(0), force.getPerParticleParameterName(1)]
    # ['epsilon', 'sigma']
    try:
        assert force.getGlobalParameterName(0) == 'ecm_lambda'
        force.setGlobalParameterDefaultValue(0, defaultValue=_lam)

    except:
        assert force.getPerParticleParameterName(0) == 'epsilon'

        for i in range(force.getNumParticles()):
            old_params = force.getParticleParameters(i)
            epsilon_i, sigma_i = old_params
            new_epsilon_i = epsilon_i*_lam
            new_params = [new_epsilon_i, sigma_i]
            force.setParticleParameters(i, new_params) # second arg : list (of two numbers)
    return force

def put_lambda_into_system_(system,
                            lam,
                            R0,
                            k_EC=5000.0,
                            verbose=True,
                            inds_true_atoms = None, # indices of atoms that have finite mass.
                            ):
    '''
    REF : http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html
    
    force.getEnergyFunction()
    '''
    # NB : deepcopy does not work : crashing
    
    if inds_true_atoms is None: inds_true_atoms = np.arange(R0.shape[0])
    else: pass

    old_forces = system.getForces() ; n_older_forces = len(old_forces)

    EC_force = mm.CustomExternalForce("gamma*periodicdistance(x, y, z, x0, y0, z0)^2")
    EC_force.addGlobalParameter('gamma', (1.0-lam)*k_EC/2)
    EC_force.addPerParticleParameter('x0')
    EC_force.addPerParticleParameter('y0')
    EC_force.addPerParticleParameter('z0')
    for i in inds_true_atoms:
        EC_force.addParticle(i,R0[i]*unit.nanometers)

    if 0.0 <= lam <= 1.0:
        count_done = 0
        for force in system.getForces():
            #print('group',force.getForceGroup())
            name = force.getName()
            if name == 'HarmonicBondForce':
                update_HarmonicBondForce_(force, lam, deepcopy=False)
            elif name == 'HarmonicAngleForce':
                update_HarmonicAngleForce_(force, lam, deepcopy=False)
            elif name == 'PeriodicTorsionForce':
                update_PeriodicTorsionForce_(force, lam, deepcopy=False)
            elif name == 'RBTorsionForce':
                update_RBTorsionForce_(force, lam, deepcopy=False)
            elif name == 'NonbondedForce':
                update_NonbondedForce_(force, lam, deepcopy=False)
            elif name == 'CustomNonbondedForce':
                update_CustomNonbondedForce_(force, lam, deepcopy=False)
            elif name == 'CMMotionRemover': pass
            else: print('\n','!! force with name',name,'is not in this list; skipping it.','\n')
            count_done += 1
        
        if verbose: print('\n',count_done,'of the relevant',n_older_forces,'default forces were rescaled by', lam,'\n')
        else: pass

        if lam == 1.0: pass
        else: 
            system.addForce(EC_force)
            if verbose: print('\n','adding EC force scaled by', 1-lam, '\n')
            else: pass

    else: print('\n','!! no change because lambda should be \in [0,1]','\n')
        
    if verbose:
        print('\n','forces of the input system:','\n')
        [print(x.getName()) for x in old_forces]

        #print('\n','forces in system being returned:','\n')
        #[print(x.getName()) for x in system.getForces()]
        #print('\n')
    else: pass

## ## 

#def inject_methods_from_another_class_(self, class_to_inject_methods_from):
#    import types
#    for name, method in class_to_inject_methods_from.__dict__.items():
#        if callable(method) and not name.startswith("__"):
#            setattr(self, name, types.MethodType(method, self))

class MM_system_helper:
    def __init__(self,):
        self.reduce_drift = False
        self.NPT = False # switches to permanently True when barostat added, stays true if barostat removed.

    def inject_methods_from_another_class_(self, class_to_inject_methods_from, **kwargs):
        inject_methods_from_another_class_(self, class_to_inject_methods_from, **kwargs)

    def corrections_to_ff_(self, verbose):
        if verbose: print('no corrections to self.system')
        else: pass
    
    @property
    def _system_mass_(self,):
        return np.array([self.system.getParticleMass(i)._value for i in range(self.system.getNumParticles())])

    def define_mu_(self, index_atom=None):
        # run this again after/if fixing one atom
        self._mass_ = self._system_mass_[:,np.newaxis] 
        self._mass_system_ = self._mass_.sum() # daltons
        self._mu_ = self._mass_ / self._mass_system_ # (N,1)

        if index_atom is not None:
            if type(index_atom) is int:
                mu = np.array([0.0]*self.N)
                mu[index_atom] = 1.0
            else:
                mu = np.array([0.0]*self.n_atoms_mol)
                mu[index_atom] = 1.0
                mu = np.concatenate([mu]*self.n_molecules, axis=0)
                mu = np.array(self._mass_[:,0]*mu)
            mu /= mu.sum()
            self._mu_ = np.array(mu[...,np.newaxis])
        else: pass

    def _set_b_(self, b):
        assert b.shape == (3,3)
        self.simulation.context.setPeriodicBoxVectors(*b)

    def _set_r_(self, r):
        assert r.shape == (self.N,3)
        self.simulation.context.setPositions(r)

    def _set_v_(self, v):
        assert v.shape == (self.N,3)
        self.simulation.context.setVelocities(v)

    def forward_atom_index_(self, inds):
        return inds
    
    def inverse_atom_index_(self, inds):
        return inds

    @property
    def _current_r_(self,):
        # positions
        # unit = nanometer
        return self.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
    
    @property
    def _current_COM_(self,):
        # global centre of mass (1,3)
        return (self._current_r_*self._mu_).sum(axis=-2,keepdims=True)
    
    def _recenter_simulation_(self,):
        self._set_r_(self._current_r_ - self._current_COM_)
    
    @property
    def _current_v_(self,):
        # velocities
        # unit = nanometer/picosecond
        return self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)._value
    
    @property
    def _current_p_(self,):
        # momenta
        # unit = nanometer*dalton/picosecond
        return self._current_v_*self._mass_
    
    @property
    def _current_K_(self,):
        # kinetic energy
        # unit = kilojoule/mole
        return self.simulation.context.getState(getEnergy=True).getKineticEnergy()._value
    
    @property
    def _current_T_(self,):
        # temperature
        # unit = Kelvin
        # self.n_DOF <= 3.*self.N
        return (2./(self.n_DOF*CONST_kB)) * self._current_K_
    
    @property
    def _current_U_(self,):
        # potential energy 
        # unit = kilojoule/mole
        return self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
    
    @property
    def _current_u_(self,):
         # reduced potential energy 
         return self._current_U_ * self.beta
    
    @property
    def _current_F_(self,):
        # forces
        # unit = kilojoule/(nanometer*mole) ; [F = -∇U]
        return self.simulation.context.getState(getForces=True).getForces(asNumpy=True)._value
    
    @property
    def _current_b_(self,):
        # box (3,3)
        # unit = nanometer                  ; [box vectors are rows]
        return self.simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)._value
    
    @property
    def _current_V_(self,):
        # volume
        # unit = nanometer**3
        return np.linalg.det(self._current_b_)
    
    @property
    def _current_rho_(self,):
        # density

        # one_dalton_in_grams = 1.66053906660e-24
        # one_nm_in_cm = 1e-07
        # 0.0016605390666000002 = (one_dalton_in_grams/(one_nm_in_cm)**3)

        rho = self._mass_system_ / self._current_V_ # (daltons/nanometers**3)
        rho *= 0.0016605390666000002
        return rho # (g/cm**3)

    '''
    @property
    def _current_P_(self,):
        ' not implemented, TODO: revisit Eq. 8 in REF : https://doi.org/10.1063/1.2363381 '
    '''

    def _add_barostat_to_system_(self,):
        if self.barostat_type == 0:
            self.system.addForce(mm.MonteCarloBarostat(self.P*unit.atmosphere,
                                                       self.T*unit.kelvin,
                                                       self.stride_barostat))
            name = 'MonteCarloBarostat' ; print('setting barostat (',name,') to P =',self.P,'atm, at',self.T,'K, trying every',self.stride_barostat,'frames.')
            self.NPT = True
        elif self.barostat_type == 1:
            self.system.addForce(mm.MonteCarloAnisotropicBarostat([self.P*unit.atmosphere]*3,
                                                                   self.T*unit.kelvin,
                                                                   self.barostat_1_scaling[0],self.barostat_1_scaling[1],self.barostat_1_scaling[2],
                                                                   self.stride_barostat))
            name = 'MonteCarloAnisotropicBarostat' ; print('setting barostat (',name,') to P =',self.P,'atm, at',self.T,'K, trying every',self.stride_barostat,'frames.')
            self.NPT = True
        elif self.barostat_type == 2:
            self.system.addForce(mm.MonteCarloFlexibleBarostat(self.P*unit.atmosphere,
                                                               self.T*unit.kelvin,
                                                               self.stride_barostat,
                                                               True)) #self.rigid))
            name = 'MonteCarloFlexibleBarostat' ; print('setting barostat (',name,') to P =',self.P,'atm, at',self.T,'K, trying every',self.stride_barostat,'frames.')
            self.NPT = True
        else: print('!! barostat_type can only be one of: 0,1,2')

    def _remove_barostat_from_system_(self,):
        a = 0
        for x in self.system.getForces():
            if isinstance(x, mm.MonteCarloBarostat) or isinstance(x, mm.MonteCarloAnisotropicBarostat) or isinstance(x,mm.MonteCarloFlexibleBarostat): 
                self.system.removeForce(a)
                print('! barostat was removed')
            else: a+=1

    '''
    def _add_thermostat_to_system_(self, thermostat_class, params):
        a = 0
        for x in self.system.getForces():
            if isinstance(x, thermostat_class):
                print('replacing previous', x ,' in the system')
                self.system.removeForce(a)
            else: a+=1  
        self.system.addForce(thermostat_class(*params))
    '''

    def initialise_integrator_(self, integrator_class, collision_rate = 1):

        if integrator_class in [mm.LangevinMiddleIntegrator]:
            print('setting integrator to:',integrator_class.__name__,'with collistion rate:',collision_rate,'/ps')
            return integrator_class(self.T*unit.kelvin, collision_rate/unit.picosecond, self.dt*unit.picoseconds)
        
        #elif  integrator_class in [mm.AndersenThermostat]:
        #    self._add_thermostat_to_system_(integrator_class, params = [self.T*unit.kelvin, collision_rate/unit.picosecond])
        #    print('setting integrator to VV, with thermostat:',integrator_class.__name__,'with collistion rate:',collision_rate)
        #    return mm_tools.integrators.VelocityVerletIntegrator(self.dt*unit.picoseconds)
        
        #elif integrator_class in [mm_tools.integrators.AndersenVelocityVerletIntegrator]:
        #    print('setting integrator to:',integrator_class.__name__,'with collistion rate:',collision_rate,'/ps')
        #    return integrator_class(self.T*unit.kelvin, collision_rate/unit.picosecond, self.dt*unit.picoseconds)
        
        #elif integrator_class in [mm.LangevinIntegrator]: # dont use..
        #    print('setting integrator to:',integrator_class.__name__,'with collistion rate:',collision_rate,'/ps')
        #    return integrator_class(self.T*unit.kelvin, collision_rate/unit.picosecond, self.dt*unit.picoseconds)
        
        #elif integrator_class in ['BDP']:
        #    print('! method removed')
        #    return mm_tools.integrators.VelocityVerletIntegrator(self.dt*unit.picoseconds)

        else: print('!! check integrator, can add here.')


    def _list_forces_(self):
        print('')
        for x in self.system.getForces():
            print(x.getName())
        print('')
        
    def turn_ON_nonbonded_SwitchingFunction(self, factor=0.95):
        for x in self.system.getForces():
            if isinstance(x, mm.NonbondedForce) or isinstance(x, mm.CustomNonbondedForce):
                x.setUseSwitchingFunction(True)
                x.setSwitchingDistance(factor * self.PME_cutoff * unit.nanometers)
            else: pass

    def adjust_EwaldErrorTolerance(self, tol, verbose=True):
        for x in self.system.getForces():
            if isinstance(x, mm.NonbondedForce):
                default = x.getEwaldErrorTolerance()
                x.setEwaldErrorTolerance(tol)
                if verbose: print('adjusted EwaldErrorTolerance from',default,'to',x.getEwaldErrorTolerance())
                else: pass
            else: pass

    def _reset_temperature_(self, T : float):
        self.T = T
        self.simulation.integrator.setTemperature(self.T*unit.kelvin)

    # energy:

    @property
    def _print_potential_enrrgy_contributions_(self):
        print('not yet implemented')
 
    '''
    def _init_CPU_evaluation_(self,):

        self.n_cores_evaluation = os.cpu_count()
        print('# cpu cores will be used for evaluation:', self.n_cores_evaluation)
        self.MC = MultiContext(n_workers = self.n_cores_evaluation,
                            system = self.system,
                            integrator = mm.LangevinMiddleIntegrator(0.,0.,0.), # any is fine.
                            platform_name = 'CPU')

        ##  _set_PU_correcton_ ## 
        _r = self._current_r_[np.newaxis,...]
        _b = self._current_b_[np.newaxis,...]
        self.PU_correction = (self._U_GPU_(_r,_b) - self._U_CPU_(_r,_b))[0,0]
        print('discepancy of',self.PU_correction,'kJ/mol between CPU and GPU is being adjusted for by a constant offset.')
        print('double check that eneregies agree on different box shapes')
        print('double check that translating the system does not change energy')
        
    def _U_CPU_(self, r, b=None):
        # not using 
        return self.MC.evaluate(r, box_vectors=b)[0][:,np.newaxis] + self.PU_correction

    def u_GPU_(self, r, b=None, r_full=True):
        # if b None : the last box it seen is assumed.
        return self._U_GPU_(np.array(r), b=b) * self.beta

    '''

    def _U_GPU_(self, r, b=None):
        n_frames = r.shape[0]

        _r = np.array(self._current_r_)
        _b = np.array(self._current_b_)

        U = np.zeros([n_frames,1])
        if b is None:
            for i in range(n_frames):
                self._set_r_(r[i])
                U[i,0] = self._current_U_
        else:
            for i in range(n_frames):
                self._set_r_(r[i])
                self._set_b_(b[i])
                U[i,0] = self._current_U_

        self._set_r_(_r)
        self._set_b_(_b)

        return U

    def u_GPU_(self, r, b=None):
         # if b None : in NVT the last box it seen is assumed.
        return self._U_GPU_(np.array(r), b=b) * self.beta
    
    # minimisation:

    def F_GPU_(self, r, b=None):
        n_frames = r.shape[0]

        _r = np.array(self._current_r_)
        _b = np.array(self._current_b_)

        F = np.zeros([n_frames,self.N,3])
        if b is None:
            for i in range(n_frames):
                self._set_r_(r[i])
                F[i] = self._current_F_
        else:
            for i in range(n_frames):
                self._set_r_(r[i])
                self._set_b_(b[i])
                F[i] = self._current_F_

        self._set_r_(_r)
        self._set_b_(_b)
        
        return F

    def minimise_(self, verbose=True):
        if verbose: print('u before minimisation:',self._current_u_,'kT')
        else: pass
        self.simulation.minimizeEnergy()
        if verbose: print('u after  minimisation:',self._current_u_,'kT')
        else: pass

    def minimise_xyz_(self, r, b=None, verbose=False):
        ''' energy minimising only the coordinates (r), box (b) is fixed
        Inputs:
            r : single frame (N,3) or trajectory (m,N,3)
            b : single frame (3,3) or trajectory (m,3,3)
        Output:
            r : single frame (N,3) or trajectory (m,N,3) after minimising in fixed box(es)
        '''
        def check_shape_(x):
            x = np.array(x)
            shape = len(x.shape)
            assert shape in [2,3]
            if len(x.shape) == 3: pass
            else: x = x[np.newaxis,...]
            return x
        
        r =  np.array(check_shape_(r))
        n_frames = r.shape[0]
        output = np.zeros([n_frames, self.N, 3])

        _r = np.array(self._current_r_)
        _b = np.array(self._current_b_)

        if b is None:
            for i in range(n_frames):
                self._set_r_(r[i])
                self.minimise_(verbose=verbose)
                output[i] = self._current_r_
        else:
            b =  np.array(check_shape_(b))
            assert b.shape[0] == n_frames 
            for i in range(n_frames):
                self._set_r_(r[i])
                self._set_b_(b[i])
                self.minimise_(verbose=verbose)
                output[i] = self._current_r_

        self._set_r_(_r)
        self._set_b_(_b)

        return output
        
    # harmonic approximation:
    
    def _Hessian_(self, r, b=None, dr = 0.0001,
                  fixed_atom_index = None,
                  temperature_reduced = True,
                  ):
        if temperature_reduced:
            assert self.beta > 1e-6, f'temperature {1/(self.beta*CONST_kB)} K is too low'
            temp_rescale = float(self.beta)
        else: temp_rescale = 1.0

        fixed_atom_mask = np.ones([self.N,3])
        if fixed_atom_index is not None: 
            fixed_atom_mask[fixed_atom_index] = 0.0
        else: pass
        fixed_atom_mask = fixed_atom_mask.flatten()
        inds_valid = np.where(fixed_atom_mask>0.5)[0]

        r_flat = np.array(r.flatten())
        Hessian = np.zeros([self.N*3, self.N*3])
        for i in inds_valid:
            dr_i = np.zeros([self.N*3])
            dr_i[i] = dr
            A = reshape_to_atoms_np_(np.array(r_flat + dr_i)[None,...], n_atoms_in_molecule=self.N, n_molecules=1)
            B = reshape_to_atoms_np_(np.array(r_flat - dr_i)[None,...], n_atoms_in_molecule=self.N, n_molecules=1)
            dUdr_A = - self.F_GPU_(A, b=b).flatten()
            dUdr_B = - self.F_GPU_(B, b=b).flatten()
            Hessian[i] = dUdr_B - dUdr_A

        return Hessian * temp_rescale / (2.0 * dr) # finite difference of the forces

    def harmonic_FE_(self, r, b, fixed_atom_index:int, dr = 0.0001,
                     # minimisation settings:
                     n_steps_openmm = 2,
                     n_steps_adam = 5000, # 0 : None to turn it off
                     alpha_adam = 1e-4,
                     verbose = True,
                     ):
        ''' Classical Harmonic Approximation : configurational part only. Momentum not included here.
        Inputs:
            r : (N,3) array of positions
            b : (3,3) array of box 
            fixed_atom_index : index of atom to keep fixed; the only constraint in a crystal.
        Parameters:
            dr : finite difference parameter; can try a few small positive values for a stable output
            n_minimisations : number of times to run potential energy gradient descent (box is fixed).

        Output: dictionary = {
            'f0' = configurational Helmholtz per system FE in kT at the local minimum
            'u0' = potential energy per system in kT at the local minimum
            'r0' = minimised structure at the local minimum
            }

        Limitation: 
            Classical and requires finite temperature
            Box is not minimised, fixed as the original input (b) provided
        '''
        # checking inputs are for a single crystal
        assert r.shape[-2:] == (self.N,3), 'Harmonic_FE_ : r.shape incorrect'
        assert b.shape[-2:] == (3,3), 'Harmonic_FE_ : b.shape incorrect'
        r = np.array(r).reshape([1, self.N, 3])
        b = np.array(b).reshape([1, 3, 3])

        fix_atom_ = lambda r : r - r[:,fixed_atom_index:fixed_atom_index+1]

        r = fix_atom_(r)
        u0_initial = self._U_GPU_(r, b=b).sum() * self.beta

        if n_steps_openmm > 0:
            for step in range(n_steps_openmm):
                # find the local minimum, r only (lattice minimsation not implemented here yet)
                r = self.minimise_xyz_(r, b=b)
                r = fix_atom_(r)
            u0_openmm_minimised = self._U_GPU_(r, b=b).sum() * self.beta
            assert u0_openmm_minimised <= u0_initial
            if verbose: print(textwrap.dedent(f'''
                minimised with OpenMM for {n_steps_openmm} steps
                u : {u0_initial} -> {u0_openmm_minimised}
                '''))

        if n_steps_adam > 0:
            r, n_steps = ADAM_np_(grad_ = lambda _r : - self.F_GPU_(_r, b=b), 
                                  x0 = r, 
                                  constraint_ = fix_atom_,
                                  max_itter = n_steps_adam,
                                  alpha = alpha_adam,
                                  betas=[0.7,0.999], tol=1e-4,
                                  )
            u0_fully_minimised = self._U_GPU_(r, b=b).sum() * self.beta
            assert u0_fully_minimised <= u0_openmm_minimised
            if verbose: print(textwrap.dedent(f'''
                minimised with ADAM for {int(n_steps)} steps
                u : {u0_openmm_minimised} -> {u0_fully_minimised}
                '''))

        u0 = self._U_GPU_(r, b=b).sum() * self.beta
        # units: kT / nm**2
        if verbose: print(f'estimating Hessian ({self.N*3}, {self.N*3}):')
        H = self._Hessian_(r, b=b, dr=dr, fixed_atom_index=fixed_atom_index)
        l_H = np.linalg.svd(H)[1][:-3]         # choosing top (N-1)*3 eigenvalues, others are zero
        FE = 0.0
        # energy:
        FE += u0                               # potential energy at the local minimum in kT
        # entropy:
        FE += np.log(l_H).sum()*0.5            # positive sign here, because H = inv(Cov)
        FE -= (self.N-1)*3*0.5*np.log(2*np.pi) # (N-1)*3 degrees of freedom !
        
        output = {'f0':FE, 'u0':u0, 'r0': r} # kT, kT, nm

        return output

    # simulation:

    def set_arrays_blank_(self,):
        self._xyz = [] # deque() from collections
        self._u = []
        self._temperature = []
        self._boxes = []
        self._COMs = []
        self.n_frames_saved = 0
        self.time_elapsed = 0.0 # seconds
        # self._Ps = []
        # self._v = []

    def save_frame_(self,):
        self._xyz.append( self._current_r_ )          # nm
        self._u.append( self._current_u_ )            #  kT
        self._temperature.append( self._current_T_  ) # K
        self._boxes.append( self._current_b_ )        # nm
        self._COMs.append( self._current_COM_ )       # nm
        self.n_frames_saved += 1                      # frames
        self.time_elapsed += self._time_elapsed_saved_frame_
        # self._Ps.append( self._current_P_ )
        # self._v.append(self._current_v_)

    def run_simulation_(self, n_saves, stride_save_frame:int=100, verbose_info : str = ''):
        self.stride_save_frame = stride_save_frame
        for i in range(n_saves):
            t_start = time.time()
            self.simulation.step(stride_save_frame)
            self._time_elapsed_saved_frame_ = (time.time() - t_start) # NB: save_frame_ is very slow! this does not count it
            self.save_frame_()
            info = 'frame: '+str(self.n_frames_saved)+' T sampled:'+str(self.temperature.mean().round(3))+' T expected:'+str(self.T)+verbose_info
            print(info, end='\r')

    def run_simulation_w_(self, n_saves, stride_save_frame:int=100, verbose_info : str = ''):
        ''' w : wrapped ; for NVT in the presence of shearing (at higher T) or alchemical '''
        self.stride_save_frame = stride_save_frame
        for i in range(n_saves):
            t_start = time.time()
            self.simulation.step(stride_save_frame)
            self._time_elapsed_saved_frame_ = (time.time() - t_start) # NB: save_frame_ is very slow! this does not count it
            state = self.simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.simulation.context.setPositions(state.getPositions())
            self.save_frame_()
            info = 'frame: '+str(self.n_frames_saved)+' T sampled:'+str(self.temperature.mean().round(3))+' T expected:'+str(self.T)+verbose_info
            print(info, end='\r')

    @property
    def xyz(self,):
        if len(self._xyz) > 0: return np.array(self._xyz)
        else:                  return None

    @property
    def velicities(self,):
        if len(self._v) > 0: return np.array(self._v)
        else:                return None

    @property
    def COMs(self,):
        if len(self._COMs) > 0: return np.array(self._COMs)
        else:                   return None      

    @property
    def boxes(self,):
        if len(self._boxes) > 0: return np.array(self._boxes)
        else:                    return None      

    @property
    def u(self,):
        if len(self._u) > 0: return np.array(self._u)[:,np.newaxis]
        else:                return None

    @property
    def temperature(self,verbose=True):
        if len(self._temperature) > 0: return np.array(self._temperature)
        else:                          return None

    @property
    def dt(self,):
        return self.timestep_ps # ps

    @property
    def timescale(self,):
        u = self.u
        if u is None:
            print('! timescale : simulation has not started')
            return None
        else:
            ps = self.dt * self.stride_save_frame * len(u)
            ns = ps*0.001
            return ns
    ##

    @property
    def average_temperature(self,):
        return np.array(self.temperature).mean()
    @property
    def average_energy(self,):
        return np.array(self.u).mean()
    @property
    def average_volume(self,):
        return np.linalg.det(np.array(self.boxes)).mean()
    
    # plotting:

    def plot_simulation_info_(self, figsize=(10,10)):
        ' one plot with all information about the simulation '
        plot_simulation_info_(self, figsize=figsize)

    def plot_temperature_(self, window : float=None):
        Ts = np.array(self.temperature).flatten()
        av_Ts = np.cumsum(Ts) / np.cumsum(np.ones_like(Ts))
        Ts_mean = Ts.mean()
        plt.plot(Ts, color='red', alpha=0.3)
        plt.plot(av_Ts, color='red')
        plt.plot([0,len(Ts)-1], [Ts_mean]*2, color='red', linestyle='--')
        plt.plot([0,len(Ts)-1], [self.T]*2, color='black')
        if window is not None: plt.ylim(Ts_mean-window, Ts_mean+window)
        else: pass
        
    @property
    def temperature_plot(self,):
        print('average temperature:', self.average_temperature)
        self.plot_temperature_()

    def plot_energy_(self, window : float=None):
        us = np.array(self.u).flatten()
        av_us = np.cumsum(us) / np.cumsum(np.ones_like(us))
        us_mean = us.mean()
        plt.plot(us, color='blue', alpha=0.3)
        plt.plot(av_us, color='blue')
        plt.plot([0,len(us)-1], [us_mean]*2, color='black', linestyle='--')
        if window is not None: plt.ylim(us_mean-window, us_mean+window)
        else: pass

    @property
    def energy_plot(self,):
        print('average potential energy:', self.average_energy)
        self.plot_energy_()

    def plot_volume_(self, window : float=None):
        Vs = np.linalg.det(np.array(self.boxes)).flatten()
        av_Vs = np.cumsum(Vs) / np.cumsum(np.ones_like(Vs))
        Vs_mean = Vs.mean()
        plt.plot(Vs, color='green', alpha=0.3)
        plt.plot(av_Vs, color='green')
        plt.plot([0,len(Vs)-1], [Vs_mean]*2, color='black', linestyle='--')
        if window is not None: plt.ylim(Vs_mean-window, Vs_mean+window)
        else: pass

    @property
    def volume_plot(self,):
        print('initial volume:', np.linalg.det(self.b0))
        print('average volume:', self.average_volume)
        self.plot_volume_()

    @property
    def box_lengths_plot(self,):
        # x,y,z components of box vectors a,b,c respectively
        [plt.plot(self.boxes[:,i,i]) for i in range(3)]
        plt.plot([0,self.n_frames_saved],[self.PME_cutoff*2]*2, color='black', linestyle='--')
        
    def index_frame_average_box_othorhombic_case_(self,):
        # can use if barostat chosen was 0 or 1. [TODO: implement general case (include angles) as for form II]
        a,b,c = [self.boxes[:,i,i] for i in range(3)]
        self._sq_distance_from_average_shape = (a-a.mean())**2 + (b-b.mean())**2 + (c-c.mean())**2
        self._sq_distance_from_average_volume = (a*b*c - (a*b*c).mean())**2
        return np.argmin(self._sq_distance_from_average_shape), np.argmin(self._sq_distance_from_average_volume)
    
    @property
    def box_shapes(self,):
        if self.boxes is None:
            return cell_lengths_and_angles_(self.b0)
        else: 
            return [cell_lengths_and_angles_(self.b0), cell_lengths_and_angles_(self.boxes)]

    @property
    def partial_charges_mol(self,):
        if self.FF_name == 'TIP4P': n_atoms_mol = 4
        else:                       n_atoms_mol = int(self.n_atoms_mol)

        nonbonded = [f for f in self.system.getForces() if isinstance(f, mm.NonbondedForce)][0]
        charges = []
        for i in range(self.system.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charges.append(charge._value)
        charges = np.array(charges)
        charges_mol = charges[:n_atoms_mol]
        if not all([np.abs(charges[i*n_atoms_mol:(i+1)*n_atoms_mol] - charges_mol).sum() < 1e-5 for i in range(self.n_mol)]):
            print('!! not all molecules have same charge, this should not happen')
        else: pass
        average_charge = charges_mol.sum()
        if np.abs(average_charge) > 1e-5: print('!! net charge of moelcule is:',average_charge)
        else: print('net charge of molecule is neutral')
        return charges_mol
    
    @property
    def box_line(self, key='CRYST1'):
        '''
        only useful for the rough save_coordiantes_as_pdb_, not used otherwise.
        '''
        file = open(self.PDB,'r')
        output = None
        for line in file:
            if key in line:
                output = str(line)
            else: pass
        file.close()
        if output is None: 
            print('PDB file here may not have a box, or "key" can be changed')
            return ''
        else: return output
        
    def load_structures_with_mdtraj_(self, r, b=None):
        # r : (N,3) or (m,N,3)
        # b : None or (m,3,3)
        shape = r.shape
        if len(shape) == 2:
            n_frames = 1
            r = np.array(r[np.newaxis,...])
        else:
            n_frames = shape[0]
            assert len(shape) == 3

        if b is None:
            b = np.array([self._current_b_]*n_frames)
        else:
            b_shape = b.shape
            if n_frames==1 and len(b_shape) == 2:
                b = np.array(b[np.newaxis,...])
            else:
                assert len(b_shape) == 3
                assert b_shape[0] == n_frames

        tr = copy.deepcopy(self.traj)
        tr.xyz = r
        tr.unitcell_vectors = b
        tr.time = np.arange(n_frames)*self.timestep_ps
        return tr

    def save_gro_(self, r, name:str, b=None):
        # saves box for each frams.
        if name[-4:] == '.gro': pass
        else: name += '.gro'
        self.load_structures_with_mdtraj_(r=r, b=b).save_gro(name) 
        print('saved:',name)

    def save_pdb_(self, r, name:str, b=None):
        # save_pdb does not save box for each frams ?
        if name[-4:] == '.pdb': pass
        else: name += '.pdb'
        self.load_structures_with_mdtraj_(r=r, b=b).save_pdb(name) 
        print('saved:',name)

    def save_xtc_(self, r, name:str, b=None, save_reference=True):
        # saves box for each frams.
        if name[-4:] == '.xtc': pass
        else: name += '.xtc'
        tr = self.load_structures_with_mdtraj_(r=r, b=b)
        tr.save_xtc(name) 
        print('saved:',name)
        if save_reference:
            ref = name[:-4]+'_ref.pdb'
            tr.xyz = tr.xyz[:1]
            tr.unitcell_vectors = tr.unitcell_vectors[:1]
            tr.time = tr.time[:1]
            tr.save_pdb(ref)
        else: pass

## ## 

def plot_simulation_info_(self:object, figsize=(10,10)):
    '''
    after running an NVT or NPT simulation using SingleComponent, this plots some of the information about the simulation as a function of time
    the plots include:
        temperature : red
        potential energy : blue
        volume : green
        box vector length and angles
        diagonal lengths of the box matrix

    useful for checking consistency (e.g., in NPT, does the supercell always relax to the same state and is this converged or needs to run longer)
    '''
    if self.NPT: fig, ax = plt.subplots(6, figsize=figsize)
    else:        fig, ax = plt.subplots(2, figsize=figsize)

    fig.suptitle('timescale of the simulation so far: '+str(self.timescale)+'ns ('+str(self.n_frames_saved)+' frames)', fontsize=20)

    k = 0
    simulation_temperatures = self.temperature
    n_frames = len(simulation_temperatures)
    cumulative_average_simulation_temperatures = cumulative_average_(simulation_temperatures)
    average_temperature = cumulative_average_simulation_temperatures[-1]
    ax[k].set_title('temperature / K'+' ;    (average temperature: '+str(average_temperature.round(3))+')')
    ax[k].plot(simulation_temperatures, color='red',alpha=0.5)
    ax[k].plot(cumulative_average_simulation_temperatures, color='red')
    ax[k].plot([0,n_frames], [average_temperature ]*2, linestyle='--', color='red')
    ax[k].scatter([0],[self.T], color='black')

    k = 1
    initial_energy = self.u_GPU_(self.r0[np.newaxis,...],self.b0[np.newaxis,...])[0]
    simulation_energies = self.u
    cumulative_average_simulation_energies = cumulative_average_(simulation_energies)
    average_energy = cumulative_average_simulation_energies[-1]
    ax[k].set_title('potential energy / kT'+' ;    (average energy: '+str(average_energy.round(3))+')')
    ax[k].plot(simulation_energies, color='blue',alpha=0.5)
    ax[k].plot(cumulative_average_simulation_energies, color='blue')
    ax[k].plot([0,n_frames], [initial_energy ]*2, linestyle='dotted', color='blue')
    ax[k].plot([0,n_frames], [average_energy ]*2, linestyle='--', color='blue')

    if self.NPT:
        k = 2
        initial_volume = np.linalg.det(self.b0)
        simulation_volumes = np.linalg.det(self.boxes)
        cumulative_average_simulation_volumes = cumulative_average_(simulation_volumes)
        average_volume = cumulative_average_simulation_volumes[-1]
        n_frames = len(simulation_energies)
        ax[k].set_title('volume / (nm)^3'+' ;    (average volume: '+str(average_volume.round(3))+')')
        ax[k].plot(simulation_volumes, color='green',alpha=0.5)
        ax[k].plot(cumulative_average_simulation_volumes, color='green')
        ax[k].plot([0,n_frames], [initial_volume ]*2, linestyle='dotted', color='green')
        ax[k].plot([0,n_frames], [average_volume ]*2, linestyle='--', color='green')

        err_volume = np.abs(simulation_volumes-average_volume)

        k = 3
        initial_box_shape, simulation_box_shapes = self.box_shapes
        cumulative_average_box_shapes = np.stack([cumulative_average_(simulation_box_shapes[:,j]) for j in range(6)],axis=-1)
        average_box_shape = cumulative_average_box_shapes[-1]
        ax[k].set_title('box lengths (a,b,c) / nm'+' ;    (average box lengths: '+' '.join([str(x) for x in average_box_shape[:3].round(3)])+')')
        ax[k].plot(simulation_box_shapes[:,:3],alpha=0.5)
        show1 = np.array([initial_box_shape[:3]]*2)
        show2 = np.array([average_box_shape[:3]]*2)
        for i in range(3):
            ax[k].plot(cumulative_average_box_shapes[:,i],color='C'+str(i))
            ax[k].plot([0,n_frames],show1[:,i],color='C'+str(i), linestyle='dotted',)
            ax[k].plot([0,n_frames],show2[:,i],color='C'+str(i), linestyle='--')

        err_box_lengths = np.abs(average_box_shape[np.newaxis,:3] - simulation_box_shapes[:,:3])

        k = 4
        simulation_box_lengths_orth = np.stack([self.boxes[:,i,i] for i in range(3)], axis=-1)
        cumulative_average_box_lengths_orth = np.stack([cumulative_average_(simulation_box_lengths_orth[:,j]) for j in range(3)],axis=-1)
        average_box_lengths_orth = cumulative_average_box_lengths_orth[-1]
        initial_box_lengths_orth = simulation_box_lengths_orth[0]
        ax[k].set_title('box lengths under othorhombic wrapping (x,y,z componets of [a,b,c]) / nm \n (simulation cannot run if any curve touches the black region)'
                        +'\n (average: '+' '.join([str(x) for x in average_box_lengths_orth.round(3)])+')'
                        )
        ax[k].plot(simulation_box_lengths_orth,alpha=0.5)
        show1 = np.array([initial_box_lengths_orth]*2)
        show2 = np.array([average_box_lengths_orth]*2)
        for i in range(3):
            ax[k].plot(cumulative_average_box_lengths_orth[:,i],color='C'+str(i))
            ax[k].plot([0,n_frames],show1[:,i],color='C'+str(i), linestyle='dotted',)
            ax[k].plot([0,n_frames],show2[:,i],color='C'+str(i), linestyle='--')

        ax[k].plot([0,n_frames], [self.PME_cutoff*2]*2, color='black')
        ax[k].fill_between([0,n_frames],[self.PME_cutoff*2-0.1]*2,[self.PME_cutoff*2]*2, color='black')

        err_box_lengths_orth = np.abs(average_box_lengths_orth[np.newaxis,:] - simulation_box_lengths_orth)

        k = 5
        ax[k].set_title('box angles (bc,ac,ab) / degrees'+' ;    (average box angles: '+' '.join([str(x) for x in average_box_shape[3:].round(3)])+')')
        ax[k].plot(simulation_box_shapes[:,3:],alpha=0.5)
        show1 = np.array([initial_box_shape[3:]]*2)
        show2 = np.array([average_box_shape[3:]]*2)
        for i in range(3):
            ax[k].plot(cumulative_average_box_shapes[:,i+3],color='C'+str(i))
            ax[k].plot([0,n_frames],show1[:,i],color='C'+str(i), linestyle='dotted',)
            ax[k].plot([0,n_frames],show2[:,i],color='C'+str(i), linestyle='--')

        err_box_angles = np.abs(average_box_shape[np.newaxis,3:] - simulation_box_shapes[:,3:])

        output = [err_volume, err_box_lengths, err_box_lengths_orth, err_box_angles]
    else: output = None

    plt.tight_layout()
    plt.show()

    return output

def cell_lengths_and_angles_(b, radians=False):
    ''' matching mdtraj, could be done there instead
    Inputs:
        b : (m,3,3) boxes (box-vectors are the rows; axis=-2)
        radians: default False
    Outputs:
        lengths_and_angles : m * [[a,b,c], [bc,ac,ab]]
    '''
    norm_ = lambda x : np.linalg.norm(x, axis=-1, keepdims=True)
    unit_ = lambda x : x / norm_(x)
    def get_angle_(v1,v2, radians=True):
        angle = np.arccos(np.clip(np.sum(unit_(v1)*unit_(v2), axis=-1), -1.0, 1.0))
        if radians: return angle
        else: return angle*180/np.pi

    b = np.array(b)
    
    lengths = np.concatenate([norm_(b[...,0,:]),norm_(b[...,1,:]),norm_(b[...,2,:])], axis=-1)
    
    angles = np.stack([
    get_angle_(b[...,1,:],b[...,2,:], radians=radians),
    get_angle_(b[...,2,:],b[...,0,:], radians=radians),
    get_angle_(b[...,0,:],b[...,1,:], radians=radians)
    ], axis=-1)
    
    lengths_and_angles = np.concatenate([lengths,angles],axis=-1)
    
    return lengths_and_angles # [[a,b,c], [bc,ac,ab]]

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def save_gro_as_pdb_(GRO:str, PDB:str=None):
    path_and_file = GRO[:-4]
    if PDB is None: PDB = path_and_file+'.pdb'
    else: pass
    try:
        mdtraj.load(GRO).save_pdb(PDB)
        used = 'mdtraj'
    except:
        import MDAnalysis as mda
        universe = mda.Universe(GRO)
        with mda.Writer(PDB) as pdb:
            pdb.write(universe)
        used = 'MDAnalysis'
    print('saved',PDB,'using',used)

def PDB_to_xyz_(PDB:str):
    return mdtraj.load(PDB,PDB).xyz[0]

def PDB_to_box_(PDB:str):
    return mdtraj.load(PDB,PDB).unitcell_vectors[0]

def box_to_lengths_angles_(b):
    ''' b : (3,3) or (m,3,3) ; box or boxes '''
    if len(b.shape) == 3: return np.stack([mdtraj.utils.box_vectors_to_lengths_and_angles(*_b) for _b in b])
    else:                 return           mdtraj.utils.box_vectors_to_lengths_and_angles(*b)
        
def lengths_angles_to_box_(x):
    ''' x : (6) or (m,6) ; lengths and angles of one or more boxes '''
    if len(x.shape) == 2: return np.stack([mdtraj.utils.lengths_and_angles_to_box_vectors(*_x) for _x in x])
    else:                 return           mdtraj.utils.lengths_and_angles_to_box_vectors(*x)

def get_index_average_box_automatic_(boxes,
                                     n_bins = 30,
                                     rules =  ['av']*3 + ['max_prob']*3, # 'max_prob' for angles incase that type of disconnection present
                                     verbose = False,
                                    ):

    set_of_rules = {'max_prob': lambda h, ax : ax[np.argmax(h)],            # not the best incase n_bins not best
                    'av'   : lambda h, ax : (ax*h).sum(),                   # the most correct
                    'min'  : lambda h, ax : np.min(ax[np.where(h>0.0)[0]]), # can use 'min' for one of the box lengths to maybe prevent rare events that later hamper ergodicity in nvt
                   }
    rules = [set_of_rules[x] for x in rules]
    
    ''' 
    a simple way to find the most likely box is to just find peaks in histogram: this is automated here
    Inputs:
        boxes  : (n_frames, 3, 3) shaped array of boxes during a simulation, or any npt data
        n_bins : int 
            number of bins for six 1D histograms that are involved
    '''
    assert len(boxes.shape) == 3
    # x : (n_frames, 6) box lengths and angles
    x = box_to_lengths_angles_(boxes)
    
    def peak_finder_(x, i):
        h,ax = np.histogram(x, bins=n_bins, density=True)
        h /= h.sum()
        ax = ax[1:]-0.5*(ax[1]-ax[0])
        return rules[i](h, ax) #ax[np.argmax(h)]
    # x_av : average according to peaks of the 6 marginals histograms, but does not exist in the data (boxes)
    x_av = np.array([peak_finder_(x[...,i], i) for i in range(6)])
    # err : distance between x and x_av
    err = np.abs(x_av - x)
    # err : standardised distance (all 6 marginals equally important)
    err /= err.max(0, keepdims=True)
    # err : err for all 6 marginals
    err = err.sum(-1)
    # index : index of frame closes to average box according to this method
    index = np.argmin(err)
    
    def plot_box_lengths_angles_histograms_(boxes, b0 = None, b1=None):
        cell_NPT = cell_lengths_and_angles_(boxes)
        if b0 is None: b0 = np.array(boxes[0])
        else: b0 = np.array(b0).reshape([3,3])
        if b1 is None: b1 = np.array(b0)
        else: b1 = np.array(b1).reshape([3,3])
    
        cell_in = cell_lengths_and_angles_(b0)
        cell_out = cell_lengths_and_angles_(b1)
        
        fig, ax = plt.subplots(1,2, figsize=(8,2), dpi=100)
        Max0 = max([
            plot_1D_histogram_(cell_NPT[:,0], bins=30, ax=ax[0], return_max_y=True, color='C0'),
            plot_1D_histogram_(cell_NPT[:,1], bins=30, ax=ax[0], return_max_y=True, color='C1'),
            plot_1D_histogram_(cell_NPT[:,2], bins=30, ax=ax[0], return_max_y=True, color='C2'),
            ])
        for i in range(0,3):
            ax[0].plot([cell_in[i]]*2, [0,Max0], color='C'+str(i), linestyle='dotted', linewidth=1.5)
            ax[0].plot([cell_out[i]]*2, [0, Max0], color='C'+str(i), linewidth=1.5)
        Max1 = max([
            plot_1D_histogram_(cell_NPT[:,3], bins=30, ax=ax[1], return_max_y=True, color='C0'),
            plot_1D_histogram_(cell_NPT[:,4], bins=30, ax=ax[1], return_max_y=True, color='C1'),
            plot_1D_histogram_(cell_NPT[:,5], bins=30, ax=ax[1], return_max_y=True, color='C2'),
            ])
        for i in range(3,6):
            ax[1].plot([cell_in[i]]*2, [0,Max1], color='C'+str(i-3), linestyle='dotted', linewidth=1.5)
            ax[1].plot([cell_out[i]]*2, [0,Max1], color='C'+str(i-3), linewidth=1.5)
    
        ax[0].set_xlabel('box vector lengths / nm')
        ax[1].set_xlabel('box vector angles / degrees')
        
        plt.tight_layout()
        plt.show()

    if verbose:
        print('output = index =', index)
        plot_box_lengths_angles_histograms_(boxes, b0 = boxes[0], b1=boxes[index])
    else: pass
    
    return index

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def get_unitcell_stack_order_(b, n_mol_unitcell=1, top_n=None):
    """ 
    not sure if this is good
    
    tells how many times to stack the unitcell in each of the three directions
        while minimising surface area to volume ratio (more spherical ~~ more cubic)
        
    Inputs:
        b : (3,3) : box of the unit cell (NB: only the diagonal distances will be used)
        n_mol_unitcell : number of molecules in the unit cell
        top_n : how many results to include in output
    Output:
        res: dictionary {n_mol_supercell : number of times to stack unitcell in each of the three unitcell vector directions}
            res[n_mol_supercell of interest] --> input for the supercell_from_unitcell_ function
    """
    b = np.array(b).reshape([3,3])
    grid = np.arange(1,11)
    b00 = b[0,0]
    b11 = b[1,1]
    b22 = b[2,2]
    Grid = np.zeros([10,10,10,3])
    As = np.zeros([10,10,10])
    Vs  = np.zeros([10,10,10]) # np.einsum('i,j,k->ijk',b00*grid,b11*grid,b22*grid)
    for i in range(10):
        n0 = grid[i]
        l0 = n0*b00
        for j in range(10):
            n1 = grid[j]
            l1 = n1*b11
            for k in range(10):
                n2 = grid[k]
                l2 = n2*b22
                As[i,j,k] = 2*l0*l1 + 2*l1*l2 + 2*l2*l0
                Vs[i,j,k] = l0*l1*l2
                Grid[i,j,k] = np.array([n0,n1,n2])
    
    AoverV = As / Vs # surface area of supercell / volume of supercell
    Grid_flat = Grid.reshape([10**3,3])
    AoverV_flat = AoverV.flatten()

    res = Grid_flat[np.argsort(-AoverV_flat)]
    res = res.astype(np.int32).tolist()
    res = dict(zip(np.prod(res,axis=-1)*n_mol_unitcell, res))

    ##
    # prevent repeats, top results are better
    n_mol_seen = []
    _res = {}
    for n_mol in res.keys():
        if n_mol not in n_mol_seen:
            _res[n_mol] = res[n_mol]
            n_mol_seen.append(n_mol)
        else: pass
    res = _res
    # if n_mol is possible but not at the top, move to the top anyway
    _res = {}
    for n_mol in np.sort(n_mol_seen):
        _res[n_mol] = res[n_mol]
    res = _res
    ##

    if top_n is not None: res = dict(zip(list(res.keys())[:top_n], [res[n_mol] for n_mol in list(res.keys())[:top_n]]))
    else: pass
        
    return res

def supercell_from_unitcell_(PDB_unitcell : str, # .pdb
                             cell:list = [1,1,1],
                             save_output = True,
                            ):
    """ copy ideal unitcell along unitcell vectors to get ideal supercell of correct shape
    Inputs:
        PDB_unitcell : unitcell coordinates, must contain the header with unitcell lengths and angles
        cell : list of three integers (> 0) for how many copies to make along each unitcell vector direction
            [default : [1,1,1] : identity : no change]
        save_output : if True, saves the supercell
    Outputs:
        instance of mdtraj of the larger supercell for further modifications
    """
    def expand_mdtraj_(input_instance, n_copies):
        output_instance = input_instance
        for _ in range(n_copies-1):
            output_instance = mdtraj.Trajectory.stack(output_instance,input_instance)
        return output_instance
    
    assert min(cell) > 0
    assert all([type(x) == int for x in cell])
    
    n_unitcell_coppies = cell[0]*cell[1]*cell[2]
    
    traj = mdtraj.load(PDB_unitcell, top=PDB_unitcell)
    b = traj.unitcell_vectors[0]
    
    r = traj.xyz + np.einsum('oi,ij->oj',
                    joint_grid_from_marginal_grids_(np.arange(cell[0]),
                                                    np.arange(cell[1]),
                                                    np.arange(cell[2]),
                                                    flatten_output=True),
                    b)[:,np.newaxis,:]
    n_atoms = r.shape[1]
    r = r.reshape([n_unitcell_coppies*n_atoms,3])
    
    b = np.array(cell).reshape(3,1)*b # the input box itself enlarged accordingly
    
    traj = expand_mdtraj_(traj, n_copies=n_unitcell_coppies)
    
    traj.xyz = r[np.newaxis,...]
    traj.unitcell_vectors = b[np.newaxis,...]

    _PDB_unitcell = PDB_unitcell.split('/')[-1]
    path = '/'.join(PDB_unitcell.split('/')[:-1])
    PDB_supercell = path + '/' + _PDB_unitcell[:-4]+'_cell'+''.join([str(x) for x in cell])+_PDB_unitcell[-4:]
    if save_output:
        traj.save_pdb(PDB_supercell)
        print('saved',PDB_supercell)
    else: pass
    
    return traj, PDB_supercell

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def box_in_reduced_form_(box):
    ''' method copied from openmm sources
    Input:
        box : (3,3) box-vectors are rows
    Output:
        bool : True if the box is already in the reduced form
    '''
    b = np.array(box)

    if b[0,1] != 0.0 or b[0,2] != 0.0:
        #print("First periodic box vector must be parallel to x.")
        return True
    elif b[1,2] != 0.0:
        #print("Second periodic box vector must be in the x-y plane.")
        return True
    elif  any([b[0,0] <= 0.0, b[1,1] <= 0.0, b[2,2] <= 0.0,
        b[0,0] < 2*abs(b[1,0]),
        b[0,0] < 2*abs(b[2,0]), 
        b[1,1] < 2*abs(b[2,1])]) :
        #print("Periodic box vectors must be in reduced form.")
        return False
    else: #print('box ok')
        return True
    
def reducePeriodicBoxVectors_(box):
    ''' method copied from openmm sources
    Input:
        box : (3,3) box-vectors are rows
    Output:
        the single box converted to format compatible with OpenMM
    '''
    box = np.array(box)
    assert box.shape == (3,3)
    
    a,b,c = [*box]
    c = c - b*round(c[1]/b[1])
    c = c - a*round(c[0]/a[0])
    b = b - a*round(b[0]/a[0])

    return np.stack([a, b, c], axis=0)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def reorder_atoms_mol_(mol_pdb_fname, template_pdb_fname, output_pdb_fname):
    '''
    REF: https://gist.github.com/fabian-paul/abba9172d394dffb93624a710acbab16
    '''
    import rdkit.Chem
    from rdkit.Chem import rdmolfiles

    mol_to_transform = rdkit.Chem.rdmolfiles.MolFromPDBFile(mol_pdb_fname, removeHs=False)
    transform_order = list(rdmolfiles.CanonicalRankAtoms(mol_to_transform))

    mol_template = rdkit.Chem.rdmolfiles.MolFromPDBFile(template_pdb_fname, removeHs=False)
    template_order = list(rdmolfiles.CanonicalRankAtoms(mol_template))

    if len(template_order) != len(transform_order):
        raise RuntimeError('Number of atoms differs between template and molecule to transform.')

    i_transform_order = [int(i) for i in np.argsort(transform_order)]
    i_template_order =  [int(i) for i in np.argsort(template_order)]

    N_atoms = len(template_order)

    pos_to_transform = mol_to_transform.GetConformers()[0].GetPositions()
    lines = [None]*N_atoms
    for _, (otr, ote) in enumerate(zip(i_transform_order, i_template_order)):
        #print(mol_to_transform.GetAtoms()[otr].GetPDBResidueInfo().GetName(),
        #      mol_template.GetAtoms()[ote].GetPDBResidueInfo().GetName())
        pdb_entry_template = mol_template.GetAtoms()[ote].GetPDBResidueInfo()
        symbol = mol_template.GetAtoms()[ote].GetSymbol()
        lines[ote] = '{ATOM:<6}{serial_number:>5} {atom_name:<4}{alt_loc_indicator:<1}{res_name:<3} {chain_id:<1}{res_seq_number:>4}{insert_code:<1}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}'.format(
                    ATOM='ATOM',
                    serial_number=pdb_entry_template.GetSerialNumber(),
                    atom_name=pdb_entry_template.GetName(),
                    alt_loc_indicator=' ',
                    res_name=pdb_entry_template.GetResidueName(),
                    chain_id=pdb_entry_template.GetChainId(),
                    res_seq_number=pdb_entry_template.GetResidueNumber(),
                    insert_code=' ',
                    x= pos_to_transform[otr, 0],
                    y= pos_to_transform[otr, 1],
                    z= pos_to_transform[otr, 2],
                    occupancy=1.0,
                    temp_factor=0.0
                ) + '           '+symbol+'  '
        
    with open(output_pdb_fname, 'w') as f:
        for line in lines:
            if line is not None:
                print(line, file=f)

def validate_reorder_atoms_mol_(template_pdb_fname, output_pdb_fname):
    '''
    REF: https://gist.github.com/fabian-paul/abba9172d394dffb93624a710acbab16
    '''
    import rdkit.Chem
    import rdkit.Chem.rdPartialCharges
    
    mol_template = rdkit.Chem.rdmolfiles.MolFromPDBFile(template_pdb_fname, removeHs=False)
    finished_molecule = rdkit.Chem.rdmolfiles.MolFromPDBFile(output_pdb_fname, removeHs=False)
    N_atoms = finished_molecule.GetNumAtoms()
    for i in range(N_atoms):
        assert finished_molecule.GetAtoms()[i].GetAtomicNum() == mol_template.GetAtoms()[i].GetAtomicNum()
        assert finished_molecule.GetAtoms()[i].GetDegree() == mol_template.GetAtoms()[i].GetDegree()
        assert finished_molecule.GetAtoms()[i].GetTotalDegree() == mol_template.GetAtoms()[i].GetTotalDegree(), i
        assert finished_molecule.GetAtoms()[i].GetHybridization() == mol_template.GetAtoms()[i].GetHybridization(), i
        assert finished_molecule.GetAtoms()[i].GetFormalCharge() == mol_template.GetAtoms()[i].GetFormalCharge(), i
        assert finished_molecule.GetAtoms()[i].GetTotalValence() == mol_template.GetAtoms()[i].GetTotalValence(), i    
    rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges(finished_molecule, throwOnParamFailure=True)
    charges_finished = np.array([float(finished_molecule.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(N_atoms)])
    rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges(mol_template, throwOnParamFailure=True)
    charges_template = np.array([float(mol_template.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(N_atoms)])
    assert np.allclose(charges_finished, charges_template)
    print('Validation OK')

def reorder_atoms_unitcell_(PDB:str, PDB_ref:str, n_atoms_mol:int):
    ''' 

    Inputs:
        PDB     : supercell or unitcell of a molecule (e.g., to_reorder.pdb)
        PDB_ref : single molecule with intended order of atoms
        n_atoms_mol : number of atoms in the molecule, including all atoms
    Outputs:
        to_reorder_reordered.pdb : file with same xyz coordiantes as PDB, but atom order and names changed to match PDB_ref

    '''
    def split_(PDB, n_atoms_mol, ref=False):
        path_name = PDB.split('/')
        path = path_name[:-1]
        
        folder_needed = '/'.join(path+ ['temp'])
        assert os.path.exists(folder_needed), f'please create empty temp folder here: {folder_needed}'

        name = path_name[-1]
        traj = mdtraj.load(PDB,PDB)
        n_mol = traj.n_atoms//n_atoms_mol
        assert traj.n_atoms/n_atoms_mol == n_mol
        if ref: n = 1 ; ext = '_ref.pdb'
        else: n = n_mol ; ext = '.pdb'
        path_names_out = []
        for i in range(n):
            tr = copy.deepcopy(traj)
            tr = tr.atom_slice(np.arange(n_atoms_mol)+i*n_atoms_mol)
            name_out = 'temp/'+str(i)+ext
            path_name_out = '/'.join(path + [name_out])
            tr.save_pdb(path_name_out)
            path_names_out.append(path_name_out)
        return path_names_out, traj

    def expand_mdtraj_(input_instance, n_copies):
        output_instance = input_instance
        for _ in range(n_copies-1):
            output_instance = mdtraj.Trajectory.stack(output_instance,input_instance)
        return output_instance
        
    template_pdb_fname = split_(PDB_ref, n_atoms_mol=n_atoms_mol, ref=True)[0][0]
    mol_pdb_fnames, traj = split_(PDB, n_atoms_mol=n_atoms_mol, ref=False)
    for x in mol_pdb_fnames:
        reorder_atoms_mol_(x, template_pdb_fname, x)
    b = traj.unitcell_vectors
    traj = mdtraj.load(mol_pdb_fnames[0],mol_pdb_fnames[0])
    traj = expand_mdtraj_(traj, n_copies=len(mol_pdb_fnames))
    r = []
    for x in mol_pdb_fnames:
        tr = mdtraj.load(x,x)
        r.append(tr.xyz)

    traj.xyz = np.concatenate(r,axis=1)
    traj.unitcell_vectors = b
    output_name = PDB[:-4]+'_reordered.pdb'
    traj.save_pdb(output_name)
    print('saved:',output_name)
    return output_name

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def vectors_between_atoms_(r, b, n_images_search=1):
    ''' ALL vectors (not just those within the diagonal subspace of the box eye(3)*b)

    naive search for the shortest distanace vector between two atoms 
    
    Inputs:
        r : (N,3) single configuration (can be selection of any N atoms of interest)
        b : (3,3) box (box-vectors are the rows)
        n_images_search : number of periodic images to search for the shortest vector
            [if b very skewed can increase n_images_search; more expensive]
    Outputs:
        vs_out : (N,N,3) shortest vectors between all atoms (from rows to columns)
    '''
    r = np.array(r) # (N,3)
    b = np.array(b)[np.newaxis,...] # (1,3,3)

    r_to_v_ = lambda rA, rB : np.expand_dims(rB,axis=-3) - np.expand_dims(rA,axis=-2) # (1,N,3) - (N,1,3)
    
    grid = np.arange(-n_images_search, n_images_search+1).astype(np.float32)
    
    rs = [r]
    for sign_a in grid:
        for sign_b in grid:
            for sign_c in grid:
                if any((sign_a,sign_b,sign_c)):
                    rs.append(r + sign_a*b[:,0] + sign_b*b[:,1] + sign_c*b[:,2])
                else:
                    pass

    len_rs = len(rs) # 27 (if n_images_search=1)

    vs = np.stack([r_to_v_(rs[0],rs[i]) for i in range(len_rs)],axis=0) # (27,N,N,3)
    ds = np.linalg.norm(vs, axis=-1)                                    # (27,N,N)
    d = np.min(ds, axis=0)                                              # (N,N) # distances 
    inds = np.expand_dims(np.argmin(ds,axis=0), axis=-1)                # (N,N,1)

    vs_out = 0.0
    for i in range(len_rs):
        vs_out += vs[i] * np.where(inds==i,1.0,0.0)

    return vs_out # (N,N,3) # = v_ij ; i->j 

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def change_box_(PDB, n_atoms_mol, make_orthorhombic=False, save_output=True, traj = None):
    '''
    dont remember
    '''
    def wrap_points_1box_(Ri,  # (... 3), 
                          box, # (3, 3) # rows
                         ):
        string = '...l,lm->...m'
        return np.einsum(string, np.mod(np.einsum(string, Ri, np.linalg.inv(box)), 1.0), box)
    
    if traj is not None: pass
    else: traj = mdtraj.load(PDB, top=PDB)
    
    r = traj.xyz[0] # (N,3)
    N = r.shape[0]
    n_mol = N // n_atoms_mol
    assert n_mol == N / n_atoms_mol
    b = traj.unitcell_vectors[0]

    if make_orthorhombic:
        b *= np.eye(3)
        changes = '_orthorhombic_cell'
    else:
        changes = '_reduced_cell'
        if not box_in_reduced_form_(b): b = reducePeriodicBoxVectors_(b)
        else: pass

    r = r.reshape([n_mol, n_atoms_mol, 3]) 
    rO = r[:,0:1,:]
    r = r - rO + wrap_points_1box_(rO,b) # (N,1,3)
    r = r.reshape([N, 3])

    traj.xyz = r[np.newaxis,...]
    traj.unitcell_vectors = b[np.newaxis,...]

    new_PDB = PDB[:-4]+changes+PDB[-4:]
    if save_output:
        traj.save_pdb(new_PDB)
        print('saved new_PDB:',new_PDB )
    else: pass

    return traj, new_PDB

def remove_clashes_(PDB_unitcell : str, tol = 0.001):
    '''
    no needed in most cases?
    '''
    ''' preparing initial structure
    ! rough script
    TODO: test further to check that whole molecules come out (not removing too many atoms...)
    [good to check the output in VMD to confirm that packing is correct and molecules are intact]
    '''
    # PDB_unitcell : str : path
    # tol : numerical tolerance (nm) to call atoms that are too close a clash 
    ##

    def wrap_points_1box_(Ri,  # (... 3), 
                          box, # (3, 3) # rows
                         ):
        string = '...l,lm->...m'
        return np.einsum(string, np.mod(np.einsum(string, Ri, np.linalg.inv(box)), 1.0), box)
    
    def minimum_image_othorhombic_(r, # (m,N,3)
                                   b, # (m,3,3)
                                  ):
    
        s = np.einsum('oik,okl->oil', r, np.linalg.inv(b))
    
        sv = np.expand_dims(s,axis=-2) - np.expand_dims(s,axis=-3) # s[:,np.newaxis,:]-s[np.newaxis,:,:]
    
        v = np.einsum('oijk,okl->oijl', sv - np.round(sv), b) # !! wrong above cut_off if not othorhombic
        
        d = np.linalg.norm(v, axis=-1)                               # !! wrong above cut_off if not othorhombic
    
        diag = np.stack([b[...,0:1,0:1],b[...,1:2,1:2],b[...,2:3,2:3]],axis=-1) # (m, 1, 1, 3)
        cut_off = np.min(diag, axis=-1)*0.5                              # (m, 1, 1)
    
        d_mask = np.where(d<=cut_off, 1.0, 0.0)
        #d_mask = tf.cast(d_mask,dtype=tf.float64)
        # ^ 1 == correct.
    
        return d, d_mask, v, cut_off[...,0] # (m,N,N), (m,N,N,3), (m,N,N), (m,1)
    
    ##
    traj = mdtraj.load(PDB_unitcell, top=PDB_unitcell)
    b = traj.unitcell_vectors[0]
    b_orth = np.array(b*np.eye(3))
    xyz = traj.xyz[0]
    n_atoms_before = traj.n_atoms
    d = minimum_image_othorhombic_(wrap_points_1box_(xyz,b_orth)[np.newaxis,...], b_orth[np.newaxis,...])[0][0]
    
    ## clashes inside the unicell:
    contact = np.where(np.abs(d-np.eye(len(d)))<tol,1.,0.)
    contact *= np.triu(np.ones_like(contact)) 
    inds_keep = np.where(contact.sum(0)<0.5)[0]
    ##

    traj = traj.atom_slice(inds_keep)
    
    ## periodic image clashes: (x - x_image < tol clashes removed here)
    import scipy as sp

    xyz = traj.xyz[0]
    remove = []

    # unitcell vectors (a,b,c) are rows of the box matrix (b) given by mdtraj
    v0 =  b[0:1]
    v1 =  b[1:2]
    v2 =  b[2:3]

    vectors = [v0, v1, v2, v0+v1, v1+v2, v2+v0, v0+v1+v2,
               v0-v1, v0-v2, v1-v0, v1-v2, v2-v1,
               ]
    print('!! warning : this method is not robust, please check .pdb outputs from this method in VMD before using.')

    for v in vectors:
        d = sp.spatial.distance.cdist(xyz, xyz + v, metric='euclidean') + np.eye(traj.n_atoms)
        remove += np.where(d.min(0)<tol)[0].tolist()

    inds_keep = list(set(np.arange(traj.n_atoms).tolist()) - set(remove))
    traj = traj.atom_slice(inds_keep)
    ##

    if n_atoms_before == traj.n_atoms:
        print('no periodic clashes were found in file')
    else:
        print('! found periodic clashes in file. removing:')
        print('# atoms before slice:',n_atoms_before )
        print('# atoms after slice:',traj.n_atoms)
    return traj

def rename_atoms_(PDB, n_atoms_mol):
    print('no rename')
    '''
    elements = dict(zip(['C','O','N','H','F','S'], [0]*6))

    file_in = open(PDB,'r')
    lines_out = []
    a = 0
    for line in file_in:
        element = line[77:78]
        if element in ['',' '] :
            new_line = str(line)
        else:
            if elements[element] == 0:
                insert = element
            else:
                insert = element + str(elements[element])
            if insert =='O1': insert = 'OXT'
            else: pass
            new_line = line[:12] + insert + ' '*(5 - len(insert)) + line[17:]
            elements[element] =  elements[element] + 1
            a += 1
        if a == n_atoms_mol:
            elements = dict(zip(['C','O','N','H','F','S'], [0]*6))
        else: pass
        lines_out.append(new_line)
    
    file_in.close()

    file_out = open(PDB,'w')
    for line in lines_out:
        file_out.write(line)
    file_out.close()
    '''
    
def process_mercury_output_(PDB, n_atoms_mol : int,
                            single_mol = False,
                            custom_path_name = None,
                            tol = 0.001):
    ''' preparing initial structure
    '''
    if single_mol: _name = '_mol'
    else: _name = '_unitcell'
    path_name = PDB.split('/')
    path = path_name[:-1]
    name = path_name[-1]
    if custom_path_name is not None: 
        path_name_out = custom_path_name
    else: 
        name_out = name[:-4] + _name + name[-4:]
        path_name_out = '/'.join(path + [name_out])

    traj = remove_clashes_(PDB, tol = tol)
    if single_mol:
        traj = traj.atom_slice(np.arange(n_atoms_mol))
    else: pass

    traj.save_pdb(path_name_out)
    rename_atoms_(path_name_out, n_atoms_mol=n_atoms_mol)
    print('('+str(traj.n_atoms//n_atoms_mol)+','+str(n_atoms_mol)+',3)')
    print('saved:', path_name_out)
    return traj, path_name_out

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

def extract_subcell_from_supercell_(n_mol_in:int, n_atoms_mol:int,
    r_in, b_in, cell_in:list, cell_out:list, ind_rO:int,
    ):
    ''' rough method, was not used!

    cuts out a smaller cell from a bigger cell
    inputs (r_in) need to first go (successfully) through tidy_crystal_xyz_
    outputs (r_out, b_out); ouput needs to be energy-minimised before being used

    '''
    def check_shape_(x):
        x = np.array(x)
        shape = len(x.shape)
        assert shape in [2,3]
        if len(x.shape) == 3: pass
        else: x = x[np.newaxis,...]
        return x
    def dot_(Ri, mat):
        string = 'oai,oij->oaj'
        return np.einsum(string, Ri, mat)

    r = reshape_to_molecules_np_(check_shape_(r_in), n_atoms_in_molecule=n_atoms_mol, n_molecules=n_mol_in)
    rO = np.array(r[:,:,ind_rO])
    b = check_shape_(b_in)

    n_frames = r.shape[0]

    cell_out = np.array(cell_out)
    cell_in = np.array(cell_in)
    cell_fraction = cell_out / cell_in

    n_mol_out_expected = (np.prod( cell_out) / np.prod(cell_in)) * n_mol_in
    print('n_mol_out expected:',n_mol_out_expected)

    r_out = []
    b_out = []
    inds_out_ref = None ; warn = False ; frames_caution = []
    for i in range(n_frames):
        x = np.mod(np.einsum('ai,ij->aj', rO[i], np.linalg.inv(b[i])), 1.0)
        inds_a = np.where(x[...,0]<= cell_fraction[0])[0]
        inds_b = np.where(x[...,1]<= cell_fraction[1])[0]
        inds_c = np.where(x[...,2]<= cell_fraction[2])[0]

        # want molecules that are in the first 'sub_cell' along all 3 box vectors
        inds_out = list(set(inds_a) & set(inds_b) & set(inds_c)) # intersection of 3 lists of indices
        if inds_out_ref is None:
            inds_out_ref = inds_out
        else:
            if len(set(inds_out) - set(inds_out_ref) | set(inds_out_ref) - set(inds_out)) > 0:
                frames_caution.append(i)
                inds_out = inds_out_ref # using indices 
                warn = True
            else: pass
        n_mol_out = len(inds_out)
        
        r_out.append(reshape_to_atoms_np_(np.take(r[i:i+1], inds_out, axis=1), n_atoms_in_molecule=n_atoms_mol, n_molecules=n_mol_out)[0])
        b_out.append(b[i] * cell_fraction[np.newaxis,:])

    r_out = np.stack(r_out,axis=0)
    b_out = np.stack(b_out,axis=0)
    print('n_mol_out =', n_mol_out)
    if n_mol_out != n_mol_out_expected:
        print('!! this simple method failed to work on this data')
    else: pass
    if warn:
        print('! recommending to check output in VMD')
    else: pass
    return r_out, b_out

""" 

# might do this properly later to compare with this LangevinMiddle:
# setting \Theta_{\text{friction}} to 0 gives VV in this, and otherwise
# it converges to the correct cumulative average T, even at high timestep.
'''
\begin{align}
\nonumber&\boldsymbol{r}_{i}(t+\Delta t)=\boldsymbol{r}_{i}(t)+\left(
\dot{\boldsymbol{r}}_{i}(t) + 
\frac{\Delta t}{2 \text{m}_{i}}\boldsymbol{F}_{i}(\boldsymbol{r}(t))
\right)\Delta t\\
\nonumber&\dot{\boldsymbol{r}}_{i}(t+\Delta t)=\left(
\dot{\boldsymbol{r}}_{i}(t) + 
\frac{\Delta t}{2 \text{m}_{i}}\boldsymbol{F}_{i}(\boldsymbol{r}(t))
\right)\alpha +
\boldsymbol{z}_{i}  +
\frac{\Delta t}{2 \text{m}_{i}}\boldsymbol{F}_{i}(\boldsymbol{r}(t+\Delta t))\\
&\text{where} 
\;\;
\alpha=\exp{(-\Theta_{\text{friction}}\Delta t)} 
\;\;\text{and} \;\;
\boldsymbol{z}_{i} \sim \mathcal{N}\left(\boldsymbol{0},\left(\frac{k_{B}T(1-\alpha^{2})}{\text{m}_{i}}\right)\mathbf{I}\right)\label{Eq:LangevinMiddleIntegrator}
\end{align}
'''

def CustomIntegrator_(temperature = 300*unit.kelvin,
                      friction = 5.0/unit.picosecond,
                      timestep = 0.002*unit.picoseconds,
                      mu_mass : np.ndarray = None,
                      VV = False,
                     ):
    '''
    was not used at any point, safer not to use
    '''
    
    kB = CONST_kB * unit.kilojoule / (unit.kelvin*unit.mole)

    if mu_mass is None:
        using_COM_constraint = False
    else:
        '''
        trying to keep centre off mass (COM) over arbitrary subset of atoms completely fixed
            mu : (N,)   ; can be a sparse or not sparse array of currect masses (non-zero for atoms for which COM is to be removed)
            mass : (N,) ; is the array of correct masses (no zeros)
        '''
        using_COM_constraint = True
        mu, mass = mu_mass
        mu = np.array(mu.flatten()) ; mu /= mu.sum() ; N = len(mu)
        MU = np.stack([mu]*3,axis=-1)

    integrator = mm.CustomIntegrator(timestep)

    if using_COM_constraint:
        integrator.addPerDofVariable("mask", 0.0)
        integrator.setPerDofVariableByName("mask", MU)
        integrator.addPerDofVariable("masked", 0.0)
        integrator.setPerDofVariableByName("masked", MU)
        integrator.addGlobalVariable("vCOMx", 0.0)
        integrator.addGlobalVariable("vCOMy", 0.0)
        integrator.addGlobalVariable("vCOMz", 0.0)
    else: pass

    if VV:

        ''' 
        not using because unstable (hydrogens break) in alchemical states at fast timestep compared to LangevinMiddle

        Velocity-Verlet (REF : openmm-master\openmmapi\include\openmm\CustomIntegrator.h)
        with 
        Bussi Donadio Parrinello thermostat (REF : arXiv:0803.4060v1) 
        '''
        # friction : higher -> (more thermal coupling, less smooth dynamics, faster convergence to temperature)

        integrator.addGlobalVariable("A", np.exp(-friction*timestep))
        integrator.addGlobalVariable("B", np.exp(-0.5*friction*timestep))
        integrator.addGlobalVariable("C", 0.5*kB*temperature*(1.0 - np.exp(-friction*timestep)))
        integrator.addGlobalVariable("KE", 0.0)

        integrator.addPerDofVariable("force_term", 0.0)
        integrator.setPerDofVariableByName("force_term", np.array([[1.,1.,1.]]*N))

        if using_COM_constraint:
            N = len(MU)
            mask = np.array([[1.,1.,1.]]*N)
            mask[N-1, :] = 0.0
            # TODO: if using other constraints, mask more elements starting from the end of the mask array
        else: pass

        mask_select_1 = np.array(mask)*0.0
        mask_select_1[0,0] = 1.0
        mask_select_others = np.array(mask)
        mask_select_others[0,0] = 0.0

        integrator.addPerDofVariable("mask_select_1", 0.0)
        integrator.setPerDofVariableByName("mask_select_1", mask_select_1)
        integrator.addPerDofVariable("mask_select_others", 0.0)
        integrator.setPerDofVariableByName("mask_select_others", mask_select_others)
        integrator.addPerDofVariable("noise", 0.0)
        integrator.setPerDofVariableByName("noise", mask)
        integrator.addGlobalVariable("noise1", 0.0)
        integrator.addGlobalVariable("sum_squared_noise_others", 0.0)
        integrator.addGlobalVariable("alpha", 0.0)
        integrator.addGlobalVariable("rand", 0.0)
        integrator.addGlobalVariable("sel", 0.0)
        integrator.addPerDofVariable("x1", 0)
        integrator.addUpdateContextState()

        if using_COM_constraint:
            integrator.addComputePerDof("masked", "v*mask")
            integrator.addComputeSum("vCOMx", "masked*vector(1, 0, 0)")
            integrator.addComputeSum("vCOMy", "masked*vector(0, 1, 0)")
            integrator.addComputeSum("vCOMz", "masked*vector(0, 0, 1)")
            integrator.addComputePerDof("v", "v - vector(vCOMx,vCOMy,vCOMz)")
        else: pass

        integrator.addComputeSum("KE", "(m*v^2)/2") # 0.5*v*v*m DOFs: 3(N-1) if using_COM_constraint, else 3N.
        integrator.addComputePerDof("noise", "gaussian")
        integrator.addComputeSum("sum_squared_noise_others", "mask_select_others*noise*noise")
        integrator.addComputeSum("noise1", "mask_select_1*noise")
        integrator.addComputeGlobal("alpha", "A + (C/KE)*(noise1*noise1 + sum_squared_noise_others) + 2.0*noise1*B*sqrt(C/KE)")
        integrator.addComputeGlobal("alpha", "sqrt(alpha)")
        # regular intervals would be better, so not using: [collision on every timestep]
        #integrator.addComputeGlobal("rand", "uniform-0.04") # 0.04 = 1/25 probability
        #integrator.addComputeGlobal("sel", "min(1,max(0,rand))/rand")
        #integrator.addComputeGlobal("alpha", "(alpha-1)*sel + max(sel,1)") # 1 or alpha (alpha 1/25 of the time)
        integrator.addComputePerDof("v", "v*alpha") # rescaling does not change COM when COM=0 
        integrator.addComputePerDof("force_term", "0.5*dt*f/m") # force on 

        if using_COM_constraint:
            integrator.addComputePerDof("masked", "force_term*mask")
            integrator.addComputeSum("vCOMx", "masked*vector(1, 0, 0)")
            integrator.addComputeSum("vCOMy", "masked*vector(0, 1, 0)")
            integrator.addComputeSum("vCOMz", "masked*vector(0, 0, 1)")
            integrator.addComputePerDof("force_term", "force_term - vector(vCOMx,vCOMy,vCOMz)")
        else: pass

        integrator.addComputePerDof("v", "v+force_term")
        integrator.addComputePerDof("x", "x+dt*v")
        integrator.addComputePerDof("force_term", "0.5*dt*f/m")
        
        if using_COM_constraint:
            integrator.addComputePerDof("masked", "force_term*mask")
            integrator.addComputeSum("vCOMx", "masked*vector(1, 0, 0)")
            integrator.addComputeSum("vCOMy", "masked*vector(0, 1, 0)")
            integrator.addComputeSum("vCOMz", "masked*vector(0, 0, 1)")
            integrator.addComputePerDof("force_term", "force_term - vector(vCOMx,vCOMy,vCOMz)")
        else: pass

        integrator.addComputePerDof("v", "v+force_term")
        integrator.addComputePerDof("x1", "x")
    else:

        '''
        not using because unsure about the math.
        '''
        
        mass = np.array(mass.flatten())
        MASS = np.stack([mass]*3,axis=-1)
        integrator.addPerDofVariable("mass", 0.0)
        integrator.setPerDofVariableByName("mass", MASS)

        SD = 1/np.sqrt(MASS)
        if using_COM_constraint:
            VAR = SD**2
            SD = SD * ( ( VAR / (VAR + np.sum( (SD*MU)**2, axis=0,keepdims=True)) )**0.5 )
        else: pass
        integrator.addPerDofVariable("SD", 0.0)
        integrator.setPerDofVariableByName("SD", SD)

        alpha = np.exp(-friction*timestep)
        beta = np.sqrt(1 - alpha**2)
        integrator.addGlobalVariable("alpha", alpha)
        integrator.addGlobalVariable("beta", beta)
        integrator.addGlobalVariable("beta_sqrt_kT", beta*((kB._value*temperature._value)**0.5))
        integrator.addUpdateContextState()
        integrator.addPerDofVariable("F_over_m", 0.0)
        integrator.setPerDofVariableByName("F_over_m", MASS)
        integrator.addPerDofVariable("noise", 0.0)
        integrator.setPerDofVariableByName("noise", MASS)
        integrator.addUpdateContextState()

        if using_COM_constraint:
            integrator.addComputePerDof("masked", "v*mask")
            integrator.addComputeSum("vCOMx", "masked*vector(1, 0, 0)")
            integrator.addComputeSum("vCOMy", "masked*vector(0, 1, 0)")
            integrator.addComputeSum("vCOMz", "masked*vector(0, 0, 1)")
            integrator.addComputePerDof("v", "v - vector(vCOMx,vCOMy,vCOMz)")
        else: pass

        integrator.addComputePerDof("F_over_m", "f/mass")

        if using_COM_constraint:
            integrator.addComputePerDof("masked", "F_over_m*mask")
            integrator.addComputeSum("vCOMx", "masked*vector(1, 0, 0)")
            integrator.addComputeSum("vCOMy", "masked*vector(0, 1, 0)")
            integrator.addComputeSum("vCOMz", "masked*vector(0, 0, 1)")
            integrator.addComputePerDof("F_over_m", "F_over_m - vector(vCOMx,vCOMy,vCOMz)")
        else: pass

        integrator.addComputePerDof("v", "v + dt*F_over_m")
        integrator.addComputePerDof("x", "x + 0.5*dt*v")

        integrator.addComputePerDof("noise", "SD*gaussian")

        if using_COM_constraint:
            integrator.addComputePerDof("masked", "noise*mask")
            integrator.addComputeSum("vCOMx", "masked*vector(1, 0, 0)")
            integrator.addComputeSum("vCOMy", "masked*vector(0, 1, 0)")
            integrator.addComputeSum("vCOMz", "masked*vector(0, 0, 1)")
            integrator.addComputePerDof("noise", "noise - vector(vCOMx,vCOMy,vCOMz)")
        else: pass

        integrator.addComputePerDof("v", "alpha*v + beta_sqrt_kT*noise")
        integrator.addComputePerDof("x", "x + 0.5*dt*v")

    return integrator
"""
## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
