
''' util_tf.py

convert arrays tf <-> np:
    f : np2tf_
    f : tf2np_

methods:
    f : norm_
    f : unit_
    f : norm_clipped_
    f : unit_clipped_
    f : det_3x3 
    f : make_COM_removal_matrix_
    f : PCA_
    c : NotWhitenFlow
    c : WhitenFlow
    f : pad_ranges_
    f : get_ranges_centres_MIN_MAX_
    f : get_ranges_centres_
    f : scale_shift_x_
    f : scale_shift_individual_x_
    c : FocusedBonds
    c : FocusedAngle
    f : average_torsion_np_
    f : centre_torsion_tf_
    c : FocusedTorsions
    f : merge_periodic_masks_
    c : Static_Rotations_Layer
    f : ample_phi_in_limits_
    f : sample_theta1_in_limits_
    f : sample_theta0_fastest_
    c : identity_shift
    c : FocusedHemisphere
    f : quat2axisangle_
    f : get_coupling_masks_
    f : reshape_to_molecules_tf_
    f : reshape_to_atoms_tf_
    f : reshape_to_flat_tf_
    f : get_distance_tf_
    f : get_angle_tf_
    f : get_torsion_tf_
    f : r_to_x_atom_
    f : IC_ladJ_inv_
    f : IC_forward_
    f : NeRF_tf_
    f : IC_inverse_
    f : mat_to_quat_tf_ (cond_0_true_, cond_1_true_, cond_2_true_, all_conds_false_)
    f : CB_ladJ_inv_
    f : CB_forward_
    f : quat_to_mat_tf_
    f : CB_inverse_
    f : test_CB_transformation_
    f : CB_single_molecule_forward_
    f : CB_single_molecule_inverse_
    f : hemisphere_ 
    f : hemisphere_forward_
    f : hemisphere_inverse_
    f : sample_q_
    f : quat_metrix_
    f : quat_product_, quat_inverse_ # not used
    f : box_forward_, box_inverse_ # not used
'''

from ..util_np import *

import tensorflow as tf

from .rqs import cast_32_, cast_64_, rqs_bin_

## ## ## ## 

DTYPE_tf = tf.float32
np2tf_ = lambda x : tf.cast(x, dtype=DTYPE_tf)
tf2np_ = lambda x : x.numpy() 

PI = 3.14159265358979323846264338327950288

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## general:

_clip_low_at_ = 1e-8
_clip_high_at_ = 1e+18
clip_positive_ = lambda x : tf.clip_by_value(x, _clip_low_at_, _clip_high_at_) 

norm_ = lambda x : tf.norm(x, axis=-1,keepdims=True)
unit_ = lambda x : x / norm_(x)

norm_clipped_ = lambda x : clip_positive_(tf.norm(x,axis=-1,keepdims=True))
unit_clipped_ = lambda x : x / norm_clipped_(x)

def det_3x3_(M, keepdims=False):
    # M : (...,3,3)
    return tf.reduce_sum( tf.linalg.cross(M[...,0], M[...,1]) * M[...,2], axis=-1, keepdims=keepdims)

def make_COM_removal_matrix_(n_particles, dim=3):
    """ 
    Output:
        M : (n_particles*dim, n_particles*dim) matrix
            (n_particles - 1)*dim eigenvalues are ones, and dim eigenvalues are zeros.
            using the top (n_particles - 1)*dim eigenvectors removes COM from data 
            similarly to whitening, but is volume preserving.
    """
    N = n_particles # same mass
    m = N*dim
    M = np.zeros([m,m])
    a = 1.0 - 1.0/N
    b =     - 1.0/N
    for i in range(m):
        for j in range(m):
            if i==j:
                M[i,j] = a
            elif not (i-j)%dim:
                M[i,j] = b
            else: pass
    return M

def PCA_(X0, removedims=3, diagonal=False, isotropic=False, not_whiten=False):
    ''' REF: https://github.com/noegroup/bgflow '''

    # X0 : (m, dim)

    if removedims is None: keepdims = X0.shape[1]
    else:                  keepdims = X0.shape[1]-removedims
    X0mean = X0.astype(np.float64).mean(axis=0).astype(np.float32)
    X0meanfree = X0 - X0mean
    C = np.matmul(X0meanfree.T, X0meanfree) / (X0meanfree.shape[0] - 1.0)

    if diagonal or isotropic:
        C = C*np.eye(C.shape[0])
        if isotropic:
            C = np.eye(C.shape[0])*(C.max())
        else: pass
    else: pass

    if not_whiten:
        C = make_COM_removal_matrix_(C.shape[0]//3, dim=3)
    else: pass

    eigval, eigvec = np.linalg.eigh(C)
    _eigval = np.array(eigval)
    I = np.argsort(eigval)[::-1]
    I = I[:keepdims]
    eigval = eigval[I]

    mask = np.where(_eigval<1e-6,1.0,0.0)
    if mask.sum() > removedims:
        print('! pca : keeping very small or negative eigenvalues')
    else: pass

    std = np.sqrt(eigval)
    eigvec = eigvec[:, I]
    Twhiten = np.matmul(eigvec, np.diag(1.0 / std))
    Tblacken = np.matmul(np.diag(std), eigvec.T)
    return X0mean, Twhiten, Tblacken, std, _eigval

class NotWhitenFlow:
    def __init__(self, r_flat, removedims=3, whiten_anyway=False):

        # r_flat : (m, dim) ; dim = n_mol*3
        self.dim_larger = r_flat.shape[1]
        self.dim_smaller = self.dim_larger - 3
        self.n_mol = self.dim_larger//3
        assert self.dim_larger/3 == self.n_mol

        if whiten_anyway:
            self.whiten_anyway = False
            self.WF_keepdim = WhitenFlow(self._forward_np(r_flat), removedims=0)
            self.whiten_anyway = True
        else:
            self.whiten_anyway = False

    def forward(self, x):
        ladJ = 0.0
        
        x = tf.reshape(x, [-1,self.n_mol,3])
        x -= x[:,:1]
        x = x[:,1:]
        x = tf.reshape(x, [-1, self.dim_smaller])

        if self.whiten_anyway: x, ladJ = self.WF_keepdim.forward(x)
        else: pass

        return x, ladJ
    
    def inverse(self, x):
        ladJ = 0.0

        if self.whiten_anyway: x, ladJ = self.WF_keepdim.inverse(x)
        else: pass

        x = tf.reshape(x, [-1,self.n_mol-1,3])
        x = tf.concat([tf.zeros_like(x[:,:1]), x], axis=1)
        x = tf.reshape(x, [-1, self.dim_larger])

        return x, ladJ

    def _forward_np(self, x):
        x = np.array(x)
        x = np.reshape(x, [-1,self.n_mol,3])
        x -= x[:,:1]
        x = x[:,1:]
        x = np.reshape(x, [-1,self.dim_smaller])

        if self.whiten_anyway: x = self.WF_keepdim._forward_np(x)
        else: pass

        return x

class WhitenFlow:
    ' REF: https://github.com/noegroup/bgflow'
    def __init__(self, r, removedims=3, diagonal=False, isotropic=False):
        
        # r : (m, dim)

        X0mean, Twhiten, Tblacken, std, eigval = PCA_(r,  removedims=removedims, diagonal=diagonal, isotropic=isotropic)
        
        self.X0mean = np2tf_(X0mean)
        self.Twhiten = np2tf_(Twhiten)
        self.Tblacken = np2tf_(Tblacken)
        self.std = np2tf_(std)
        self.eigval = eigval

        if np.any(std <= 0):
            raise ValueError("Cannot construct whiten layer because trying to keep nonpositive eigenvalues.")
            # if warning (at removedims=3) -> probably not enough data
            # (number of datapoints needs to be > number of dimensions)
        else: pass
        self.jacobian_xz = -tf.reduce_sum(tf.math.log(self.std))

    def _whiten(self, x):
        y = tf.einsum('oi,ij->oj',x - self.X0mean, self.Twhiten) 
        dlogp = self.jacobian_xz * tf.ones((x.shape[0], 1))
        return y, dlogp

    def _blacken(self, x):
        y = tf.einsum('oi,ij->oj', x, self.Tblacken) + self.X0mean
        dlogp = -self.jacobian_xz * tf.ones((x.shape[0], 1))
        return y, dlogp

    def forward(self, x):
        y, dlogp = self._whiten(x)
        return y, dlogp

    def inverse(self, x):
        y, dlogp = self._blacken(x)
        return y, dlogp

    def _forward_np(self, x):
        y = np.einsum('oi,ij->oj',x - self.X0mean.numpy(), self.Twhiten.numpy()) 
        return y

def pad_ranges_(Min, Max, factor):
    Range = Max - Min
    assert factor >= 1.0
    return (Range*factor - Range)*0.5

def get_ranges_centres_MIN_MAX_(x, axis:list, percentage_pad=0.0, range_limits:list=None, keepdims=False):
    # old
    axis = tuple(axis)

    eps = pad_ranges_(x.min(axis=axis),
                      x.max(axis=axis),
                      factor = 1.0 + 0.01*percentage_pad,
                      )
    Min = x.min(axis=axis, keepdims=keepdims) - eps
    Max = x.max(axis=axis, keepdims=keepdims) + eps
    
    if range_limits is not None:
        MIN, MAX = range_limits
        assert MAX > MIN
        if np.where(Min<MIN,1,0).sum() > 0 or np.where(Max>MAX,1,0).sum() > 0: 
            print('!!! range_limits were provided but do not match the data, this will give an error below')
        else: pass
        assert not np.where(Min <= MIN,1,0).sum()
        assert not np.where(Max >= MAX,1,0).sum()
        
        Min = np.where(Min<MIN,MIN,Min)
        Max = np.where(Max>MAX,MAX,Max)
    else: pass

    ranges = np2tf_(Max - Min)
    centres = np2tf_(Min + 0.5*ranges)

    return ranges, centres
'''
def get_ranges_centres_(x, axis:list, range_limits=None, keepdims=False, centroid_method_=np.median):
    axis = tuple(axis)

    centres = centroid_method_(x, axis=axis, keepdims=keepdims)
    ranges  = np.abs(x - centroid_method_(x, axis=axis, keepdims=True)).max(axis=axis, keepdims=keepdims)*2.0

    Min = centres - ranges*0.5
    Max = centres + ranges*0.5

    if range_limits is not None:
        MIN, MAX = range_limits
        Min = np.where(Min<MIN,MIN,Min)
        Max = np.where(Max>MAX,MAX,Max)
        ranges = np.stack([np.abs(centres - Min), np.abs(Max - centres)],axis=-1).max(axis=-1)*2.0
        A = np.where(centres - ranges*0.5 < MIN, 1, 0).sum()
        B = np.where(centres + ranges*0.5 > MAX, 1, 0).sum()
        if  A or B:
            ranges, centres = get_ranges_centres_MIN_MAX_(x, axis=axis, percentage_pad=0.0, range_limits=range_limits, keepdims=keepdims)
            # print('the older (MIN/MAX) method was used to standardise the marginal DOFs in this case')
            return ranges, centres
        else:
            pass
    else: pass
        
    return np2tf_(ranges), np2tf_(centres)
'''
#'''
def get_ranges_centres_(x, axis:list, range_limits=None, keepdims=False):
    return get_ranges_centres_MIN_MAX_(x, axis=axis, percentage_pad=0.0, range_limits=range_limits, keepdims=keepdims)
#'''

def scale_shift_x_(x,
                   physical_ranges_x,  # (n,)
                   physical_centres_x, # (n,)
                   forward=True,
                   ):
    n_dof = x.shape[-1]

    y = [] ; ladJ = 0.0
    if forward:
        for i in range(n_dof):
            ji = 2.0 / physical_ranges_x[i]
            y.append( ji*(x[...,i] - physical_centres_x[i]) )
            ladJ += tf.math.log(ji)

    else:
        for i in range(n_dof):
            ji = physical_ranges_x[i] * 0.5
            y.append( ji*x[...,i] + physical_centres_x[i]   )
            ladJ += tf.math.log(ji)
    
    if len(x.shape)>2:
        ladJ *= tf.cast(tf.math.reduce_prod(x.shape[1:-1]),dtype=tf.float32)
    else: pass

    return tf.stack(y, axis=-1), ladJ

def scale_shift_x_general_(x,
                           physical_ranges_x,  # (n,)
                           physical_centres_x, # (n,)
                           model_range = 2.0,
                           model_centre = 0.0,
                           forward=True,
                           ):
    n_dof = x.shape[-1]

    y = [] ; ladJ = 0.0
    if forward:
        for i in range(n_dof):
            ji = model_range / physical_ranges_x[i]
            y.append( ji*(x[...,i] - physical_centres_x[i]) + model_centre )
            ladJ += tf.math.log(ji)

    else:
        for i in range(n_dof):
            ji = physical_ranges_x[i] / model_range
            y.append( ji*(x[...,i] - model_centre) + physical_centres_x[i] )
            ladJ += tf.math.log(ji)
    
    if len(x.shape)>2:
        ladJ *= tf.cast(tf.math.reduce_prod(x.shape[1:-1]),dtype=tf.float32)
    else: pass

    return tf.stack(y, axis=-1), ladJ

def scale_shift_individual_x_(x,
                              physical_ranges_x,  # [1,...] use add_batch_axis True
                              physical_centres_x, # [1,...] use add_batch_axis True
                              forward=True,
                             ):
    axes_sum = - tf.range(1,tf.rank(x))
    
    if forward:
        J = 2.0 / physical_ranges_x
        y = J*(x - physical_centres_x)
        ladJ = tf.reduce_sum(tf.math.log(J), axis=axes_sum)[...,tf.newaxis]
        
    else:
        J = 0.5 * physical_ranges_x
        y = J*x + physical_centres_x
        ladJ = tf.reduce_sum(tf.math.log(J), axis=axes_sum)[...,tf.newaxis]

    return y, ladJ

## ## focused bonds and angles:

"""
    def _replace_ranges_(self, new_ranges):
        ''' dont remember what this was
        for merging different ic_map ranges into a broader range
        '''
        # shape matching
        assert new_ranges.shape == self.ranges.shape      
        # a bond length cannot be singular, or negative
        A = np.where(new_ranges <= 0.0, 1, 0).sum()
        B = np.where(self.centres - new_ranges*0.5 <= 0.0, 1, 0).sum()
        assert not A and not B
        assert not np.where(self.centres - new_ranges*0.5 <= 0.0, 1, 0).sum()
        # warning if range longer than maximum heuristic of self.max_bond_length
        if np.where(self.centres + new_ranges*0.5 > self.max_bond_length, 1, 0).sum() > 0:
            print('! some bond lengths exceed a length expected from an un-focused representation (range limits)')
        else: pass
        self.ranges = new_ranges
"""

class FocusedBonds:
    def __init__(self, 
                 X, # (m,n_mol,D) or (m, n_mol)
                 axis = [0,1], # default axis for first case above
                 #percentage_pad = 1.0,
                 focused = True,
                 ):
        self.axis = tuple(axis)

        self.min_bond_length = 0.05 # (units: nm)
        self.max_bond_length = 0.22 # (units: nm)
        #self.percentage_pad = percentage_pad 
        self.range_limits = [self.min_bond_length,self.max_bond_length]
        self.focused = focused

        #print('bonds min, max:', np.min(X), np.max(X))
        assert np.min(X) >= self.min_bond_length, f'{np.min(X)} < {self.min_bond_length} nm'
        assert np.max(X) <= self.max_bond_length, f'{np.max(X)} > {self.max_bond_length} nm'

        ranges, centres = get_ranges_centres_(np.array(X), axis=self.axis, keepdims=True,
                                              #percentage_pad=self.percentage_pad,
                                              range_limits=self.range_limits)
        
        if self.focused: pass
        else:
            ranges = ranges*0.0 + self.range_limits[1]-self.range_limits[0]
            centres = centres*0.0 + 0.5*(self.range_limits[1]-self.range_limits[0])
        
        self.ranges = np2tf_(ranges)
        self.centres = np2tf_(centres)
        self.mask_periodic = np.zeros_like(self.ranges).astype(np.int32) # just for shape

        ##
        # Y = self.forward_(np2tf_(X))[0].numpy()
        # self.loc = np.mean(Y, axis=self.axis)
        #s elf.scale = np.mean((Y - self.loc[tf.newaxis,:])**2, axis=self.axis)**0.5 # SD
        ##

    def forward_(self, X):
        return scale_shift_individual_x_(X, self.ranges, self.centres, forward=True)
    def inverse_(self, X):
        return scale_shift_individual_x_(X, self.ranges, self.centres, forward=False)
    def __call__(self, X, forward=True):
        if forward: return self.forward_(X)
        else:       return self.inverse_(X)

class FocusedAngles:
    def __init__(self, 
                 X, # (m,n_mol,D) or (m, n_mol)
                 axis = [0,1], # default axis for first case above
                 #percentage_pad = 1.0,
                 range_limits = [0.0,PI],
                 focused = True,
                 ):
        self.axis = tuple(axis)

        self.min_angle = range_limits[0]
        self.max_angle = range_limits[1]

        assert np.min(X) >= self.min_angle
        assert np.max(X) <= self.max_angle

        #self.percentage_pad = percentage_pad 
        self.range_limits = range_limits
        self.focused = focused
        ranges, centres = get_ranges_centres_(np.array(X), axis=self.axis, keepdims=True,
                                              #percentage_pad=self.percentage_pad,
                                              range_limits=self.range_limits)
        
        if self.focused: pass
        else:
            ranges = ranges*0.0 + self.range_limits[1]-self.range_limits[0]
            centres = centres*0.0 + 0.5*(self.range_limits[1]-self.range_limits[0])
        
        self.ranges = np2tf_(ranges)
        self.centres = np2tf_(centres)
        self.mask_periodic = np.zeros_like(self.ranges).astype(np.int32) # just for shape

        ##
        # Y = self.forward_(np2tf_(X))[0].numpy()
        # self.loc = np.mean(Y, axis=self.axis)
        # self.scale = np.mean((Y - self.loc[tf.newaxis,:])**2, axis=self.axis)**0.5 # SD
        ##

    def forward_(self, X):
        return scale_shift_individual_x_(X, self.ranges, self.centres, forward=True)
    def inverse_(self, X):
        return scale_shift_individual_x_(X, self.ranges, self.centres, forward=False)
    def __call__(self, X, forward=True):
        if forward: return self.forward_(X)
        else:       return self.inverse_(X)

## focused torsions:

# median_tf_ = lambda x, axis=0, keepdims=True: tfp.stats.percentile(x, 50.0, interpolation='midpoint', axis=axis, keepdims=keepdims)

def average_torsion_np_(x,
                        axis = 0,
                        keepdims = True,
                        pooling_method_ = np.mean,
                        ):
    return np.arctan2(pooling_method_(np.sin(x),axis=axis,keepdims=keepdims),
                      pooling_method_(np.cos(x),axis=axis,keepdims=keepdims)
                     )

def centre_torsion_tf_(x, x_mean, forward=True):
    if forward:
        return tf.math.floormod(x - x_mean + PI, 2.0*PI) - PI
    else:
        return tf.math.floormod(x + x_mean + PI, 2.0*PI) - PI
    
class FocusedTorsions:
    def __init__(self, 
                 X, # (m, n_mol, D)
                 axis = [0,1], # default axis for first case above
                 #percentage_pad = 0.0,
                 focused = True,
                 verbose = True,
                 mask_periodic = None,
                 ):
        self.axis = tuple(axis)

        #self.percentage_pad = percentage_pad 
        self.range_limits = [-PI-0.000001,PI+0.000001]
        self.focused = focused

        X = np2tf_(X)
        self.centres_0 = np2tf_(
                                average_torsion_np_(np.array(X), 
                                                    axis = self.axis,
                                                    keepdims = True,
                                                    pooling_method_=np.median)
                               )
        Y = centre_torsion_tf_(X, self.centres_0, forward=True)
        self.ranges_here, centres_1 = get_ranges_centres_(np.array(Y), axis=self.axis, keepdims=True,
                                                          #percentage_pad=self.percentage_pad,
                                                          range_limits = self.range_limits)
        ranges = np.array(self.ranges_here)
        if self.focused: pass
        else:
            ranges = ranges*0.0 + 2*PI
            centres_1 = centres_1*0.0
        
        self._ranges = np.array(ranges)
        self._centres_1 = np.array(centres_1)

        self.set_ranges_(mask_periodic)

    def set_ranges_(self, mask_periodic):

        if mask_periodic is not None:
            assert set(mask_periodic.flatten().tolist()) in [set([0]),set([0,1]),set([1])]
            shape = mask_periodic.shape
            assert len(shape) == len(self._ranges.shape)
            assert shape[-1] == self._ranges.shape[-1]
            assert all([x==1 for x in shape[:-1]])
            self.mask_periodic = mask_periodic
        else:
            # same in all molecules
            mask_periodic = np.where(self._ranges>0.8*2*PI, 1, 0)     # (1, n_mol, D)
            self.mask_periodic = mask_periodic.max(-2, keepdims=True) # (1, 1, D)

        self.maks_periodic_molecules = np.broadcast_to(self.mask_periodic, self._ranges.shape)

        self.mask_nonperiodic_molecules = 1.0 - self.maks_periodic_molecules
        self.ranges    = np2tf_(self.maks_periodic_molecules*2.0*PI + self.mask_nonperiodic_molecules*self._ranges)
        self.centres_1 = np2tf_(self.maks_periodic_molecules*0.0 + self.mask_nonperiodic_molecules*self._centres_1)

        print(self.mask_periodic.sum(),'out of', np.prod(mask_periodic.shape),'potentially periodic marginal variables are set to periodic')
        print('This topology is shared over all molecules.\n')

    def forward_(self, X):
        Y = centre_torsion_tf_(X, self.centres_0, forward=True)
        Z, ladJ = scale_shift_individual_x_(Y, self.ranges, self.centres_1, forward=True)
        return Z, ladJ
    
    def inverse_(self, Z):
        Y, ladJ = scale_shift_individual_x_(Z, self.ranges, self.centres_1, forward=False)
        X = centre_torsion_tf_(Y, self.centres_0, forward=False)
        return X, ladJ
    
    def __call__(self, X, forward=True):
        if forward: return self.forward_(X)
        else:       return self.inverse_(X)

def merge_periodic_masks_(list_periodic_masks):
    ''' inputs arrays with integer elements 0 and 1 only
    '''
    n_inputs = len(list_periodic_masks)
    shapes = [x.shape for x in list_periodic_masks]
    assert all(shapes[0] == x for x in shapes)
    summed = np.sum([list_periodic_masks[k] for k in range(n_inputs)],axis=0)
    output = np.array(np.ma.divide(summed,summed)).astype(np.int32).reshape(shapes[0])
    return output

## focused rotations:

# cell600 : a uniform grid on S3 (used for grid search in Static_Rotations_Layer)
cell600 = [ [ 0.5       ,  0.5       ,  0.5       ,  0.5       ],
            [-0.5       ,  0.5       ,  0.5       ,  0.5       ],
            [ 0.5       , -0.5       ,  0.5       ,  0.5       ],
            [-0.5       , -0.5       ,  0.5       ,  0.5       ],
            [ 0.5       ,  0.5       , -0.5       ,  0.5       ],
            [-0.5       ,  0.5       , -0.5       ,  0.5       ],
            [ 0.5       , -0.5       , -0.5       ,  0.5       ],
            [-0.5       , -0.5       , -0.5       ,  0.5       ],
            [ 0.5       ,  0.5       ,  0.5       , -0.5       ],
            [-0.5       ,  0.5       ,  0.5       , -0.5       ],
            [ 0.5       , -0.5       ,  0.5       , -0.5       ],
            [-0.5       , -0.5       ,  0.5       , -0.5       ],
            [ 0.5       ,  0.5       , -0.5       , -0.5       ],
            [-0.5       ,  0.5       , -0.5       , -0.5       ],
            [ 0.5       , -0.5       , -0.5       , -0.5       ],
            [-0.5       , -0.5       , -0.5       , -0.5       ],
            [ 1.        ,  0.        ,  0.        ,  0.        ],
            [-1.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  1.        ,  0.        ,  0.        ],
            [ 0.        , -1.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  1.        ,  0.        ],
            [ 0.        ,  0.        , -1.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  1.        ],
            [ 0.        ,  0.        ,  0.        , -1.        ],
            [ 0.        ,  0.80901699,  0.5       ,  0.30901699],
            [ 0.        ,  0.5       ,  0.30901699,  0.80901699],
            [ 0.        ,  0.30901699,  0.80901699,  0.5       ],
            [ 0.80901699,  0.        ,  0.30901699,  0.5       ],
            [ 0.80901699,  0.5       ,  0.        ,  0.30901699],
            [ 0.80901699,  0.30901699,  0.5       ,  0.        ],
            [ 0.5       ,  0.        ,  0.80901699,  0.30901699],
            [ 0.5       ,  0.80901699,  0.30901699,  0.        ],
            [ 0.5       ,  0.30901699,  0.        ,  0.80901699],
            [ 0.30901699,  0.        ,  0.5       ,  0.80901699],
            [ 0.30901699,  0.80901699,  0.        ,  0.5       ],
            [ 0.30901699,  0.5       ,  0.80901699,  0.        ],
            [ 0.        ,  0.80901699,  0.5       , -0.30901699],
            [ 0.        ,  0.5       , -0.30901699,  0.80901699],
            [ 0.        , -0.30901699,  0.80901699,  0.5       ],
            [ 0.80901699,  0.        , -0.30901699,  0.5       ],
            [ 0.80901699,  0.5       ,  0.        , -0.30901699],
            [ 0.80901699, -0.30901699,  0.5       ,  0.        ],
            [ 0.5       ,  0.        ,  0.80901699, -0.30901699],
            [ 0.5       ,  0.80901699, -0.30901699,  0.        ],
            [ 0.5       , -0.30901699,  0.        ,  0.80901699],
            [-0.30901699,  0.        ,  0.5       ,  0.80901699],
            [-0.30901699,  0.80901699,  0.        ,  0.5       ],
            [-0.30901699,  0.5       ,  0.80901699,  0.        ],
            [ 0.        ,  0.80901699, -0.5       ,  0.30901699],
            [ 0.        , -0.5       ,  0.30901699,  0.80901699],
            [ 0.        ,  0.30901699,  0.80901699, -0.5       ],
            [ 0.80901699,  0.        ,  0.30901699, -0.5       ],
            [ 0.80901699, -0.5       ,  0.        ,  0.30901699],
            [ 0.80901699,  0.30901699, -0.5       ,  0.        ],
            [-0.5       ,  0.        ,  0.80901699,  0.30901699],
            [-0.5       ,  0.80901699,  0.30901699,  0.        ],
            [-0.5       ,  0.30901699,  0.        ,  0.80901699],
            [ 0.30901699,  0.        , -0.5       ,  0.80901699],
            [ 0.30901699,  0.80901699,  0.        , -0.5       ],
            [ 0.30901699, -0.5       ,  0.80901699,  0.        ],
            [ 0.        ,  0.80901699, -0.5       , -0.30901699],
            [ 0.        , -0.5       , -0.30901699,  0.80901699],
            [ 0.        , -0.30901699,  0.80901699, -0.5       ],
            [ 0.80901699,  0.        , -0.30901699, -0.5       ],
            [ 0.80901699, -0.5       ,  0.        , -0.30901699],
            [ 0.80901699, -0.30901699, -0.5       ,  0.        ],
            [-0.5       ,  0.        ,  0.80901699, -0.30901699],
            [-0.5       ,  0.80901699, -0.30901699,  0.        ],
            [-0.5       , -0.30901699,  0.        ,  0.80901699],
            [-0.30901699,  0.        , -0.5       ,  0.80901699],
            [-0.30901699,  0.80901699,  0.        , -0.5       ],
            [-0.30901699, -0.5       ,  0.80901699,  0.        ],
            [ 0.        , -0.80901699,  0.5       ,  0.30901699],
            [ 0.        ,  0.5       ,  0.30901699, -0.80901699],
            [ 0.        ,  0.30901699, -0.80901699,  0.5       ],
            [-0.80901699,  0.        ,  0.30901699,  0.5       ],
            [-0.80901699,  0.5       ,  0.        ,  0.30901699],
            [-0.80901699,  0.30901699,  0.5       ,  0.        ],
            [ 0.5       ,  0.        , -0.80901699,  0.30901699],
            [ 0.5       , -0.80901699,  0.30901699,  0.        ],
            [ 0.5       ,  0.30901699,  0.        , -0.80901699],
            [ 0.30901699,  0.        ,  0.5       , -0.80901699],
            [ 0.30901699, -0.80901699,  0.        ,  0.5       ],
            [ 0.30901699,  0.5       , -0.80901699,  0.        ],
            [ 0.        , -0.80901699,  0.5       , -0.30901699],
            [ 0.        ,  0.5       , -0.30901699, -0.80901699],
            [ 0.        , -0.30901699, -0.80901699,  0.5       ],
            [-0.80901699,  0.        , -0.30901699,  0.5       ],
            [-0.80901699,  0.5       ,  0.        , -0.30901699],
            [-0.80901699, -0.30901699,  0.5       ,  0.        ],
            [ 0.5       ,  0.        , -0.80901699, -0.30901699],
            [ 0.5       , -0.80901699, -0.30901699,  0.        ],
            [ 0.5       , -0.30901699,  0.        , -0.80901699],
            [-0.30901699,  0.        ,  0.5       , -0.80901699],
            [-0.30901699, -0.80901699,  0.        ,  0.5       ],
            [-0.30901699,  0.5       , -0.80901699,  0.        ],
            [ 0.        , -0.80901699, -0.5       ,  0.30901699],
            [ 0.        , -0.5       ,  0.30901699, -0.80901699],
            [ 0.        ,  0.30901699, -0.80901699, -0.5       ],
            [-0.80901699,  0.        ,  0.30901699, -0.5       ],
            [-0.80901699, -0.5       ,  0.        ,  0.30901699],
            [-0.80901699,  0.30901699, -0.5       ,  0.        ],
            [-0.5       ,  0.        , -0.80901699,  0.30901699],
            [-0.5       , -0.80901699,  0.30901699,  0.        ],
            [-0.5       ,  0.30901699,  0.        , -0.80901699],
            [ 0.30901699,  0.        , -0.5       , -0.80901699],
            [ 0.30901699, -0.80901699,  0.        , -0.5       ],
            [ 0.30901699, -0.5       , -0.80901699,  0.        ],
            [ 0.        , -0.80901699, -0.5       , -0.30901699],
            [ 0.        , -0.5       , -0.30901699, -0.80901699],
            [ 0.        , -0.30901699, -0.80901699, -0.5       ],
            [-0.80901699,  0.        , -0.30901699, -0.5       ],
            [-0.80901699, -0.5       ,  0.        , -0.30901699],
            [-0.80901699, -0.30901699, -0.5       ,  0.        ],
            [-0.5       ,  0.        , -0.80901699, -0.30901699],
            [-0.5       , -0.80901699, -0.30901699,  0.        ],
            [-0.5       , -0.30901699,  0.        , -0.80901699],
            [-0.30901699,  0.        , -0.5       , -0.80901699],
            [-0.30901699, -0.80901699,  0.        , -0.5       ],
            [-0.30901699, -0.5       , -0.80901699,  0.        ] ]

class Static_Rotations_Layer:
    def __init__(self,
                 q,
                 indices : list = None,
                 ):
        ' self.n_mol here can be diffent from self.n_mol in the ic_map or PGMcrys model, depending on how q was reshaped '
        self.n_mol = q.shape[1] 
        ' convert cell600 vectors (a uniform grid on S3) to rotation matrices '
        self.PS = quat_metrix_(np2tf_(cell600)) # (120,4,4)
        ' find a set of self.n_mol number of self.best_indices (rows of self.PS) '
        if indices is None: self.find_best_(np2tf_(q))
        else: self.best_indices = indices
        ' self.best_indices corespond to unitvectors that give best rotation for each molecule ' 
        self.P  = tf.stack([self.PS[index] for index in self.best_indices])
        """a 'best rotation' for a given molecule, or a set of molecules, is one that
           jointly minimises the 3 errors on the margianls of s = hemisphere_forward_(...)[0]
           The three error functions are dicussed in self.find_best_
        """
    def find_best_(self, q):

        self.best_indices = []
        ''' 
        s = [theta0, theta1, phi] = s(q) = hemisphere_forward_(s, rescale_marginals=False)[0]
            theta0 \in [0, PI/2] ; theta0_rescaled \in [-1,1] when rescale_marginals=True (default)
            theta1 \in [0, PI]   ; theta1_rescaled \in [-1,1] when rescale_marginals=True (default)
            phi    \in [-PI,PI)  ; phi_rescaled    \in [-1,1) when rescale_marginals=True (default)

        log volume change of hemisphere_forward_:
            \gamma = log(|det(ds(q)/dq)|) = - log( (tf.sin(theta0)**2) * tf.sin(theta1) )

        the three error functions to be jointly minimised by some 'best' rotation of the molecule are:
            
            err_theta0_(theta0_rescaled)
                lower is better, minimised by rotation the data away from theta0 = -1, 1
                when a datapoint is at -1, the \gamma is not finite (a singularity)
                when a datapoint is at  1, this is the edge of the hemisphere, where 
                the distribution in s is diconnected in theta1 and phi.
            
            err_theta1_(theta1_rescaled)
                lower is better, minimised by rotation the data away from theta1 = -1, 1
                when a datapoint is at -1 or 1, \gamma is not finited (singularities)

            err_phi_(phi_rescaled)
                optional, since a datapoint anywhere on [-1,1) is well defined
                still used because phi_rescaled localised in a smaller interval than [-1,1) 
                allows this marginal variable to be later treated as non-periodic inside the model
        '''
        # arbitrary heuristic choice of error functions that are higher in places that are worse
        # https://www.desmos.com/calculator/k351slu5g3
        err_theta0_ = lambda x : tf.reduce_sum(  5*tf.sin(x+0.1)**10 + (x+0.1)**2  )
        err_theta1_ = lambda x : tf.reduce_sum(  9*tf.sin(0.8*x)**8 + (0.8*x)**2   )
        err_phi_    = lambda x : tf.reduce_sum(    tf.sin(0.8*x)**4 + 0.5*(x**2)   )

        error_function_ = lambda s : err_theta0_(s[...,:1]) + err_theta1_(s[...,1:2]) + err_phi_(s[...,2:])
        ''' grid search for best rotations:
        for each molecule, or a set of molecules, a single rotation that minimises the error_function_ above
        is chosen after evaluating on the entire grid of rotations (the set of all cell600 vectors):
        '''
        for i in range(self.n_mol):
            errs = []
            for P in self.PS:
                errs.append( error_function_( hemisphere_forward_(tf.einsum('ij,...j->...i',P,q[:,i]))[0] ) )
            self.best_indices.append(np.argmin(errs))
            
    def forward_(self, q):
        # k is index for different molecules
        return tf.einsum('kij,...kj->...ki',self.P,q)

    def inverse_(self, q):
        # k is index for different molecules
        return tf.einsum('kji,...kj->...ki',self.P,q)

def sample_phi_in_limits_(m, E, F):
    # m : int
    # E : (1,n_mol)
    # F : (1,n_mol)
    # F-E > 0
    n_mol = len(E[0])
    return np.random.rand(m,n_mol)*(F-E) + E

def sample_theta1_in_limits_(m, C, D):
    # m : int
    # C : (1,n_mol)
    # D : (1,n_mol)
    # D-C > 0
    c = 0.5*PI
    rand = sample_phi_in_limits_(m, np.sin(C-c), np.sin(D-c)) # -cos
    return np.arcsin(rand) + c

def sample_theta0_fastest_(m, A, B, test=False):

    def rqs_here_(  x,
                    w, 
                    h,
                    s,
                    interval = [0., 0.5*PI],
                    forward = False,
                ):
        x = cast_64_(x)
        w = cast_64_(w)
        h = cast_64_(h)
        s = cast_64_(s)
        
        eps = 1e-6
        
        dim = x.shape[-1]
        shape_parallel = [_ for _ in x.shape[1:-1]] # list of ints, empty if ... = batch size
        n_axes = len(shape_parallel) + 2
        #print(n_axes)
        n_bins = (w.shape[-1]-1) // dim
        a, b = interval ; L = b - a
        
        grid_x = w[...,tf.newaxis,:] # (m, dim, n_bins+1)
        grid_y = h[...,tf.newaxis,:] # (m, dim, n_bins+1)
    
        x = tf.clip_by_value(x, a+eps, b-eps)
        
        if forward:
            diff_grid_x = grid_x - tf.expand_dims(x, axis=-1)
            ind_right_grid = tf.argmin(tf.where(diff_grid_x>=0.0, diff_grid_x, L+1.0), axis=-1)[...,tf.newaxis]
        else:
            diff_grid_y = grid_y - tf.expand_dims(x, axis=-1)
            ind_right_grid = tf.argmin(tf.where(diff_grid_y>=0.0, diff_grid_y, L+1.0), axis=-1)[...,tf.newaxis]
    
        xA = tf.gather_nd(grid_x, ind_right_grid-1, batch_dims=n_axes) # (m,dim)
        xB = tf.gather_nd(grid_x, ind_right_grid,   batch_dims=n_axes) # (m,dim)
        yA = tf.gather_nd(grid_y, ind_right_grid-1, batch_dims=n_axes) # (m,dim)
        yB = tf.gather_nd(grid_y, ind_right_grid,   batch_dims=n_axes) # (m,dim)
        
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    
        grid_slopes = tf.reshape(s, [-1]+shape_parallel+[dim, n_bins+1])
        
        sA = tf.gather_nd(grid_slopes, ind_right_grid-1, batch_dims=n_axes) # (m,dim)
        sB = tf.gather_nd(grid_slopes, ind_right_grid,   batch_dims=n_axes) # (m,dim)
            
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        
        y, ladJ = rqs_bin_(x, xA, xB, yA, yB, sA=sA, sB=sB, forward=forward) # A, B refer to the two knots.
        
        y = cast_32_(y)
        #ladJ = cast_32_(ladJ)
    
        return y.numpy()

    f_ = lambda x : x - 0.5*np.sin(2.0*x) # could not find inverse to approximating it using the spline
    dfdx_ = lambda x : 1.0 - np.cos(2.0*x) # 2.0*(np.sin(x)**2)

    n_bins = 20
    grid = np.linspace(0, 0.5*PI, n_bins)
    w = grid
    h = f_(grid)
    s = dfdx_(grid)

    n_mol = len(A[0])
    M = n_mol*m

    w = np.stack([w]*M,axis=0)
    h = np.stack([h]*M,axis=0)
    s = np.stack([s]*M,axis=0)
    
    x = sample_phi_in_limits_(m, f_(A), f_(B))
    samples = rqs_here_(sample_phi_in_limits_(m, f_(A), f_(B)).reshape([M,1]),
                        w, h, s,
                       ).reshape([m,n_mol])

    if test:
        z = np.linspace(0, 0.5*PI, M).reshape([M,1])
        err = np.abs( rqs_here_(z, w, h, s, forward=True) - f_(z) ).max()
        print('err:', err)
    else:
        err = 0.0
        
    return samples, err

class identity_shift:
    def __init__(self,):
        ''
    def forward_(self,x):
        return x
    def inverse_(self,x):
        return x
                
class FocusedHemisphere:
    def __init__(self, 
                 q, # (m, n_mol, 4)
                 srl_indices_known = None, # redundant because selection already pickled when saving model
                 focused = True,
                 static_rotations_defined_externally = False,
                 mask_periodic_Phi = None,
                 ):
        self.focused = focused
        q = np2tf_(q)
        '''
        Static_Rotations_Layer makes the MD quaternion distrubtions 'good' for the hemisphere representation.
        The sense of 'good' or'well-behaved' is dicussed in Static_Rotations_Layer (argmin_{p} error_function_(p,q)).
        '''
        if static_rotations_defined_externally:
            self.static_rotations_layer = identity_shift()
        else:
            self.static_rotations_layer = Static_Rotations_Layer(
                tf.gather(q, np.random.choice(len(q), min([20000,len(q)]), replace=False), axis=0),
                indices = srl_indices_known)
            q = self.static_rotations_layer.forward_(q)

        self.n_mol = q.shape[1]
        '''
        q above is now 'centred' which should be well-behaved in relation to s(q) = hemisphere_forward_(q)
            *(well-behaved for the entire MD dataset provided, where the crystal is stable)
        '''
        s = hemisphere_forward_(q, rescale_marginals=False)[0]
        theta0 = s[...,0]
        theta1 = s[...,1]
        phi = s[...,2]

        self.Focused_Theta0 = FocusedAngles(theta0, axis=[0], range_limits = [0.0, PI*0.5], focused=self.focused)
        self.Focused_Theta1 = FocusedAngles(theta1, axis=[0], focused=self.focused)
        self.Focused_Phi    = FocusedTorsions(phi,  axis=[0],  focused=self.focused, mask_periodic = mask_periodic_Phi)
        '''
        if all the molecules present along the -2 axis of q do not explore a wide distribution in q,
        the mapped s(q) distribution should also be non-periodic and simply connected
        For instance, all crystallographically unique molecule (molecules of the unitcell)
        have rotations centred around some average rotation with a fairly small variance.
        In turn, the focused = True allows to focus on the small rectangular region in s.
        '''
        self.set_ranges_(mask_periodic_Phi)

    def set_ranges_(self, mask_periodic_Phi):
        if mask_periodic_Phi is None: pass
        else:  self.Focused_Phi.set_ranges_(mask_periodic_Phi)
        
        self.mask_periodic = np.stack([self.Focused_Theta0.mask_periodic, # 0 just for shape
                                       self.Focused_Theta1.mask_periodic, # 0 just for shape
                                       self.Focused_Phi.mask_periodic], axis=-1) # (1,n_mol,3)
        ##
        if self.focused:
            centre = np.array(self.Focused_Theta0.centres)
            L = np.array(self.Focused_Theta0.ranges)
            A = centre-L*0.5 ; self.A = A # (1,n_mol)
            B = centre+L*0.5 ; self.B = B # (1,n_mol)
            centre = np.array(self.Focused_Theta1.centres)
            L = np.array(self.Focused_Theta1.ranges)
            C = centre-L*0.5 ; self.C = C # (1,n_mol)
            D = centre+L*0.5 ; self.D = D # (1,n_mol)
            centre = np.array(self.Focused_Phi.centres_1)
            L = np.array(self.Focused_Phi.ranges)
            E = centre-L*0.5 ; self.E = E # (1,n_mol)
            F = centre+L*0.5 ; self.F = F # (1,n_mol)
            # integrate on S3 within data limits to find area of the patch of S3
            # this is the support of the uniform base distribution to sample from.
            # two versions (v1,v2) of how it can be sampled, v2 is much faster.
            Vt0 = 0.5 * ( B - A - 0.5*( np.sin(2.*B) - np.sin(2.*A) ) )
            Vt1 = np.cos(C) - np.cos(D)
            Vp = F - E
            self.V_patch = Vt0*Vt1*Vp # (n_mol,)
            self.log_area_patch = np.log(self.V_patch).sum() # sum over molecules
            # self.log_area_patch is for all patches.
            ##
            assert self.B.max() < PI*0.5
            if self.B.min() < 0.2: 
                print('!! (warning 1/2) FocusedHemisphere : sampling the patch may be slow.')
                print('!! (warning 2/2) This is unusual (check the dataset).')
            else: pass
            self.sample_quaternion_patch_ = self.sample_quaternion_patch_v2_
            # uniform sampling.
        else:
            # uniform sampling.
            self.sample_quaternion_patch_ = lambda m : hemisphere_(sample_q_([m, self.n_mol]))
            # was sample_q_([m, self.n_mol]), because fliped when converted to flow by any packing layer.
            self.V_patch = np.array([PI**2]*self.n_mol) # (n_mol,)
            self.log_area_patch = np2tf_(np.log(self.V_patch).sum()) # sum over molecules
            # self.log_area_patch is for all patches.

    def forward_(self, q):
        q = self.static_rotations_layer.forward_(q)
        s, ladJ = hemisphere_forward_(q, rescale_marginals=False)
        theta0, ladj_scale_t0 = self.Focused_Theta0.forward_(s[...,0])
        theta1, ladj_scale_t1 = self.Focused_Theta1.forward_(s[...,1])
        phi, ladj_scale_phi = self.Focused_Phi.forward_(s[...,2])
        s_scaled = tf.stack([theta0, theta1, phi], axis=-1)
        ladJ = tf.reduce_sum(ladJ, axis=-2) + ladj_scale_t0 + ladj_scale_t1 + ladj_scale_phi
        return s_scaled, ladJ

    def inverse_(self, s_scaled):
        theta0, ladj_scale_t0 = self.Focused_Theta0.inverse_(s_scaled[...,0])
        theta1, ladj_scale_t1 = self.Focused_Theta1.inverse_(s_scaled[...,1])
        phi, ladj_scale_phi = self.Focused_Phi.inverse_(s_scaled[...,2])
        s = tf.stack([theta0, theta1, phi], axis=-1)
        qh, ladJ = hemisphere_inverse_(s, rescale_marginals=False)
        ladJ = tf.reduce_sum(ladJ, axis=-2) + ladj_scale_t0 + ladj_scale_t1 + ladj_scale_phi
        qh = self.static_rotations_layer.inverse_(qh)
        return qh, ladJ
    
    def __call__(self, X, forward=True):
        if forward: return self.forward_(X)
        else:       return self.inverse_(X)

    def sample_quaternion_patch_v1_(self, m):
        # accept uniform samples at places where it is in the rectangle
        # if is in the rectangle when base samples model would see are in the data range (focused)
        # samples falling outside of this range are rejected, until enough samples are collected
        samples = np.zeros([m,self.n_mol,4])
        mask = np.zeros([m,self.n_mol,4]).astype(np.int32)
        while 0 in mask:
            qh = hemisphere_(sample_q_([m,self.n_mol])).numpy()
            this = self.forward_(qh)[0].numpy()
            where0 = np.where(np.where((this<=1.0)&(this>=-1.0),1,0).sum(-1)==3)
            where1 = np.where(mask.sum(-1) == 0)
            N0 = len(where0[0])
            N1 = len(where1[0])
            N = min([N0,N1])
            where0b = [x[:N] for x in where0]
            where1b = [x[:N] for x in where1]
            mask[where1b[0],where0b[1]] = 1
            samples[where1b[0],where0b[1]] = qh[where0b[0],where0b[1]]
        return np2tf_(samples) # (m,n_mol,4)

    def sample_quaternion_patch_v2_(self, m):
        # the rectangle area integral in s is seperable into 3 parts
        # sampling can thus be done seperately on each of the marginals of s
        t0 = np2tf_(sample_theta0_fastest_(m, self.A, self.B, test=False)[0])
        #t0, self.n_itter = sample_theta0_in_limits_numerically_(m, self.A, self.B)
        #t0, self.n_itter = sample_theta0_in_limits_(m, self.A, self.B) # (m, n_mol), ()
        t1 = np2tf_(sample_theta1_in_limits_(m, self.C, self.D)) # (m, n_mol)
        p = np2tf_(sample_phi_in_limits_(m, self.E, self.F)) # (m, n_mol)
        # missed the following step initially.
        p = centre_torsion_tf_(p, self.Focused_Phi.centres_0, forward=False)
        #s_patch = tf.stack([t0, t1, p], axis=-1) # (m,n_mol,3)
        ct0 = tf.cos(t0) ; st0 = tf.sin(t0)
        ct1 = tf.cos(t1) ; st1 = tf.sin(t1)
        cp  = tf.cos(p)  ; sp  = tf.sin(p)
        q_patch = tf.stack([ct0, st0*ct1, st0*st1*cp, st0*st1*sp], axis=-1)
        return np2tf_(q_patch) # (m,n_mol,4)

#'''
def quat2axisangle_(q):
    'can be used for visualising rotational distributions in 3D'
    q0 = q[...,:1]
    q123 = q[...,1:]
    angle = 2.0 * tf.math.acos(q0)
    n = (1.0 - q0**2)**0.5
    axis = q123 / n
    return axis*angle

## ##

def get_coupling_masks_(dim_flow : int):
    """ REF: TABLE 1 in arXiv:2001.05486v2 (i-flow)
    Input:
        dim_flow : number of DOFs being transformed using coupling flow
    Output:
        cond_masks : (n_layers, dim_flow) array of zeros and ones
            'conditionaling masks' or 'coupling masks'
    Usage:
        In a NF model that is based on coupling:
            each coupling layer splits the total number (dim_flow) of 1D margianl DOFs into two sets A and B
            setting one coupling layer for each row of cond_masks allows all DOFs to be coupled
                in practice it is not necesary to use all of the rows (just first 4 were generally used; n_layers = 4)
    """
    cond_masks = []
    for i in range(dim_flow):
        a = 2**i
        x = np.array((([0]*a + [1]*a)*dim_flow)[:dim_flow])
        if 1 in x: pass
        else: break
        cond_masks.append(x)
    cond_masks = np.array(cond_masks)
    return cond_masks # plt.matshow(cond_masks)

##

def reshape_to_molecules_tf_(r, n_molecules, n_atoms_in_molecule):
    # same as numpy version in utils_np
    return tf.reshape(r,[r.shape[0], n_molecules, n_atoms_in_molecule, 3])

def reshape_to_atoms_tf_(r, n_molecules, n_atoms_in_molecule):
    # same as numpy version in utils_np
    return tf.reshape(r,[r.shape[0], n_molecules*n_atoms_in_molecule, 3])
    
def reshape_to_flat_tf_(r, n_molecules, n_atoms_in_molecule):
    # same as numpy version in utils_np
    return tf.reshape(r,[r.shape[0], n_molecules*n_atoms_in_molecule*3])

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## forward IC (internal coordinates):

def get_distance_tf_(R, inds_2_atoms):
    ''' bond distance '''
    # R            : (..., # atoms, 3)
    # inds_2_atoms : (2,)
    A,B = inds_2_atoms
    rA = R[...,A,:]  # (...,3)
    rB = R[...,B,:]  # (...,3)
    vBA = rA - rB    # (...,3)
    return norm_clipped_(vBA) # (...,1)

def get_angle_tf_(R, inds_3_atoms):
    ''' bond angle '''
    # R            : (..., # atoms, 3)
    # inds_3_atoms : (3,)

    A,B,C = inds_3_atoms
    rA = R[...,A,:] # (...,3)
    rB = R[...,B,:] # (...,3)
    rC = R[...,C,:] # (...,3)

    uBA = unit_clipped_(rA - rB) # (...,3)
    uBC = unit_clipped_(rC - rB) # (...,3)

    dot = tf.reduce_sum(uBA*uBC, axis=-1, keepdims=True)             # (...,1)
    dot = tf.clip_by_value(dot, -1.0, 1.0)                           # (...,1)
    
    theta = tf.acos(dot) # (...,1)
    theta = tf.clip_by_value(theta, _clip_low_at_, PI-_clip_low_at_) # (...,1)
 
    return theta # (...,1)

def get_torsion_tf_(R, inds_4_atoms):
    ''' REF: https://github.com/noegroup/bgflow '''
    # R            : (..., # atoms, 3)
    # inds_4_atoms : (4,)
    
    A,B,C,D = inds_4_atoms
    rA = R[...,A,:] # (...,3)
    rB = R[...,B,:] # (...,3)
    rC = R[...,C,:] # (...,3)
    rD = R[...,D,:] # (...,3)
    
    vBA = rA - rB   # (...,3)
    vBC = rC - rB   # (...,3)
    vCD = rD - rC   # (...,3)
    
    uBC = unit_clipped_(vBC) # (...,3)

    w = vCD - tf.reduce_sum(vCD*uBC, axis=-1, keepdims=True)*uBC # (...,3)
    v = vBA - tf.reduce_sum(vBA*uBC, axis=-1, keepdims=True)*uBC # (...,3)
    
    uBC1 = uBC[...,0] # (...,)
    uBC2 = uBC[...,1] # (...,)
    uBC3 = uBC[...,2] # (...,)
    
    zero = tf.zeros_like(uBC1) # (...,)
    S = tf.stack([tf.stack([ zero, uBC3,-uBC2],axis=-1),
                  tf.stack([-uBC3, zero, uBC1],axis=-1),
                  tf.stack([ uBC2,-uBC1, zero],axis=-1)],axis=-1) # (...,3,3)
    
    y = tf.expand_dims(tf.einsum('...j,...jk,...k->...',w,S,v), axis=-1) # (...,1)
    x = tf.expand_dims(tf.einsum('...j,...j->...',w,v), axis=-1)         # (...,1)
    
    phi = tf.math.atan2(y,x) # (...,1)
    
    return phi # (...,1)

def r_to_x_atom_(R, row_ABCD_IC):
    ''' transforms Cartesian coordiante of one atom into internal coordinate (IC) [i.e., a Z-matrix coordinate]
        r_to_x_atom_ : r_{A} -> x_{A}
    Inputs:
        R            : (..., n_atoms_mol, 3) ; n_atoms_mol = number of atoms in a single molecule
        row_ABCD_IC  : (4,) ; four indices of atoms in the molecule [A,B,C,D]
            the indices corespond to atoms that are covalently bonded A-B-C-D
    Output:
        x_IC         : (..., 3) ; x_IC = x_{A}
    '''
    bond     = get_distance_tf_( R, row_ABCD_IC[:2] )   # (...,1)
    angle    = get_angle_tf_(    R, row_ABCD_IC[:3] )   # (...,1)
    torsion  = get_torsion_tf_(  R, row_ABCD_IC[:4] )   # (...,1)
    x_IC = tf.concat([bond, angle, torsion],axis=-1)    # (...,3)
    return x_IC

def IC_ladJ_inv_(X_IC):
    ''' log volume change of the internal coordinate representation
    Input: 
        X_IC : (..., n_atoms, 3) where the last axis must be [bond, angle, torsion]
    Output:
        ladJ_inv : (..., 1) log volume change in the inverse direction (x -> r)
            = - log(det(Jacobian)) of the forward Jacobian = dx(r)/dr ; x(r) = r_to_x_atom_
    this is d*S2 coordinate ; d > 0
        size the sphere (bond distance) and the longitude (bond angle) contribute to local volume
    '''
    bonds  = X_IC[...,:1]                             # (..., n_atoms, 1)
    angles = X_IC[...,1:2]                            # (..., n_atoms, 1)
    ladJ_inv = tf.math.log(tf.sin(angles)*(bonds**2)) # (..., n_atoms, 1)
    ladJ_inv = tf.reduce_sum(ladJ_inv, axis=-2)       # (..., 1) # sum along all atoms (in the -2 axis)
    return ladJ_inv

def IC_forward_(r, ABCD_IC):
    '''
    Inputs:
        r       : (m, n_mol, n_atoms_mol, 3) Cartesian coordiantes of the (single component) system
        ABCD_IC : (n_atoms_IC, 4) matrix of indices for all bonded atoms that are represented using ICs
    Ouputs:
        X_IC    : (m, n_mol, n_atoms_IC, 3) internal coordinates of the (single component) system
        ladJ    : (m, n_mol, 1) log volume change of the IC representation. Sum over n_mol later.
    '''
    X_IC = tf.stack([r_to_x_atom_(r, row_ABCD_IC) for row_ABCD_IC in  ABCD_IC], axis=-2)
    ladJ = - IC_ladJ_inv_(X_IC)
    return X_IC, ladJ

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## inverse IC (internal coordinates):

def NeRF_tf_(d, theta, phi, rB, rC, rD, no_ladJ = True):
    ''' REF: DOI 10.1002/jcc.20237
    
    xA -> rA (inverse of r_to_x_atom_) ; xA = [d, theta, phi]

    Inputs:
        internal coordinate of atom A (relative to atoms B, C and D):
            d     : (..., ) ; A-B bond distance
            theta : (..., ) ; ABC bond angle
            phi   : (..., ) ; AB-CD torsional angle
        constants:
            rB    : (..., 3) Cartesian coordinate of atom B (bonded to A)
            rC    : (..., 3) Cartesian coordinate of atom C (bonded to B)
            rD    : (..., 3) Cartesian coordinate of atom D (bonded to C)
        no_ladJ   : bool ; True because IC_ladJ_inv_ is a cheaper way to get the same number.

    Outputs:
        rA        : (..., 3) Cartesian coordinate of atom A
    '''

    uCB = unit_clipped_(rB-rC)                    # (...,3)
    uDC = unit_clipped_(rC-rD)                    # (...,3)
    uv  = unit_clipped_(tf.linalg.cross(uDC,uCB)) # (...,3)

    M = tf.stack([ uCB, tf.linalg.cross(uv, uCB), uv ], axis=-1) # (...,3,3)

    the = PI - theta # (m,)
    c_t = tf.cos(the) ; s_t = tf.sin(the) # (...,)
    c_p = tf.cos(phi) ; s_p = tf.sin(phi) # (...,)

    v = tf.stack([ d*c_t, d*s_t*c_p, d*s_t*s_p ], axis=-1) # (...,3)

    rA = rB + tf.einsum('...ij,...j->...i', M, v) # (....,3)

    if no_ladJ:
        return rA # (...,3)
    else:
        # not using because not needed for only volume.
        zero = tf.zeros_like(c_t) # (...,)
        partials = tf.stack([tf.stack([  c_t,     -d*s_t,      zero      ], axis=-1), # (...,3)
                             tf.stack([  s_t*c_p,  d*c_t*c_p, -d*s_t*s_p ], axis=-1), # (...,3)
                             tf.stack([  s_t*s_p,  d*c_t*s_p,  d*s_t*c_p ], axis=-1), # (...,3)
                            ], axis=-2) # (...,3,3)
        jacobian_drA_dxA = tf.einsum('...ij,...jk->...ik', M, partials) # (...,3,3)
        ladJ = tf.math.log(tf.abs(det_3x3_(jacobian_drA_dxA, keepdims=True)))
        return rA, ladJ # (...,3), (...,1)

def IC_inverse_(X_IC, r_CB,
                ABCD_IC_inverse : list,
                inds_unpermute_atoms,
                ):
    '''
    reconstruct Cartesian coordinates of a molecule from internal coordinates
        Cartesian coordinates of the first 3 atoms are specified already (r_CB)
        All other atoms of the molecule reconstructed one-by-one using NeRF_tf_
    Inputs:
        X_IC : (..., n_atoms_IC, 3) ; internal coordinates of atoms that are 
        r_CB : (..., 3, 3) ; the Cartesian coordinates of the 'Cartesian block' (CB)
        ABCD_IC_inverse : (n_atoms_IC, 4) ; n_atoms_IC == n_atoms_mol - 3
            same as ABCD_IC used for the forward transfromation but the order of the rows is now different
            the order of the rows is set such that NeRF_tf_ is 'walking' along the molecule, (i.e.,
            any new atom (rA) being reconstructed has [rB, rC, rD] already reconstructed earlier).
        inds_unpermute_atoms : (n_atoms_mol,) indices to permute the atoms in the molecule back to original order
    Outputs:
        R    : (..., n_atoms_mol, 3) ; Cartesian coordinates of the entire system
        ladJ : (..., 1) ; log volume change of the current reconstruction (sum over molecules later)
    '''
    distances = X_IC[...,0] # (..., n_atoms_IC)
    angles    = X_IC[...,1] # (..., n_atoms_IC)
    torsions  = X_IC[...,2] # (..., n_atoms_IC)

    R = [r_CB[...,i,:] for i in range(3)]

    for i in range(len(ABCD_IC_inverse)):
        A, B, C, D = ABCD_IC_inverse[i] # i = A-3 probably
        rA = NeRF_tf_(distances[...,i], angles[...,i], torsions[...,i], R[B], R[C], R[D])
        R.append(rA)

    R = tf.stack(R, axis=-2)
    R = tf.gather(R, inds_unpermute_atoms, axis=-2)

    ladJ = IC_ladJ_inv_(X_IC)

    return R, ladJ

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## forward CB (Cartesian block):

def cond_0_true_(R):
    ''' for mat_to_quat_tf_ '''
    tr = tf.linalg.trace(R)
    s = tf.sqrt(tr + 1.0) * 2.0
    qw = 0.25 * s
    qx = (R[..., 2, 1] - R[..., 1, 2]) / s
    qy = (R[..., 0, 2] - R[..., 2, 0]) / s
    qz = (R[..., 1, 0] - R[..., 0, 1]) / s
    return tf.stack([qw, qx, qy, qz], axis=-1)

def cond_1_true_(R):
    ''' for mat_to_quat_tf_ '''
    s = tf.sqrt(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]) * 2.0
    qw = (R[..., 2, 1] - R[..., 1, 2]) / s
    qx = 0.25 * s
    qy = (R[..., 0, 1] + R[..., 1, 0]) / s
    qz = (R[..., 0, 2] + R[..., 2, 0]) / s
    return tf.stack([qw, qx, qy, qz], axis=-1)

def cond_2_true_(R):
    ''' for mat_to_quat_tf_ '''
    s = tf.sqrt(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]) * 2.0
    qw = (R[..., 0, 2] - R[..., 2, 0]) / s
    qx = (R[..., 0, 1] + R[..., 1, 0]) / s
    qy = 0.25 * s
    qz = (R[..., 1, 2] + R[..., 2, 1]) / s    
    return tf.stack([qw, qx, qy, qz], axis=-1)

def all_conds_false_(R):
    ''' for mat_to_quat_tf_ '''
    s = tf.sqrt(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]) * 2.0
    qw = (R[..., 1, 0] - R[..., 0, 1]) / s
    qx = (R[..., 0, 2] + R[..., 2, 0]) / s
    qy = (R[..., 1, 2] + R[..., 2, 1]) / s
    qz = 0.25 * s 
    return tf.stack([qw, qx, qy, qz], axis=-1)

def mat_to_quat_tf_(R):
    ''' REF: arXiv:2301.11355 (rigid body flows)

    transformation: rotation matrix (R) -> quaternion (q)
    ''' 
    shape = R.shape[:-2] ; M = tf.reduce_prod(shape)
    R = tf.reshape(R, [M] + [3,3])
    tr = tf.linalg.trace(R)
    
    inds_cond0_T = tf.where(tr > 0.0)[:,0]
    R_cond0_T = tf.gather(R,inds_cond0_T,axis=0)
    q_cond0_T = cond_0_true_(R_cond0_T)
    
    inds_cond0_F = tf.where(tr <= 0.0)[:,0]
    R_cond0_F = tf.gather(R,inds_cond0_F,axis=0)
    
    inds_cond0_F_cond1_T = tf.where((R_cond0_F[...,0, 0] > R_cond0_F[...,1, 1]) & (R_cond0_F[...,0, 0] > R_cond0_F[...,2, 2]))[:,0]
    R_cond0_F_cond1_T = tf.gather(R_cond0_F,inds_cond0_F_cond1_T,axis=0)
    q_cond0_F_cond1_T = cond_1_true_(R_cond0_F_cond1_T)
    
    inds_cond0_F_cond1_F = tf.where((R_cond0_F[...,0, 0] <= R_cond0_F[...,1, 1]) | (R_cond0_F[...,0, 0] <= R_cond0_F[...,2, 2]))[:,0]
    R_cond0_F_cond1_F = tf.gather(R_cond0_F,inds_cond0_F_cond1_F,axis=0)
    
    inds_cond0_F_cond1_F_cond2_T = tf.where(R_cond0_F_cond1_F[...,1, 1] > R_cond0_F_cond1_F[...,2, 2])[:,0]
    R_cond0_F_cond1_F_cond2_T = tf.gather(R_cond0_F_cond1_F, inds_cond0_F_cond1_F_cond2_T, axis=0)
    q_cond0_F_cond1_F_cond2_T = cond_2_true_(R_cond0_F_cond1_F_cond2_T)
    
    inds_cond0_F_cond1_F_cond2_F = tf.where(R_cond0_F_cond1_F[...,1, 1] <= R_cond0_F_cond1_F[...,2, 2])[:,0]
    R_cond0_F_cond1_F_cond2_F = tf.gather(R_cond0_F_cond1_F, inds_cond0_F_cond1_F_cond2_F, axis=0)
    q_cond0_F_cond1_F_cond2_F = all_conds_false_(R_cond0_F_cond1_F_cond2_F)
    
    q_cond0_F_cond1_F = tf.gather(tf.concat([q_cond0_F_cond1_F_cond2_T,q_cond0_F_cond1_F_cond2_F],axis=0),
                        tf.argsort(tf.concat([inds_cond0_F_cond1_F_cond2_T,inds_cond0_F_cond1_F_cond2_F],axis=0)))
    
    q_cond0_F = tf.gather(tf.concat([q_cond0_F_cond1_T, q_cond0_F_cond1_F],axis=0),
                tf.argsort(tf.concat([inds_cond0_F_cond1_T, inds_cond0_F_cond1_F],axis=0)))
        
    q = tf.gather(tf.concat([q_cond0_T, q_cond0_F],axis=0),
        tf.argsort(tf.concat([inds_cond0_T, inds_cond0_F],axis=0)))
        
    return tf.reshape(q, shape+[4])

def CB_ladJ_inv_(a, d0, d1):
    ''' *REF: arXiv:2301.11355 (rigid body flows) 
    log volume change of transforming between 'Cartesian block' (CB) and the representation of CB from *REF
    
    Cartesian block (e.g,. a water molecule H0 - O - H1) is : r_{CB} = [r_{O}, r_{H0}, r_{H1}]
    The representation is: x_{CB} = [r_{O}, a, d0, d1, q]

    Inputs:
        a  : angle between bonds H0 - O and H1 - O
        d0 : bond distance between H0 and O
        d1 : bond distance between H1 and O
    Output:
        ladJ_inv : log volume change of transformation x_{CB} -> r_{CB}

    '''
    d0d0 = d0**2
    d1d1 = d1**2
    ladJ_inv = tf.math.log(8.0 * d0d0 * d1d1 * tf.sin(a)) + 0.5 * tf.math.log(4.0 * (d0d0 + d1d1) + 1)
    #if forward: return -ladJ_inv
    #else:       return  ladJ_inv
    return ladJ_inv

def CB_forward_(xyz_CB):
    ''' REF: arXiv:2301.11355 (rigid body flows)

    Input:
        xyz_CB : (..., 3, 3) Cartesian coordinates of the 'Cartesian block' (CB) = r_{CB} = [rA, rB, rC]

    Outputs:
        X       : list of 5 arrays [rA, q, a, dAB, dAC] = x_{CB}
            rA  : (..., 3) ; Cartesian coordinate of atom A unchanged
            q   : (..., 4) ; unit-quaternion describing the rotation of the CB
            a   : (..., 1) ; angle between bonds rB - rA and rC - rA
            dAB : (..., 1) ; bond distance between rB and rA
            dAC : (..., 1) ; bond distance between rC and rA
        ladJ    : (..., 1) ; log volume change of the transformation r_{CB} -> x_{CB}

    '''

    rA = xyz_CB[...,0,:]
    rB = xyz_CB[...,1,:]
    rC = xyz_CB[...,2,:]

    rAB = rB - rA ; dAB = tf.linalg.norm(rAB, axis=-1, keepdims=True)
    rAC = rC - rA ; dAC = tf.linalg.norm(rAC, axis=-1, keepdims=True)

    uOP1 = rAB / dAB
    uOOP = unit_clipped_(tf.linalg.cross(rAB,rAC))
    uOP2 = unit_clipped_(tf.linalg.cross(uOOP,uOP1))

    R = tf.stack([uOP2,uOOP,uOP1], axis=-1)
    
    q = mat_to_quat_tf_(R)

    dot = tf.clip_by_value(tf.einsum('...i,...i->...',uOP1,rAC/dAC), -1.0, 1.0)
    a = tf.clip_by_value(tf.acos(dot), _clip_low_at_, PI-_clip_low_at_) # \in (0,pi)
    a = tf.expand_dims(a, axis=-1)

    X = [rA, q, a, dAB, dAC]
    
    # rA : (...,3) \in T3        # the range later set to data range after centered and COM removed
    # q  : (...,4) \in S3        # the range later set in s(q*p) ; p static, from data using FocusedHemisphere
    # a  : (...,1) \in [0, pi]   # the range later set to [a_min, a_max] from data using FocusedAngles
    # dAB: (...,1) \in [0, \inf] # the range later set to [d_min, d_max] from data using FocusedBonds
    # dAC: (...,1) \in [0, \inf] # the range later set to [d_min, d_max] from data using FocusedBonds
    
    ladJ = - CB_ladJ_inv_(a, dAB, dAC) # (...,1)

    return X, ladJ

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## inverse CB (Cartesian block):

def quat_to_mat_tf_(q):
    ''' REF: INTRODUCTION TO ROBOTICS MECHANICS, PLANNING, AND CONTROL F. C. Park and K. M. Lynch

    transformation: quaternion (q) -> rotation matrix (R)
    '''
    q0, q1, q2, q3 = q[...,0], q[...,1], q[...,2], q[...,3]
    
    R = tf.stack([tf.stack([ q0**2 + q1**2 - q2**2 - q3**2 , 2.*(q1*q2-q0*q3)              , 2.*(q0*q2+q1*q3) ],axis=-1),
                  tf.stack([ 2.*(q0*q3+q1*q2)              , q0**2 - q1**2 + q2**2 - q3**2 , 2.*(q2*q3-q0*q1) ],axis=-1),
                  tf.stack([ 2.*(q1*q3-q0*q2)              , 2.*(q0*q1+q2*q3)              , q0**2 - q1**2 - q2**2 + q3**2 ],axis=-1),
                 ],axis=-2)
    return R

def CB_inverse_(X):
    ''' REF: arXiv:2301.11355 (rigid body flows)

    inverse of CB_forward_

    Inputs:
        X      : list of 5 arrays [rA, q, a, dAB, dAC] = x_{CB}
            explained in CB_forward_
    Outputs:
        xyz_CB : (..., 3, 3) Cartesian coordinates of the 'Cartesian block' (CB) = r_{CB}
        ladJ   : (..., 1) ; log volume change of the transformation x_{CB} -> r_{CB}
    '''

    rA, q, a, dAB, dAC = X
    
    R = quat_to_mat_tf_(q)

    I = tf.broadcast_to(tf.eye(3), tf.shape(R))
    
    rB = tf.einsum('...ij,...j->...i', R, I[...,-1]*dAB) + rA

    rC = tf.einsum('...ij,...j->...i', R, I[...,0]*tf.sin(a) + I[...,-1]*tf.cos(a))*dAC + rA

    xyz_CB = tf.stack([rA,rB,rC], axis=-2)

    ladJ = CB_ladJ_inv_(a, dAB, dAC)

    return xyz_CB, ladJ

def test_CB_transformation_(n_frames=1000, n_molecules=40):
    '''
    comparing the analytical expression of log volume change in CB_ladJ_inv_ to a numerical analogue (using full jacobian)
    comparison on random Cartesian blocks
    comparisons show that the closed-form expression is correct and very efficient
    '''
    xyz = np2tf_( np.random.randn(n_frames, n_molecules, 3*3) )

    X_list, ladJ_forward = CB_forward_( tf.reshape(xyz, [n_frames, n_molecules, 3, 3]) )
    
    X = tf.concat( X_list, axis=-1)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)

        _xyz, ladJ_inverse = CB_inverse_([X[...,:3], X[...,3:7], X[...,7:8], X[...,8:9], X[...,9:10]])
        
        _xyz = tf.reshape(_xyz, [n_frames, n_molecules, 9])
    
        rs = [_xyz[...,i] for i in range(9)]
    
    J = tf.stack([tape.gradient(r,X) for r in rs],axis=-2).numpy()
    
    print('inversion accuracy (coordiantes): abs(xyz-_xyz).max() :', np.abs(xyz-_xyz).max())
    print('J_Cartesian(9)_by_model(10) shape:', J.shape)
    
    ladJ_explicit_inverse = 0.5 * np.log(np.linalg.svd(np.einsum('...ij,...ik->...jk',J,J))[1][...,:9]).sum(-1,keepdims=True)
    
    print('volume accuracy : abs(ladJ_explicit_inverse - ladJ_analytical_inverse).max() :', np.abs(ladJ_explicit_inverse - ladJ_inverse).max())
    #print(np.abs(ladJ_inverse+ladJ_forward).sum()) # 0

#test_CB_transformation_(n_frames=1000, n_molecules=100) # ~ no error

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## CB single molecule:

def CB_single_molecule_forward_(xyz_CB):
    ''' for single molecule in vaccum
    TODO: was not tested properly yet, but seems to work
    '''
    rA, q, a, dAB, dAC = CB_forward_(xyz_CB)[0]
    # a  : (...,1) \in [0,pi]    # actully [a_min, a_max] from data
    # dAB: (...,1) \in [0, \inf] # actully [d_min, d_max] from data
    # dAC: (...,1) \in [0, \inf] # actully [d_min, d_max] from data
    '''
    # potential energy of single_molecule in vaccum is invariant to rigid transformations:
    # this means that the energy is the same for any choice of [rA,q], 
    # this means that can give any [rA,q] to the molecule during inversion
    # this means that can drop the redundant [rA,q] information here.

    # suppose q = [1,0,0,0] is given during inversion (rotation matrix = eye(3))
    # suppose rA = [0,0,0]  is given during inversion

    # plugging this choice of [rA,q] into CB_inverse_ gives:
        [rA,rB,rC] = [[0, 0,   dAC*sin(a)],
                      [0, 0,   0         ],
                      [0, dAB, dAC*cos(a)]]
        ; i.e, 
        rA = [0,0,0]
        rB = [0,0,dAB]
        rC = [dAC*sin(a), 0, dAC*cos(a)]
        
        therefore:
            rB[2] = dAB ; slope 1, volume change 0

            rC[0] = dAC*sin(a)
            rC[2] = dAC*cos(a)
            # this is polar coordianates
            # transforming TO polar coordianates (forward), means that log volue change is:
            ladJ = - log(dAC)
    '''

    ladJ = - tf.math.log(dAC)

    return [a, dAB, dAC], ladJ

def CB_single_molecule_inverse_(X):

    a, dAB, dAC = X

    rA = tf.concat([tf.zeros_like(a)]*3, axis=-1)
    q = tf.concat([tf.ones_like(a), rA], axis=-1)

    xyz_CB = CB_inverse_([rA, q, a, dAB, dAC])[0]

    ladJ = tf.math.log(dAC) # explained in CB_single_molecule_forward_

    return xyz_CB, ladJ

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## hemisphere:

hemisphere_ = lambda x : x * tf.where(x[...,:1]<0.0, -1.0, 1.0)

def hemisphere_forward_(q, rescale_marginals=True):
    '''
    REF: https://doi.org/10.1021/acs.jctc.4c01612
    '''
    '''
    R = tf.linalg.norm(c, axis=-1) ; Rsq  = R**2
    sq_n_c1c2c3 = n_c1c2c3**2
    sq_n_c2c3   = n_c2c3**2
    zero = tf.zeros_like(p)
    J = tf.stack(
    [   c/tf.expand_dims(R, axis=-1),
        tf.stack([-n_c1c2c3/Rsq,  (c0*c1)/(Rsq*n_c1c2c3),  (c0*c2)/(Rsq*n_c1c2c3),       (c0*c3)/(Rsq*n_c1c2c3)      ], axis=-1),
        tf.stack([zero,           -n_c2c3/sq_n_c1c2c3,     (c1*c2)/(sq_n_c1c2c3*n_c2c3), (c1*c3)/(sq_n_c1c2c3*n_c2c3)], axis=-1),
        tf.stack([zero,            zero,                   -c3/sq_n_c2c3,                 c2/sq_n_c2c3               ], axis=-1),
    ], axis=-1) # ds_dc 
    detJ = tf.expand_dims(tf.linalg.det(J), axis=-1)
    ladJ = tf.math.log( tf.abs( detJ ) )
    '''

    c = hemisphere_(q) # (...,4) volume preserving log(abs(-1 or 1)) = 0 ; TODO: check/include full derivation in report.
    # no information is lost because R(q) = R(-q)
    # hemisphere edge is still problematic, so rotate input (q) a priori away from hemisphere edge

    n_c1c2c3 = clip_positive_(tf.linalg.norm(c[...,1:], axis=-1))
    n_c2c3   =  clip_positive_(tf.linalg.norm(c[...,2:], axis=-1))

    c0 = c[...,0]
    c1 = c[...,1]
    c2 = c[...,2]
    c3 = c[...,3]

    t0 = tf.atan2(n_c1c2c3,c0) # tf.acos(c0/R)
    t1 = tf.atan2(n_c2c3,c1)   # tf.acos(c1/n_c1c2c3)
    p = tf.atan2(c3,c2)
    s = tf.stack([t0,t1,p], axis=-1)

    sqrt_det_g = (tf.sin(t0)**2) * tf.sin(t1)
    ladJ = - tf.expand_dims(tf.math.log( clip_positive_(sqrt_det_g) ), axis=-1)

    if rescale_marginals:
        xq, ladj_rescale = scale_shift_x_(s,
                                          physical_ranges_x = [PI*0.5, PI, PI*2.0],
                                          physical_centres_x = [PI*0.25, PI*0.5, 0.0],
                                          forward = True)
        ladJ += ladj_rescale
    else: xq = s
    return xq, ladJ

def hemisphere_inverse_(xq, rescale_marginals=True):
    '''
    REF: https://doi.org/10.1021/acs.jctc.4c01612
    '''
    '''
    zero = tf.zeros_like(R)
    J = tf.stack(
    [   tf.stack([ct0,       -R*st0,         zero,          zero        ], axis=-1),
        tf.stack([st0*ct1,    R*ct0*ct1,    -R*st0*st1,     zero        ], axis=-1),
        tf.stack([st0*st1*cp, R*ct0*st1*cp,  R*st0*ct1*cp, -R*st0*st1*sp], axis=-1),
        tf.stack([st0*st1*sp, R*ct0*st1*sp,  R*st0*ct1*sp,  R*st0*st1*cp], axis=-1),
    ], axis=-1) # dc_ds 
    detJ = tf.expand_dims(tf.linalg.det(J), axis=-1)
    ladJ = tf.math.log( tf.abs( detJ ) )
    '''
    ladJ = 0.0
    if rescale_marginals:
        s, ladj_rescale = scale_shift_x_(xq,
                                        physical_ranges_x = [PI*0.5, PI, PI*2.0],
                                        physical_centres_x = [PI*0.25, PI*0.5, 0.0],
                                        forward = False) # (m,n,3) -> (m,n,3)
        ladJ += ladj_rescale
    else: s = xq
    ##

    t0 = s[...,0] ; ct0 = tf.cos(t0) ; st0 = tf.sin(t0)
    t1 = s[...,1] ; ct1 = tf.cos(t1) ; st1 = tf.sin(t1)
    p  = s[...,2] ; cp  = tf.cos(p)  ; sp  = tf.sin(p)
    c = tf.stack([ct0, st0*ct1, st0*st1*cp, st0*st1*sp], axis=-1)
    sqrt_det_g = (st0**2) * st1
    ladJ += tf.expand_dims(tf.math.log( clip_positive_(sqrt_det_g) ), axis=-1)

    ##
    qh = c
    return qh, ladJ #, Js # (...,4), (...,1), (...,4,4)

##

def sample_q_(shape : list):
    ''' samples random 4D unit-vector '''
    '''
    uvw = tf.random.uniform(shape + [3], minval=0.0, maxval=1.0, dtype=tf.float32) 
    u = uvw[...,0] ; v = uvw[...,1] ; w = uvw[...,2]
    q = tf.stack(
    [tf.sqrt(1.-u)*tf.sin(2.*PI*v),
     tf.sqrt(1.-u)*tf.cos(2.*PI*v),
     tf.sqrt(u)*tf.sin(2.*PI*w),
     tf.sqrt(u)*tf.cos(2.*PI*w)
    ], axis = -1)
    '''
    q = unit_clipped_(tf.random.normal(shape+[4]))
    return q

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## misc:

def quat_metrix_(q, inverse=False):
    '''  for quaternion product done as a matrix multiplication (less efficent)
    Inputs:
        q       : (..., 4)    ; unitvector in R^4
    Ouput:
        R4      : (..., 4, 4) ; rotation matrix in R^4
        inverse : bool        ; if True, R4.transpose() = inv(R4) returned
    Usage: in FocusedHemisphere
        Used in Static_Rotations_Layer to rotate MD data away from problematic regions 
        of the hyperspherical representation of 'Cartesian block' rotations (s = xq).
    '''
    q0 = q[...,0]
    q1 = q[...,1]
    q2 = q[...,2]
    q3 = q[...,3]
    if inverse: axis = -1
    else: axis = -2
    R4 = tf.stack([ tf.stack([ q0, -q1, -q2, -q3],axis=-1),
                    tf.stack([ q1,  q0, -q3,  q2],axis=-1),
                    tf.stack([ q2,  q3,  q0, -q1],axis=-1),
                    tf.stack([ q3, -q2,  q1,  q0],axis=-1), ], axis=axis)
    return R4

def quat_product_(q,p):
    ''' closed-from quaternion product (more efficent) '''
    q0 = q[...,:1]; qv = q[...,1:]
    p0 = p[...,:1]; pv = p[...,1:]
    w0 = q0*p0 - tf.einsum('...i,...i->...',qv,pv)[...,tf.newaxis]
    wv = q0*pv + p0*qv + tf.linalg.cross(qv,pv)
    w = tf.concat([w0,wv],axis=-1)
    return w

def quat_inverse_(q):
    # ! assumes the input is already a unit-vector
    # allows quat_product_ to be inverted
    q0 = q[...,:1]
    q123 = q[...,1:]
    return tf.concat([q0,-q123],axis=-1)

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
## box: [placeholder only; further work]

def box_forward_(b):
    ' placeholder '
    ladJ = 0.0 # (m,1)
    h = tf.stack([b[...,0,0],b[...,1,1],b[...,2,2],b[...,1,0],b[...,2,0],b[...,2,1]],axis=-1)
    return h, ladJ

def box_inverse_(h):
    ' placeholder '
    ladJ = 0.0 # (m,1)
    zero = tf.zeros_like(h[...,0])
    b = tf.stack([tf.stack([ h[...,0], zero    , zero    ],axis=-1),
                  tf.stack([ h[...,3], h[...,1], zero    ],axis=-1),
                  tf.stack([ h[...,4], h[...,5], h[...,2]],axis=-1),], axis=-2)
    return b, ladJ




