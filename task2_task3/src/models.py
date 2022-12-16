"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I
Task 3: Hyperelasticity II

==================

Authors: Henrik Hembrock, Jonathan Stollberg

12/2022
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import trace, det, matmul, matrix_transpose
from tensorflow.keras.constraints import non_neg
from utils import voigt_to_tensor, tensor_to_voigt, cofactor

class InvariantsTransIso(layers.Layer):
    def call(self, F):
        """
        Compute the invariants for a transvers isotropic material.
        
        The order of the returned invariants is (I1, J, -J, I4, I5).

        Parameters
        ----------
        F : tensorflow.Tensor
            The deformation gradient in Voigt or tensor notation.

        Returns
        -------
        ret : tensorflow.Tensor
            The invariants.

        """
        # convert to tensor notation if neccessary
        if F.shape.ndims == 2:
            F = voigt_to_tensor(F)

        # define transersely isotropic structural tensro
        G = np.array([[4.0, 0.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0, 0.0, 0.5]])
        G = tf.convert_to_tensor(G, dtype=F.dtype)
        G = tf.tile(G, [len(F),1])
        G = tf.reshape(G, (len(F),3,3))
        
        # compute right Cauchy-Green tensor
        C = matmul(matrix_transpose(F), F)
        cofC = cofactor(C)
        
        # compute invariants
        I1   = trace(C)
        J    = det(F)
        I4   = trace(matmul(C, G))
        I5   = trace(matmul(cofC, G))
        
        # collect all invariants in one tensor
        ret = tf.stack([I1, J, -J, I4, I5], axis=1)
    
        return ret
    
class InvariantsCubic(layers.Layer):
    def call(self, F):
        """
        Compute the invariants for a cubic material.
        
        The order of the returned invariants is (I1, I2, J, -J, I7, I11).

        Parameters
        ----------
        F : tensorflow.Tensor
            The deformation gradient in Voigt or tensor notation.

        Returns
        -------
        ret : tensorflow.Tensor
            The invariants.

        """
        # convert to tensor notation if neccessary
        if F.shape.ndims == 2:
            F = voigt_to_tensor(F)
            
        double_dot_2 = lambda a, b: tf.einsum("mij,mij->m", a, b)
        double_dot_4 = lambda a, b: tf.einsum("mijkl,mkl->mij", a, b)
            
        # define cubic structural tensor
        G = np.zeros((3,3,3,3))
        G[0,0,0,0] = 1.0
        G[1,1,1,1] = 1.0
        G[2,2,2,2] = 1.0
        G = tf.convert_to_tensor(G, dtype=F.dtype)
        G = tf.tile(G, [len(F),1,1,1])
        G = tf.reshape(G, (len(F),3,3,3,3))
            
        # compute right Cauchy-Green tensor
        C = matmul(matrix_transpose(F), F)
        cofC = cofactor(C)
        
        # compute invariants
        I1  = trace(C)
        I2  = trace(cofC)
        J   = det(F)
        I7  = double_dot_2(double_dot_4(G, C), C)
        I11 = double_dot_2(double_dot_4(G, cofC), cofC)
        
        # collect all invariants in one tensor
        ret = tf.stack([I1, I2, J, -J, I7, I11], axis=1)
            
        return ret
    
class StrainEnergyTransIso(layers.Layer):
    def call(self, invariants):
        """
        Compute the strain energy for a transverse isotropic material.
        
        Reference for the energy formulation:
            J. Schröder, P. Neff and V. Ebbing. “Anisotropic polyconvex 
            energies on the basis of crystallographic motivated structural 
            tensors”. In: Journal of the Mechanics and Physics of Solids 56 
            (2008), pp. 3486–3506. doi: 10.1016/j.jmps.2008.08.008.

        Parameters
        ----------
        invariants : tensorflow.Tensor
            The invariants in order (I1, J, -J, I4, I5).

        Returns
        -------
        W : tensorflow.Tensor
            The strain energy.

        """
        # extract invariants
        I1 = invariants[:,0]
        J  = invariants[:,1]
        I4 = invariants[:,3]
        I5 = invariants[:,4]
        
        # compute strain energy
        W = 8*I1 + 10*J**2 - 56*tf.math.log(J) + 0.2*(I4**2 + I5**2) - 44
        W = tf.reshape(W, (len(invariants),1))
        
        return W
    
class PiolaKirchhoff(layers.Layer):
    def call(self, F, invariants, strain_energy):
        """
        Compute the first Piola-Kirchhoff-stress.
        
        The stress is computed by differentiating a strain energy formulated
        in terms of the invariants w.r.t. the deformation gradient.

        Parameters
        ----------
        F : tensorflow.Tensor
            The deformation gradient.
        invariants : function
            A function that computes the invariants and takes F as input.
        strain_energy : function
            A function that evaluates the strain energy and takes the
            invariants as input.

        Returns
        -------
        P : tensorflow.Tensor
            The first Piola-Kirchhoff stress.
        W : tensorflow.Tensor
            The strain energy.

        """
        with tf.GradientTape() as tape:
            tape.watch(F)
            I = invariants(F)
            W = strain_energy(I)
        P = tape.gradient(W, F)
        
        return P, W
    
class ModelMS(tf.keras.Model):
    """
    FFNN that maps the right Cauchy-Green strain directly to the first
    Piola-Kirchhoff stress.

    Parameters
    ----------
    nlayers : int, optional
        Number of hidden layers. The default is 3.
    units : int, optional
        Number of nodes per hidden layer. The default is 8.
        
    Attributes
    ----------
    ls : list
        The list of hidden layers.
        
    Methods
    -------
    call:
        Evaluate the FFNN.
        
    """
    def __init__(self,
                 nlayers=3,
                 units=8):
        super(ModelMS, self).__init__()
        self.ls = [layers.Dense(units, activation="softplus", 
                                input_shape=(6,))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, activation="softplus")]
        self.ls += [layers.Dense(9)]
      
    def call(self, C):
        """
        Evaluate the FFNN.

        Parameters
        ----------
        C : tensorflow.Tensor
            The right Cauchy-Green strain in Voigt notation.

        Returns
        -------
        C : tensorflow.Tensor
            The first Piola-Kirchhoff stress in Voigt notation.

        """
        for l in self.ls:
            C = l(C)
        return C
    
class ModelWI(tf.keras.Model):
    """
    ICNN that models the strain energy based on invariants.

    Parameters
    ----------
    invariants : function
        A function that computes the invariants and takes F as input.
    nlayers : int, optional
        Number of hidden layers. The default is 3.
    units : int, optional
        Number of nodes per hidden layer. The default is 8.
        
    Attributes
    ----------
    ls : list
        The list of hidden layers.
    invariants : function
        The function that computes the invariants and takes F as input.
        
    Methods
    -------
    call:
        Evaluate the ICNN.
        
    """
    def __init__(self,
                 invariants,
                 nlayers=3,
                 units=8):
        super(ModelWI, self).__init__()
        self.ls = [layers.Dense(units, activation="softplus",
                                kernel_constraint=non_neg(), 
                                input_shape=(9,))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, activation="softplus",
                                     kernel_constraint=non_neg())]
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
        self.invariants = invariants
        
    def call(self, F):
        """
        Evaluate the ICNN.

        Parameters
        ----------
        F : tensorflow.Tensor
            The deformation gradient.

        Returns
        -------
        P : tensorflow.Tensor
            The first Piola-Kirchhoff-stress.
        W : tensorflow.Tensor
            The strain energy.

        """
        P , W = PiolaKirchhoff()(F, self.invariants, self._strain_energy)
        return P, W
    
    def _strain_energy(self, I):
        """The evaluation of the hidden layers to model the energy."""
        for l in self.ls:
            I = l(I)
        return I
        
class ModelWF(tf.keras.Model):
    """
    ICNN that models the strain energy based on the polyconvexity condition.

    Parameters
    ----------
    nlayers : int, optional
        Number of hidden layers. The default is 3.
    units : int, optional
        Number of nodes per hidden layer. The default is 8.
        
    Attributes
    ----------
    ls : list
        The list of hidden layers.
        
    Methods
    -------
    call:
        Evaluate the ICNN.
        
    """
    def __init__(self,
                 nlayers=3,
                 units=8):
        super(ModelWF, self).__init__()
        self.ls = [layers.Dense(units, activation="softplus", 
                                input_shape=(19,))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, activation="softplus",
                                     kernel_constraint=non_neg())]
        self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
        
    def call(self, F):
        """
        Evaluate the ICNN.

        Parameters
        ----------
        F : tensorflow.Tensor
            The deformation gradient.

        Returns
        -------
        P : tensorflow.Tensor
            The first Piola-Kirchhoff-stress.
        W : tensorflow.Tensor
            The strain energy.

        """
        with tf.GradientTape() as tape:
            tape.watch(F)
            
            # compute cofactor and determinant
            F = voigt_to_tensor(F)
            cofF = cofactor(F)
            detF = tf.reshape(det(F), (-1,1))
            F = tensor_to_voigt(F)
            cofF = tensor_to_voigt(cofF)
            
            # evaluate energy
            inp = tf.concat([F, cofF, detF], axis=1)
            W = self._strain_energy(inp)
            
        P = tape.gradient(W, F)

        return P, W
    
    def _strain_energy(self, inp):
        """The evaluation of the hidden layers to model the energy."""
        for l in self.ls:
            inp = l(inp)
        return inp