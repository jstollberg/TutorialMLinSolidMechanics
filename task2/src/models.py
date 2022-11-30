"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I

==================

Authors: Henrik Hembrock, Jonathan Stollberg

11/2022
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import trace, inv, det
from tensorflow.keras.constraints import non_neg
from utils import tensor_to_voigt, voigt_to_tensor

class InvariantsTransIso(layers.Layer):
    def call(self, F):
        # convert to tensor notation if neccessary
        if len(F.shape) == 2:
            F = voigt_to_tensor(F)

        # define transersely isotropic structural tensro
        G = np.array([[4.0, 0.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0, 0.0, 0.5]])
        G = tf.convert_to_tensor(G, dtype=F.dtype)
        G = tf.tile(G, [len(F),1])
        G = tf.reshape(G, (len(F),3,3))
        
        # compute right Cauchy-Green tensor
        C = tf.linalg.matrix_transpose(F)*F
        cofC = tf.reshape(det(C), (len(F),1,1))*inv(C)
        
        # compute invariants
        I1   = trace(C)
        J    = det(F)
        I4   = trace(C*G)
        I5   = trace(cofC*G)
        
        # collect all invariants in one tensor
        ret = tf.stack([I1, J, -J, I4, I5], axis=1)
    
        return ret
    
class InvariantsCubic(layers.Layer):
    def call(self, F):
        # convert to tensor notation if neccessary
        if len(F.shape) == 2:
            F = voigt_to_tensor(F)
            
        # compute right Cauchy-Green tensor
        C = tf.linalg.matrix_transpose(F)*F
        cofC = tf.reshape(det(C), (len(F),1,1))*inv(C)
        
        # compute invariants
        I1  = trace(C)
        I2  = trace(cofC)
        J   = det(F)
        I7  = 
        I11 = 
        
        # collect all invariants in one tensor
        ret = tf.stack([I1, I2, J, -J, I7, I11], axis=1)
            


class StrainEnergyTransIso(layers.Layer):
    def call(self, invariants):
        # extract invariants
        I1 = invariants[:,0]
        J  = invariants[:,1]
        I4 = invariants[:,3]
        I5 = invariants[:,4]
        
        # compute strain energy
        W = 8*I1 + 10*J**2 - 56*tf.math.log(J) + 0.2*(I4**2 + I5**2) - 44
        W = tf.reshape(W, (len(invariants),1))
        
        return W
    
class PiolaKirchhoffTransIso(layers.Layer):
    def call(self, F, strain_energy):
        with tf.GradientTape() as tape:
            tape.watch(F)
            I = InvariantsTransIso()(F)
            W = strain_energy(I)
        P = tape.gradient(W, F)
        
        return P, W
    
class MS(tf.keras.Model):
    def __init__(self,
                 nlayers=3,
                 units=8):
        super(MS, self).__init__()
        self.ls = [layers.Dense(units, activation="softplus", 
                                input_shape=(9,))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, activation="softplus")]
        self.ls += [layers.Dense(9)]
      
    def call(self, C):
        for l in self.ls:
            C = l(C)
        return C
    
class WI(tf.keras.Model):
        def __init__(self,
                     nlayers=3,
                     units=8):
            super(WI, self).__init__()
            self.ls = [layers.Dense(units, activation="softplus",
                                    kernel_constraint=non_neg(), 
                                    input_shape=(5,))]
            for l in range(nlayers - 1):
                self.ls += [layers.Dense(units, activation="softplus",
                                         kernel_constraint=non_neg())]
            self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
            
        def call(self, F):
            P , W = PiolaKirchhoffTransIso()(F, self._strain_energy)
            return P, W
        
        def _strain_energy(self, I):
            for l in self.ls:
                I = l(I)
            return I