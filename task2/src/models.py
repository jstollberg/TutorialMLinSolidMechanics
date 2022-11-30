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
        C = tf.linalg.matrix_transpose(F)*F
        cofC = tf.reshape(det(C), (len(F),1,1))*inv(C)
        
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
    
class PiolaKirchhoffCubic(layers.Layer):
    def call(self, F, strain_energy):
        with tf.GradientTape() as tape:
            tape.watch(F)
            I = InvariantsCubic()(F)
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
                                    input_shape=(6,))]
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
        
# if __name__ == "__main__":
#     import os
#     import tensorflow as tf
#     from data import load_data, load_invariants, plot_load_path
#     from data import loc_bcc_uniaxial, loc_bcc_shear, loc_bcc_biaxial
#     from data import loc_biaxial_test, loc_mixed_test
#     from models import MS, WI
#     from utils import weight_L2, voigt_to_tensor, tensor_to_voigt
#     from data import plot_load_path
    
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    
#     # load data
#     F_biaxial, C_biaxial, P_biaxial, W_biaxial = load_data(loc_bcc_biaxial)
#     F_uniaxial, C_uniaxial, P_uniaxial, W_uniaxial = load_data(loc_bcc_uniaxial)
#     F_shear, C_shear, P_shear, W_shear = load_data(loc_bcc_shear)
    
#     # compute sample weights
#     weight = weight_L2(P_biaxial, P_uniaxial, P_shear)
    
#     training_in = tf.concat([F_biaxial, F_uniaxial, F_shear], axis=0)
#     training_out = [tf.concat([P_biaxial, P_uniaxial, P_shear], axis=0),
#                     tf.concat([W_biaxial, W_uniaxial, W_shear], axis=0)]
    
#     sample_weight = weight
#     loss_weights = None
#     kwargs = {"nlayers": 3, "units": 16}
#     epochs = 1000
    
#     model = WI(**kwargs)
#     model.compile("adam", "mse", loss_weights=loss_weights)
    
#     # fit to data
#     tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
#     h = model.fit(training_in, 
#                   training_out, 
#                   epochs=epochs, 
#                   sample_weight=sample_weight,
#                   verbose=2)
    
#     #%% interpolation
#     P, W = model.predict(F_biaxial)
#     plot_load_path(voigt_to_tensor(F_biaxial), voigt_to_tensor(P_biaxial))
#     plot_load_path(voigt_to_tensor(F_biaxial), voigt_to_tensor(P))
    
    