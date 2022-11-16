# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 19:42:02 2022

@author: jonat
"""

"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I

==================

Authors: Henrik Hembrock, Jonathan Stollberg

11/2022
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import trace, inv, det
from tensorflow.keras.constraints import non_neg

class Invariants(layers.Layer):
    def __call__(self, F):
        # define transersely isotropic structural tensro
        n = len(F)
        G = tf.constant([[[4.0, 0.0, 0.0],
                          [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5]]]*n, dtype=F.dtype)
        
        # compute right Cauchy-Green tensor
        C = tf.linalg.matrix_transpose(F)*F

        # compute invariants
        I1   = trace(C)
        J    = det(F)
        I4   = trace(C*G)
        cofC = tf.reshape(det(C), (len(F),1,1))*inv(C)
        I5   = trace(cofC*G)
        
        # collect all invariants in one tensor
        ret = tf.stack([I1, J, -J, I4, I5], axis=1)
    
        return ret

class StrainEnergy(layers.Layer):
    def __call__(self, invariants):
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
    def __call__(self, F, W):
        with tf.GradientTape() as tape:
            tape.watch(F)
            W = W(F)
        P = tape.gradient(W, F)
        
        return P, W
    
class PiolaKirchhoffFFNN(tf.keras.Model):
    def __init__(self,
                  nlayers=3,
                  units=8):
        super(PiolaKirchhoffFFNN, self).__init__()
        self.ls = [layers.Dense(units, activation="softplus")]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, activation="softplus")]
        self.ls += [layers.Dense(3)]
      
    def call(self, C):
        for l in self.ls:
            C = l(C)
        return C
    
class PiolaKirchhoffICNN(tf.keras.Model):
        def __init__(self,
                     nlayers=3,
                     units=8):
            super(PiolaKirchhoffICNN, self).__init__()
            self.ls = [layers.Dense(units, activation="softplus")]
            for l in range(nlayers - 1):
                self.ls += [layers.Dense(units, activation="softplus",
                            kernel_constraint=non_neg())]
            self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
            
        def call(self, F):
            P, W = PiolaKirchhoff()(F, lambda F: self.ls(Invariants()(F)))
            return P, W
    
if __name__ == "__main__":
    import os
    from data import load_data, load_invariants, plot_load_path
    from data import loc_uniaxial, loc_pure_shear, loc_biaxial
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # setup
    data_file, invariants_file = loc_biaxial
    
    # load data
    F_data, C_data, P_data, W_data = load_data(data_file)
    I_data = load_invariants(invariants_file)
    
    kwargs = {"nlayers": 3, "units": 16}

    model = PiolaKirchhoffFFNN(**kwargs)
    model.compile("adam", "mse")

    epochs = 500

    tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
    h = model.fit(C_data, P_data, epochs=epochs, verbose=2)
    
    plot_load_path(C_data, model.predict(C_data))
    plot_load_path(C_data, P_data)