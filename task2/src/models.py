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
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import trace, inv, det
from tensorflow.keras.constraints import non_neg

class Invariants(layers.Layer):
    def __call__(self, F):
        # define transersely isotropic structural tensro
        G = np.array([[4.0, 0.0, 0.0],
                      [0.0, 0.5, 0.0],
                      [0.0, 0.0, 0.5]])
        G = tf.convert_to_tensor(G, dtype=F.dtype)
        G = tf.tile(G, [len(F),1])
        G = tf.reshape(G, (len(F),3,3))
        
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
    def __call__(self, F):
        with tf.GradientTape() as tape:
            tape.watch(F)
            I = Invariants()(F)
            W = StrainEnergy()(I)
        P = tape.gradient(W, F)
        
        return P
    
class PiolaKirchhoffFFNN(tf.keras.Model):
    def __init__(self,
                 nlayers=3,
                 units=8):
        super(PiolaKirchhoffFFNN, self).__init__()
        self.ls = [layers.Dense(units, activation="softplus", 
                                input_shape=(3,3))]
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
            self.ls = [layers.Dense(units, activation="softplus",
                                    kernel_constraint=non_neg(), 
                                    input_shape=(5,))]
            for l in range(nlayers - 1):
                self.ls += [layers.Dense(units, activation="softplus",
                                         kernel_constraint=non_neg())]
            self.ls += [layers.Dense(1, kernel_constraint=non_neg())]
            
        def call(self, F):
            with tf.GradientTape() as tape:
                tape.watch(F)
                I = Invariants()(F)
                for l in self.ls:
                    I = l(I)
            P = tape.gradient(I, F)
            return P, I
    
if __name__ == "__main__":
    import os
    from data import load_data, load_invariants, plot_load_path
    from data import loc_uniaxial, loc_pure_shear, loc_biaxial
    from utils import weight_L2
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    
    # load data
    F_biaxial, C_biaxial, P_biaxial, W_biaxial = load_data(loc_biaxial[0])
    F_uniaxial, C_uniaxial, P_uniaxial, W_uniaxial = load_data(loc_uniaxial[0])
    F_shear, C_shear, P_shear, W_shear = load_data(loc_pure_shear[0])

    # compute sample weights
    weight = weight_L2(P_biaxial, P_uniaxial, P_shear)
    
    # setup of the neural network
    training_in = tf.concat([F_biaxial, F_uniaxial, F_shear], axis=0)
    training_out = [tf.concat([P_biaxial, P_uniaxial, P_shear], axis=0),
                    tf.concat([W_biaxial, W_uniaxial, W_shear], axis=0)]
    # training_in = tf.concat([C_biaxial, C_uniaxial, C_shear], axis=0)
    # training_out = tf.concat([P_biaxial, P_uniaxial, P_shear], axis=0)
    sample_weight = weight
    loss_weights = None
    kwargs = {"nlayers": 3, "units": 16}
    
    # compile FFNN
    # model = PiolaKirchhoffFFNN(**kwargs)
    model = PiolaKirchhoffICNN(**kwargs)
    model.compile("adam", "mse", loss_weights=loss_weights)
    
    # fit to data
    epochs = 1000
    tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
    h = model.fit(training_in, 
                  training_out, 
                  epochs=epochs, 
                  sample_weight=sample_weight,
                  verbose=2)
    
    # interpolate data
    F_model = F_biaxial
    P_model, W_model = model.predict(F_model)
    # P_model = model.predict(F_model)
    plot_load_path(F_model, P_model)
    # plot_load_path(C_biaxial, P_biaxial)