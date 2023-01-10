"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Henrik Hembrock, Jonathan Stollberg, Dominik K. Klein
         
01/2023
"""
import tensorflow as tf
from tensorflow.keras import layers

class MaxwellModel(layers.Layer):
    def __init__(self, E_infty, E, eta, **kwargs):
        super(MaxwellModel, self).__init__(**kwargs)
        self.E_infty = E_infty
        self.E = E
        self.eta = eta
    
    def call(self, inp):
        strains = inp[0]
        dts = inp[1]
        
        strains = tf.stack(strains, axis=1)
        dts = tf.stack(dts, axis=1)
        gammas = tf.Variable(tf.zeros_like(strains, dtype=strains.dtype))
        stresses = tf.Variable(tf.zeros_like(strains, dtype=strains.dtype))
        
        for i, eps in enumerate(strains[1::]):
            eps_n = strains[i]
            dt_n = dts[i]
            gamma_n = gammas[i]
            
            gamma = gamma_n + dt_n*(self.E/self.eta)*(eps_n - gamma_n)
            sigma = self.E_infty*eps + self.E*(eps - gamma)
            
            gammas[i + 1].assign(gamma)
            stresses[i + 1].assign(sigma)
            
        stresses = tf.stack(stresses, axis=1)
        gammas = tf.stack(gammas, axis=1)
        
        return stresses
            
# class MaxwellModelFFNN(tf.keras.Models):
#     def __init__(self, E_infty, E, eta, nlayers=3, units=8):
#         super(MaxwellModelFFNN, self).__init__()
#         self.ls = [layers.Dense(units, activation="softplus", 
#                                 input_shape=(1,1))]
#         for l in range(nlayers - 1):
#             self.ls += [layers.Dense(units, activation="softplus")]
#         self.ls += [layers.Dense(1)]
    
#     def call(self, inp):
#         pass