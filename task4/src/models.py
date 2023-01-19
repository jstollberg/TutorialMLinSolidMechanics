"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Henrik Hembrock, Jonathan Stollberg, Dominik K. Klein
         
01/2023
"""
import tensorflow as tf
from tensorflow.keras import layers

class MaxwellModel(tf.keras.Model):
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
    
class RNNCell(layers.AbstractRNNCell):
    def __init__(self, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.ls = [layers.Dense(32, 'softplus')]
        self.ls += [layers.Dense(2)]
        
    @property
    def state_size(self):
        return [[1]]

    @property
    def output_size(self):
        return [[1]]
        
    def call(self, inputs, states):
        # n: current time step, N: old time step
        eps_n = inputs[0]
        hs = inputs[1]
        gamma_N = states[0]  # internal variable
        x = tf.concat([eps_n, hs, gamma_N], axis = 1)
                
        for l in self.ls:
            x = l(x)
        sig_n = x[:,0:1]
        gamma_n = x[:,1:2]
            
        return sig_n , [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # define initial values of the internal variables as zero
        return [tf.zeros([batch_size, 1])]

def build_naive_RNN(**kwargs):
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')
        
    cell = RNNCell(**kwargs)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1((eps, hs))

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile('adam', 'mse')
    
    return model
            
class MaxwellModelCell(layers.AbstractRNNCell):
    # TODO: is bias_constraint = non_neg() also okay instead of no bias?
    def __init__(self, nlayers=3, units=8, activation="softplus"):
        super(MaxwellModelCell, self).__init__()
        self.ls = [layers.Dense(units, 
                                activation=activation,
                                # use_bias=False,
                                input_shape=(1,1,1))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, 
                                     activation=activation,
                                     # use_bias=False
                                     )]
        self.ls += [layers.Dense(2, activation=activation, use_bias=False)]
        
    @property
    def state_size(self):
        return [[1]]
    
    @property
    def output_size(self):
        return [[1]]
    
    def call(self, inputs, states):
        # n: current time step, N: old time step
        eps_n = inputs[0]
        hs = inputs[1]
        gamma_N = states[0]  # internal variable
        x = tf.concat([eps_n, hs, gamma_N], axis=1)
             
        # evaluate layers
        for l in self.ls:
            x = l(x)
        sig_n = x[:,0:1]
        gamma_n = x[:,1:2]
            
        return sig_n, [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # define initial values of internal variables as zero
        return [tf.zeros([batch_size, 1])]
    
def build_maxwell_RNN(**kwargs):
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')
        
    cell = MaxwellModelCell(**kwargs)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1((eps, hs))

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile('adam', 'mse')
    
    return model

class GSMCell(layers.AbstractRNNCell):
    # TODO: is bias_constraint = non_neg() also okay instead of no bias?
    def __init__(self, nlayers=3, units=8, activation="softplus"):
        super(GSMCell, self).__init__()
        self.ls = [layers.Dense(units, 
                                activation=activation,
                                input_shape=(1,1,1))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, 
                                     activation=activation,
                                     )]
        self.ls += [layers.Dense(2, activation=activation)]
        
    @property
    def state_size(self):
        return [[1]]
    
    @property
    def output_size(self):
        return [[1]]
    
    def call(self, inputs, states):
        # n: current time step, N: old time step
        eps_n = inputs[0]
        hs = inputs[1]
        gamma_N = states[0]  # internal variable
             
        # evaluate layers
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                tape1.watch(gamma_N)
                tape2.watch(eps_n)
                x = tf.concat([eps_n, hs, gamma_N], axis=1)
                for l in self.ls:
                    x = l(x)
                
        sig_n = tape2.gradient(x, eps_n)
        gamma_n = -1*tape1.gradient(x, gamma_N)

        return sig_n, [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # define initial values of internal variables as zero
        return [tf.zeros([batch_size, 1])]
    
def build_GSM_RNN(**kwargs):
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')
        
    cell = GSMCell(**kwargs)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1((eps, hs))

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile('adam', 'mse')
    
    return model