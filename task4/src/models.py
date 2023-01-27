"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Henrik Hembrock, Jonathan Stollberg, Dominik K. Klein
         
01/2023
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import non_neg

class MaxwellModel(tf.keras.Model):
    """
    Standard implementation of the 1d analytical Maxwell model.
    
    Parameters
    ----------
    E_infty : float
        Young's modulus of the equilibrium spring.
    E : float
        Young's modulus of the non-equilibrium spring.
    eta : float
        Viscosity of the damper.
    **kwargs
        
    Attributes
    ----------
    E_infty : float
        Young's modulus of the equilibrium spring.
    E : float
        Young's modulus of the non-equilibrium spring.
    eta : float
        Viscosity of the damper.
        
    """
    def __init__(self, E_infty, E, eta, **kwargs):
        super(MaxwellModel, self).__init__(**kwargs)
        self.E_infty = E_infty
        self.E = E
        self.eta = eta
    
    def call(self, inp):
        """
        Evaluate the Maxwell model.

        Parameters
        ----------
        inp : list
            Input list with strains at position 0 and time steps at position 1.

        Returns
        -------
        stresses : list
            The stresses corresponding to the input strain states.

        """
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
    """
    Cell for a naive RNN model for viscoelasticity.
    
    Parameters
    ----------
    nlayers : int, optional
        Number of hidden layers. Default is 3.
    units : int, optional
        Number of nodes per hidden layer. Default is 8.
    activation : str, optional
        Activation function to use. Default "softplus".
        
    Attributes
    ----------
    ls : list
        Layers of the cell.
    
    """
    def __init__(self, nlayers=3, units=8, activation="softplus", **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.ls = [layers.Dense(units, 
                                activation=activation,
                                input_shape=(1,1,1))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, 
                                     activation=activation)]
        self.ls += [layers.Dense(2)]
        
    @property
    def state_size(self):
        return [[1]]

    @property
    def output_size(self):
        return [[1]]
        
    def call(self, inputs, states):
        """
        Evaluate the cell.

        Parameters
        ----------
        inputs : list
            List of inputs with strains at position 0 and time steps at
            position 1.
        states : list
            List of states with internal variable at position 0.
        
        Returns
        -------
        sig_n : list
            Stress states corresponding to the strain states.
        list
            List of state variables that will be used again as model input.

        """
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
        """Initialize all state variables as zero."""
        return [tf.zeros([batch_size, 1])]

def build_naive_RNN(**kwargs):
    """
    Build the naive RNN for viscoelasticity.

    Parameters
    ----------
    **kwargs

    Returns
    -------
    model : tensorflow.keras.Model
        The RNN model.

    """
    eps = tf.keras.Input(shape=[None, 1], name="input_eps")
    hs = tf.keras.Input(shape=[None, 1], name="input_hs")
        
    cell = RNNCell(**kwargs)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1((eps, hs))

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile("adam", "mse")
    
    return model
            
class MaxwellModelCell(layers.AbstractRNNCell):
    """
    Cell for a the evolution function of the Maxwell model.
    
    Parameters
    ----------
    E_infty : float
        Young's modulus of the equilibrium spring.
    E : float
        Young's modulus of the non-equilibrium spring.
    nlayers : int, optional
        Number of hidden layers. Default is 3.
    units : int, optional
        Number of nodes per hidden layer. Default is 8.
    activation : str, optional
        Activation function to use. Default is "softplus".
        
    Attributes
    ----------
    E_infty : float
        Young's modulus of the equilibrium spring.
    E : float
        Young's modulus of the non-equilibrium spring.
    ls : list
        Layers of the cell.
    
    """
    def __init__(self, E_infty, E, nlayers=3, units=8, activation="softplus"):
        super(MaxwellModelCell, self).__init__()
        
        # material constants
        self.E_infty = E_infty
        self.E = E
        
        # FFNN
        self.ls = [layers.Dense(units, 
                                activation=activation,
                                input_shape=(1,1))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, 
                                     activation=activation)]
        self.ls += [layers.Dense(1, activation=activation, 
                                 bias_constraint=non_neg())]
        
    @property
    def state_size(self):
        return [[1],[1]]
    
    @property
    def output_size(self):
        return [[1]]
    
    def call(self, inputs, states):
        """
        Evaluate the cell.

        Parameters
        ----------
        inputs : list
            List of inputs with strains at position 0 and time steps at
            position 1.
        states : list
            List of states with old strain at position 0 and internal 
            variable at position 1.
        
        Returns
        -------
        sig_n : list
            Stress states corresponding to the strain states.
        list
            List of state variables that will be used again as model input.

        """
        # n: current time step, N: old time step
        eps_n = inputs[0]
        hs = inputs[1]
        eps_N = states[0]
        gamma_N = states[1]  # internal variable
        x = tf.concat([eps_n, gamma_N], axis=1)
             
        # evaluate layers
        for l in self.ls:
            x = l(x)
            
        # compute stress
        gamma_n = gamma_N + hs*x*(eps_N - gamma_N)
        sig_n = self.E_infty*eps_n + self.E*(eps_n - gamma_n)
            
        return sig_n, [eps_n, gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Initialize all state variables as zero."""
        return [tf.zeros([batch_size, 1]), tf.zeros([batch_size, 1])]
    
def build_maxwell_RNN(E_infty, E, **kwargs):
    """
    Build the Maxwell RNN.

    Parameters
    ----------
    E_infty : float
        Young's modulus of the equilibrium spring.
    E : float
        Young's modulus of the non-equilibrium spring.
    **kwargs

    Returns
    -------
    model : tensorflow.keras.Model
        The Maxwell RNN model.

    """
    eps = tf.keras.Input(shape=[None, 1], name="input_eps")
    hs = tf.keras.Input(shape=[None, 1], name="input_hs")
        
    cell = MaxwellModelCell(E_infty, E, **kwargs)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1((eps, hs))

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile("adam", "mse")
    
    return model

class GSMCell(layers.AbstractRNNCell):
    """
    Cell for the GSM model.
    
    Parameters
    ----------
    eta : float
        Viscosity of the damper.
    nlayers : int, optional
        Number of hidden layers. Default is 3.
    units : int, optional
        Number of nodes per hidden layer. Default is 8.
    activation : str, optional
        Activation function to use. Default is "softplus".
        
    Attributes
    ----------
    eta : float
        Viscosity of the damper.
    ls : list
        Layers of the cell.
        
    """
    def __init__(self, eta, nlayers=3, units=8, activation="softplus"):
        super(GSMCell, self).__init__()
        
        # material constants
        self.eta = eta
        
        # FFNN
        self.ls = [layers.Dense(units, 
                                activation=activation,
                                input_shape=(1,1))]
        for l in range(nlayers - 1):
            self.ls += [layers.Dense(units, 
                                     activation=activation)]
        self.ls += [layers.Dense(1, activation=activation, 
                                 bias_constraint=non_neg())]
        
    @property
    def state_size(self):
        return [[1]]
    
    @property
    def output_size(self):
        return [[1]]
    
    def call(self, inputs, states):
        """
        Evaluate the cell.

        Parameters
        ----------
        inputs : list
            List of inputs with strains at position 0 and time steps at
            position 1.
        states : list
            List of states with internal variable at position 1.
        
        Returns
        -------
        sig_n : list
            Stress states corresponding to the strain states.
        list
            List of state variables that will be used again as model input.

        """
        # n: current time step, N: old time step
        eps_n = inputs[0]
        hs = inputs[1]
        gamma_N = states[0]  # internal variable
             
        # evaluate layers
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            tape1.watch(gamma_N)
            tape2.watch(eps_n)
            x = tf.concat([eps_n, gamma_N], axis=1)
            for l in self.ls:
                x = l(x)
                
        sig_n = tape2.gradient(x, eps_n)
        gamma_n = gamma_N + -1/self.eta*hs*tape1.gradient(x, gamma_N)

        return sig_n, [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Initialize all state variables as zero."""
        return [tf.zeros([batch_size, 1])]
    
def build_GSM_RNN(eta, **kwargs):
    """
    Build the GSM RNN model.

    Parameters
    ----------
    eta : float
        Viscosity of the damper.
    **kwargs

    Returns
    -------
    model : tensorflow.keras.Model
        The GSM RNN model.

    """
    eps = tf.keras.Input(shape=[None, 1], name="input_eps")
    hs = tf.keras.Input(shape=[None, 1], name="input_hs")
        
    cell = GSMCell(eta, **kwargs)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1((eps, hs))

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile("adam", "mse")
    
    return model