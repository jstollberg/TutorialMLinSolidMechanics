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
from tensorflow.linalg import inv, det

def deviator(field):
    """
    Compute the deviators for a batch of 3x3 tensors.

    Parameters
    ----------
    field : tensorflow.Tensor
        The batch of 3x3 tensors.

    Returns
    -------
    deviator : tensorflow.Tensor
        The batch of deviators.

    """
    # define second order identity tensor
    n = len(field)
    I = tf.eye(3, batch_shape=[n], dtype=field.dtype)
    
    # c0mpute deviator
    trace = tf.linalg.trace(field)
    trace = tf.reshape(trace, (len(field),1,1))
    deviator = field - 1/3*trace*I
    
    return deviator

def equivalent(field, field_type):
    """
    Compute scalar equivalent quantities for a batch of 3x3 input tensors.

    Parameters
    ----------
    field : tensorflow.Tensor
        The batch of 3x3 tensors.
    field_type : str
        Identifier for the type of field, i.e. `"stress"` or `"strain"`.

    Returns
    -------
    eq : tensorflow.Tensor
        The batch of equivalent quantities.

    """
    # get deviator
    dev = deviator(field)

    # compute equivalent quantity
    if field_type == "strain":
        k = 2/3
    elif field_type == "stress":
        k = 3/2
    else:
        raise ValueError("No valid field type.")
        
    eq = tf.math.sqrt(k*tf.einsum("kij,kij->k", dev, dev))
    eq = tf.reshape(eq, (len(field),1))

    return eq

def weight_L2(*tensors):
    """
    Compute weights based on the L2 norm for the loss weight strategy.

    Parameters
    ----------
    *tensors : tensorflow.Tensor
        The tensors to weight.

    Returns
    -------
    weights : numpy.ndarray
        The weights in the same order as the input tensors.

    """
    weights = np.array([])
    for T in tensors:
        if len(T.shape) == 2:
            T = tf.reshape(T, (-1,3,3))
        
        norm = tf.norm(T, ord="fro", axis=[-2,-1])
        w = tf.reduce_sum(norm)
        w = np.repeat(1/w.numpy(), len(T))
        weights = np.concatenate((weights, w))
        
    return weights
    
def tensor_to_voigt(tensor):
    """
    Convert a batch of (nonsymmetric) 3x3 tensors to Voigt notation.

    Parameters
    ----------
    tensor : tensorflow.Tensor
        The batch of tensors in 3x3 tensor notation.

    Returns
    -------
    tensorflow.Tensor
        The batch of input tensors in Voigt notation.

    """
    return tf.reshape(tensor, (-1,9))

def voigt_to_tensor(tensor):
    """
    Convert a batch of (nonsymmetric) tensors from Voigt notation to 3x3 
    tensor notation.
    
    The output order will be (11, 12, 13, 21, 22, 23, 31, 32, 33).

    Parameters
    ----------
    tensor : tensorflow.Tensor
        The batch of tensors in Voigt notation.

    Returns
    -------
    tensorflow.Tensor
        The batch of tensors in 3x3 tensor notation.

    """
    return tf.reshape(tensor, (-1,3,3))

def symmetric(tensor):
    """
    Reduce symmetric tensors in 9 component Voigt notation to 6 independent
    components.
    
    The output order will be (11, 22, 33, 23, 13, 12).

    Parameters
    ----------
    tensor : tensorflow.Tensor
        The batch of symmetric tensors in 9 component Voigt notation.

    Returns
    -------
    tensorflow.Tensor
        The batch of symmetric tensors in 6 component Voigt notation.

    """
    tensor = tensor.numpy()[:,[0,4,8,5,2,1]]
    return tf.convert_to_tensor(tensor)

def cofactor(tensor):
    """
    Compute the cofactor of a batch of 3x3 tensors.

    Parameters
    ----------
    tensor : tensorflow.Tensor
        The tensors.

    Returns
    -------
    tensorflow.Tensor
        The cofactor of each tensor.

    """
    return tf.reshape(det(tensor), (len(tensor),1,1))*inv(tensor)