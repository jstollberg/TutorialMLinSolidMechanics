"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I

==================

Authors: Henrik Hembrock, Jonathan Stollberg

11/2022
"""
import numpy as np
import tensorflow as tf

def deviator(field):
    # define second order identity tensor
    n = len(field)
    I = tf.eye(3, batch_shape=[n], dtype=field.dtype)
    
    # c0mpute deviator
    trace = tf.linalg.trace(field)
    trace = tf.reshape(trace, (len(field),1,1))
    deviator = field - 1/3*trace*I
    
    return deviator

def equivalent(field, field_type):
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
    weights = np.array([])
    for T in tensors:
        norm = tf.norm(T, ord="fro", axis=[-2,-1])
        w = tf.reduce_sum(norm)
        w = np.repeat(1/w.numpy(), len(T))
        weights = np.concatenate((weights, w))
        
    return weights
    