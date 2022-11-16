"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I

==================

Authors: Henrik Hembrock, Jonathan Stollberg

11/2022
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.linalg import matmul, matrix_transpose, trace, inv, det
import matplotlib.pyplot as plt

# loc_base = os.path.join(".", "task2", "data")
loc_base = os.path.join(".", "..", "data")
loc_biaxial = (os.path.join(loc_base, "calibration", "biaxial.txt"),
               os.path.join(loc_base, "invariants", "I_biaxial.txt"))
loc_pure_shear = (os.path.join(loc_base, "calibration", "pure_shear.txt"),
                  os.path.join(loc_base, "invariants", "I_pure_shear.txt"))
loc_uniaxial = (os.path.join(loc_base, "calibration", "uniaxial.txt"),
                os.path.join(loc_base, "invariants", "I_uniaxial.txt"))

def load_data(file):
    data = np.loadtxt(file)
    
    # define function to convert Voigt notation to tensor
    voigt_to_tensor = lambda A: tf.constant([[A[0], A[1], A[2]],
                                             [A[3], A[4], A[5]],
                                             [A[6], A[7], A[8]]])

    # get data count
    n = data.shape[0]
    
    # convert to tensor notation
    F = np.empty((n,3,3))
    P = np.empty((n,3,3))
    W = np.empty((n,1))
    for i, d in enumerate(data):
        F[i,:,:] = voigt_to_tensor(d[0:9])
        P[i,:,:] = voigt_to_tensor(d[9:18])
        W[i,:] = d[18]
        
    F = tf.convert_to_tensor(F)
    P = tf.convert_to_tensor(P)
    W = tf.convert_to_tensor(W)

    return F, P, W

def load_invariants(file):
    data = np.loadtxt(file)
    I1 = data[:, 0]
    J  = data[:, 1]
    I4 = data[:, 2]
    I5 = data[:, 3]
    
    # convert to tensor
    I1 = tf.convert_to_tensor(I1)
    J = tf.convert_to_tensor(J)
    I4 = tf.convert_to_tensor(I4)
    I5 = tf.convert_to_tensor(I5)
    
    # collect in one tensor
    ret = tf.stack([I1, J, -J, I4, I5], axis=1)

    return ret

class Invariants(layers.Layer):
    def __call__(self, F):
        # define transersely isotropic structural tensro
        G = tf.constant([[[4.0, 0.0, 0.0],
                          [0.0, 0.5, 0.0],
                          [0.0, 0.0, 0.5]]]*len(F), dtype=F.dtype)
        
        # compute right Cauchy-Green tensor
        C = matmul(matrix_transpose(F), F)

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

def plot_load_path(F, P):
    # plot stress and strain in normal direction
    F11, F22, F33 = F[:,0,0], F[:,1,1], F[:,2,2]
    P11, P22, P33 = P[:,0,0], P[:,1,1], P[:,2,2]

    fig1, ax1 = plt.subplots(dpi=600)
    ax1.plot(F11, P11, label="11")
    ax1.plot(F22, P22, label="22")
    ax1.plot(F33, P33, label="33")
    ax1.set(xlabel="deformation gradient",
            ylabel="first Piola-Kirchhoff stress")
    ax1.legend()
    ax1.grid()

    # plot stress and strain in shear direction
    F12, F13, F21, F23, F31, F32 = (F[:,0,1], F[:,0,2], F[:,1,0], F[:,1,2],
                                    F[:,2,0], F[:,2,1])
    P12, P13, P21, P23, P31, P32 = (P[:,0,1], P[:,0,2], P[:,1,0], P[:,1,2],
                                    P[:,2,0], P[:,2,1])

    fig2, ax2 = plt.subplots(dpi=600)
    ax2.plot(F12, P12, label="12")
    ax2.plot(F13, P13, label="13")
    ax2.plot(F21, P21, label="21")
    ax2.plot(F23, P23, label="23")
    ax2.plot(F31, P31, label="31")
    ax2.plot(F32, P32, label="32")
    ax2.set(xlabel="deformation gradient",
            ylabel="first Piola-Kirchhoff stress")
    ax2.legend()
    ax2.grid()

    plt.show

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # setup
    data_file, invariants_file = loc_biaxial
    
    # load data
    F_data, P_data, W_data = load_data(data_file)
    I_data = load_invariants(invariants_file)
    
    # evaluate invariants, energy and stress
    I = Invariants()(F_data)
    W = StrainEnergy()(I)
    P = PiolaKirchhoff()(F_data)
    
    # check if the implementation is valid
    assert np.allclose(I.numpy(), I_data.numpy(), rtol=1e-3, atol=1e-3)
    assert np.allclose(W.numpy(), W_data.numpy(), rtol=1e-3, atol=1e-3)
    assert np.allclose(P.numpy(), P_data.numpy(), rtol=1e-3, atol=1e-3)

    # plot load path
    plot_load_path(F_data, P_data)
