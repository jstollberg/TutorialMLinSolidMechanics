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
import matplotlib.pyplot as plt

from utils import tensor_to_voigt, voigt_to_tensor, equivalent, weight_L2
from models import Invariants, StrainEnergy, PiolaKirchhoff
from random import randrange

# loc_base = os.path.join(".", "task2", "data")
loc_base = os.path.join(".", "..", "data")

loc_biaxial = (os.path.join(loc_base, "calibration", "biaxial.txt"),
               os.path.join(loc_base, "invariants", "I_biaxial.txt"))
loc_pure_shear = (os.path.join(loc_base, "calibration", "pure_shear.txt"),
                  os.path.join(loc_base, "invariants", "I_pure_shear.txt"))
loc_uniaxial = (os.path.join(loc_base, "calibration", "uniaxial.txt"),
                os.path.join(loc_base, "invariants", "I_uniaxial.txt"))
loc_path_data = (os.path.join(loc_base, "concentric"))

loc_biaxial_test = os.path.join(loc_base, "test", "biax_test.txt")
loc_mixed_test = os.path.join(loc_base, "test", "mixed_test.txt")

# Randomly pick N load paths from concentric folder
# F11 F12 F13 F21 F22 F23 F31 F32 F33
def load_rand_path_data(N, loc, count=100, lines=50):
    F = np.empty((N*lines, 9))

    for i in range(N):
        start, stop = i*50, i*50+50
        ind = randrange(1, 100)
        loc_n = (os.path.join(loc, str(ind)+".txt"))
        data = np.loadtxt(loc_n)
        print("read file " + str(ind) + ".txt")
        F[start:stop, :] = data
    
    return F

def load_data(file, voigt=True):
    data = np.loadtxt(file)
    
    # convert numpy array to tensorflow tensor
    F = tf.convert_to_tensor(data[:,[0,1,2,3,4,5,6,7,8]])
    P = tf.convert_to_tensor(data[:,[9,10,11,12,13,14,15,16,17]])
    W = tf.reshape(tf.convert_to_tensor(data[:,18]), (-1,1))
    
    # convert to tensor notation
    F = voigt_to_tensor(F)
    P = voigt_to_tensor(P)
    C = tf.linalg.matrix_transpose(F)*F
    
    # convert to voigt notation if requested
    if voigt:
        F = tensor_to_voigt(F)
        C = tensor_to_voigt(C)
        P = tensor_to_voigt(P)

    return F, C, P, W

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

    plt.show()
    
def plot_equivalent(F, P):
    # compute equivalent quantities
    F_eq = equivalent(F, "strain")
    P_eq = equivalent(P, "stress")

    fig, ax = plt.subplots(dpi=600)
    ax.plot(F_eq, P_eq)
    ax.set(xlabel="equivalent deformation gradient",
           ylabel="equivalent first Piola-Kirchhoff stress")
    ax.grid()
    plt.show()

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # setup
    data_file, invariants_file = loc_biaxial
    
    # load data
    F_data, C_data, P_data, W_data = load_data(data_file, voigt=True)
    I_data = load_invariants(invariants_file)
    path_data = load_rand_path_data(3, loc_path_data)
    
    # evaluate invariants, energy and stress
    I = Invariants()(F_data)
    P, W = PiolaKirchhoff()(F_data, StrainEnergy())
    
    # check if the implementation is valid
    assert np.allclose(I.numpy(), I_data.numpy(), rtol=1e-3, atol=1e-3)
    assert np.allclose(W.numpy(), W_data.numpy(), rtol=1e-3, atol=1e-3)
    assert np.allclose(P.numpy(), P_data.numpy(), rtol=1e-3, atol=1e-3)
    
    # plot load path
    # plot_load_path(F_data, P)
    # plot_equivalent(F_data, P)
