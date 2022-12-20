"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I
Task 3: Hyperelasticity II

==================

Authors: Henrik Hembrock, Jonathan Stollberg

12/2022
"""
import os
import random
import numpy as np
import tensorflow as tf

from utils import tensor_to_voigt, voigt_to_tensor, symmetric
from models import InvariantsTransIso, StrainEnergyTransIso
from models import PiolaKirchhoff

# depending on the system one of these two base paths should work
# loc_base = os.path.join(".", "task2_task3", "data")
loc_base = os.path.join(".", "..", "data")

loc_biaxial = (os.path.join(loc_base, "calibration", "biaxial.txt"),
               os.path.join(loc_base, "invariants", "I_biaxial.txt"))
loc_pure_shear = (os.path.join(loc_base, "calibration", "pure_shear.txt"),
                  os.path.join(loc_base, "invariants", "I_pure_shear.txt"))
loc_uniaxial = (os.path.join(loc_base, "calibration", "uniaxial.txt"),
                os.path.join(loc_base, "invariants", "I_uniaxial.txt"))

loc_biaxial_test = (os.path.join(loc_base, "test", "biax_test.txt"),
                    os.path.join(loc_base, "invariants", "I_biax_test.txt"))
loc_mixed_test = (os.path.join(loc_base, "test", "mixed_test.txt"),
                  os.path.join(loc_base, "invariants", "I_mixed_test.txt"))

loc_concentric = os.path.join(loc_base, "concentric")

loc_path_bcc = os.path.join(loc_base, "soft_beam_lattice_metamaterials")
loc_bcc_uniaxial = os.path.join(loc_path_bcc, "BCC_uniaxial.txt")
loc_bcc_biaxial = os.path.join(loc_path_bcc, "BCC_biaxial.txt")
loc_bcc_planar = os.path.join(loc_path_bcc, "BCC_planar.txt")
loc_bcc_shear = os.path.join(loc_path_bcc, "BCC_shear.txt")
loc_bcc_volumetric = os.path.join(loc_path_bcc, "BCC_volumetric.txt")

def load_random_gradient_data(folder, sample_size):
    """
    Load deformation gradient data from text files and randomly split into
    sample and test data.

    Parameters
    ----------
    folder : str
        The folder where the text files are stored.
    sample_size : int
        Number of load cases used for calibration.

    Returns
    -------
    F_samples : tensorflow.Tensor
        The calibration data.
    F_test : tensorflow.Tensor
        The test data.

    """
    # collect all data
    F = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            file = os.path.join(folder, file)
            data = np.loadtxt(file)
            
            # skip file in case it is not formatted correctly
            if data.shape[1] != 9:
                continue
            F.append(data)
    F = np.array(F)
            
    if sample_size >= len(F):
        raise RuntimeError("sample_size too large.")
        
    # take random load cases for calibration
    all_indices = np.arange(len(F))
    sample_indices = random.sample(all_indices.tolist(), sample_size)
    sample_indices = np.array(sample_indices)
    test_indices = np.setdiff1d(all_indices, sample_indices)
    F_samples = tf.convert_to_tensor(F[sample_indices])
    F_test = tf.convert_to_tensor(F[test_indices])
    
    # Reshape so we get one stack of data
    F_samples = tf.reshape(F_samples, (-1,9))
    
    return F_samples, F_test

def load_data(file, a=1.0, voigt=True):
    """
    Read in deformation gradient, stress and energy data from a text file.
    
    The text file must be formatted in order (F11, F12, F13, F21, F22, F23,
    F31, F32, F33, P11, P12, P13, P21, P22, P23, P31, P32, P33, W). The stress
    and energy data can be scaled with parameter `a`.

    Parameters
    ----------
    file : str
        The paths to the text file.
    voigt : bool, optional
        If `True`, the data will be returned in Voigt notation. The default is 
        `True`.

    Returns
    -------
    F : tensorflow.Tensor
        The batch of deformation gradients.
    C : tensorflow.Tensor
        The batch of right Cauchy-Green-strains.
    P : tensorflow.Tensor
        the batch of first Piola-Kirchhoff-stresses.
    W : tensorflow.Tensor
        the batch of strain energies.

    """
    try:
        data = np.loadtxt(file)
    
        # convert numpy array to tensorflow tensor
        F = tf.convert_to_tensor(data[:,[0,1,2,3,4,5,6,7,8]])
        P = tf.convert_to_tensor(data[:,[9,10,11,12,13,14,15,16,17]])
        W = tf.reshape(tf.convert_to_tensor(data[:,18]), (-1,1))
        
    except Exception:
        raise RuntimeError("Could not read in the data file")

    # convert to tensor notation
    F = voigt_to_tensor(F)
    P = voigt_to_tensor(P)
    C = tf.linalg.matmul(tf.linalg.matrix_transpose(F), F)
    
    # convert to voigt notation if requested
    if voigt:
        F = tensor_to_voigt(F)
        C = tensor_to_voigt(C)
        C = symmetric(C)
        P = tensor_to_voigt(P)

    return F, C, P, W

def load_invariants(file):
    """
    Read in the invariants from a text file.
    
    The text file must be formatted in order (I1, J, I4, I5).

    Parameters
    ----------
    file : str
        The path to the text file.

    Returns
    -------
    ret : tensorflow.Tensor
        The batch of invariants in order (I1, J, -J, I4, I5).

    """
    try:
        data = np.loadtxt(file)
        I1 = data[:, 0]
        J  = data[:, 1]
        I4 = data[:, 2]
        I5 = data[:, 3]
        
    except Exception:
        raise RuntimeError("Could not read in the data file.")
    
    # convert to tensor
    I1 = tf.convert_to_tensor(I1)
    J = tf.convert_to_tensor(J)
    I4 = tf.convert_to_tensor(I4)
    I5 = tf.convert_to_tensor(I5)
    
    # collect in one tensor
    ret = tf.stack([I1, J, -J, I4, I5], axis=1)

    return ret

def scale_data(data, a=1.0):
    """
    Scale the data by multiplying it with a scaling parameter.

    Parameters
    ----------
    data : tensorflow.Tensor
        The data to scale.
    a : float, optional
        The scaling parameter. If this is set to `None`, the data will be 
        scaled so that the data is between -1 and 1. The default is 1.0.

    Returns
    -------
    data : tensorflow.Tensor
        The scaled data.
    a : tensorflow.Tensor
        The scaling parameter.

    """
    if a is None:
        a = 1/tf.math.reduce_max(tf.math.abs(data))
    data *= a
    return data, a

if __name__ == "__main__":
    # check if implementation works
    data = [loc_uniaxial, 
            loc_biaxial, 
            loc_pure_shear, 
            loc_biaxial_test, 
            loc_mixed_test]
    
    for d in data:
        F_data, C_data, P_data, W_data = load_data(d[0], a=1)
        I_data = load_invariants(d[1])
        
        # evaluate invariants, energy and stress
        I = InvariantsTransIso()(F_data)
        P, W = PiolaKirchhoff()(F_data, 
                                InvariantsTransIso(),
                                StrainEnergyTransIso())
        
        assert np.allclose(I.numpy(), I_data.numpy(), rtol=1e-3, atol=1e-3), d
        assert np.allclose(W.numpy(), W_data.numpy(), rtol=1e-3, atol=1e-3), d
        assert np.allclose(P.numpy(), P_data.numpy(), rtol=1e-3, atol=1e-3), d
        
    print("OK")