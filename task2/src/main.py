"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I

==================

Authors: Henrik Hembrock, Jonathan Stollberg

11/2022
"""
import os
import tensorflow as tf
from data import load_data, load_invariants, plot_load_path
from data import loc_uniaxial, loc_pure_shear, loc_biaxial
from data import loc_biaxial_test, loc_mixed_test
from models import PiolaKirchhoffFFNN, PiolaKirchhoffICNN
from utils import weight_L2, voigt_to_tensor, tensor_to_voigt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# load data
F_biaxial, C_biaxial, P_biaxial, W_biaxial = load_data(loc_biaxial[0])
F_uniaxial, C_uniaxial, P_uniaxial, W_uniaxial = load_data(loc_uniaxial[0])
F_shear, C_shear, P_shear, W_shear = load_data(loc_pure_shear[0])

(F_biaxial_test, C_biaxial_test, 
 P_biaxial_test, W_biaxial_test) = load_data(loc_biaxial_test)
(F_mixed_test, C_mixed_test, 
 P_mixed_test, W_mixed_test) = load_data(loc_mixed_test)

# compute sample weights
weight = weight_L2(P_biaxial, P_uniaxial, P_shear)

#%% setup
NN_type = "ICNN"  # FFNN or ICNN
sample_weight = weight
loss_weights = None
kwargs = {"nlayers": 3, "units": 16}
epochs = 100

interpolation_data = F_mixed_test

if NN_type == "FFNN":
    training_in = tf.concat([C_biaxial, C_uniaxial, C_shear], axis=0)
    training_out = tf.concat([P_biaxial, P_uniaxial, P_shear], axis=0)
    
elif NN_type == "ICNN":
    training_in = tf.concat([F_biaxial, F_uniaxial, F_shear], axis=0)
    training_out = [tf.concat([P_biaxial, P_uniaxial, P_shear], axis=0),
                    tf.concat([W_biaxial, W_uniaxial, W_shear], axis=0)]
    
else:
    raise RuntimeError("Chosen neural network not imlemented.")
   
#%% training
if NN_type == "FFNN":
    model = PiolaKirchhoffFFNN(**kwargs)
elif NN_type == "ICNN":
    model = PiolaKirchhoffICNN(**kwargs)
model.compile("adam", "mse", loss_weights=loss_weights)

# fit to data
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit(training_in, 
              training_out, 
              epochs=epochs, 
              sample_weight=sample_weight,
              verbose=2)

#%% interpolation

if NN_type == "FFNN":
    P = model.predict(interpolation_data)
elif NN_type == "ICNN":
    P, W = model.predict(interpolation_data)
    
#%% plot results