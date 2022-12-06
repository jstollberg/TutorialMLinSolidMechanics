"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I
Task 3: Hyperelasticity II

==================

Authors: Henrik Hembrock, Jonathan Stollberg

11/2022
"""
import os
import tensorflow as tf
from data import load_data
from data import loc_uniaxial, loc_pure_shear, loc_biaxial
from data import loc_biaxial_test, loc_mixed_test
from data import (loc_bcc_uniaxial, loc_bcc_biaxial, loc_bcc_planar, 
                  loc_bcc_shear, loc_bcc_volumetric)
from models import ModelMS, ModelWI, ModelWF
from models import InvariantsTransIso, InvariantsCubic
from utils import weight_L2, voigt_to_tensor, tensor_to_voigt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# load calibration data (task 2)
F_biaxial, C_biaxial, P_biaxial, W_biaxial = load_data(loc_biaxial[0])
F_uniaxial, C_uniaxial, P_uniaxial, W_uniaxial = load_data(loc_uniaxial[0])
F_shear, C_shear, P_shear, W_shear = load_data(loc_pure_shear[0])

# load test data (task 2)
(F_biaxial_test, C_biaxial_test, 
 P_biaxial_test, W_biaxial_test) = load_data(loc_biaxial_test[0])
(F_mixed_test, C_mixed_test, 
 P_mixed_test, W_mixed_test) = load_data(loc_mixed_test[0])

# loadd bcc data (task 3)
(F_bcc_uniaxial, C_bcc_uniaxial, 
 P_bcc_uniaxial, W_bcc_uniaxial) = load_data(loc_bcc_uniaxial)
(F_bcc_biaxial, C_bcc_biaxial,
 P_bcc_biaxial, W_bcc_biaxial) = load_data(loc_bcc_biaxial)
(F_bcc_planar, C_bcc_planar,
 P_bcc_planar, W_bcc_planar) = load_data(loc_bcc_planar)
(F_bcc_shear, C_bcc_shear,
 P_bcc_shear, W_bcc_shear) = load_data(loc_bcc_shear)
(F_bcc_volumetric, C_bcc_volumetric,
 P_bcc_volumentric, W_bcc_volumentric) = load_data(loc_bcc_volumetric)

#%% setup
NN_type = "WF"  # options: MS, WI, WF
invariants = InvariantsTransIso()
sample_weight = weight_L2(P_uniaxial, P_biaxial, P_shear)
loss_weights = None
kwargs = {"nlayers": 3, "units": 16}
epochs = 100

test_data = F_mixed_test

if NN_type == "MS":
    training_in = tf.concat([C_biaxial, C_uniaxial, C_shear], axis=0)
    training_out = tf.concat([P_biaxial, P_uniaxial, P_shear], axis=0)
    
elif NN_type == "WF":
    training_in = tf.concat([F_biaxial, F_uniaxial, F_shear], axis=0)
    training_out = [tf.concat([P_biaxial, P_uniaxial, P_shear], axis=0),
                    tf.concat([W_biaxial, W_uniaxial, W_shear], axis=0)]
    
elif NN_type == "WF":
    assert False
    
else:
    raise RuntimeError("Chosen neural network not imlemented.")
   
#%% training
if NN_type == "MS":
    model = ModelMS(**kwargs)
elif NN_type == "WI":
    model = ModelWI(invariants, **kwargs)
elif NN_type == "WF":
    model = ModelWF(**kwargs)
model.compile("adam", "mse", loss_weights=loss_weights)

# fit to data
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit(training_in, 
              training_out, 
              epochs=epochs, 
              sample_weight=sample_weight,
              verbose=2)

#%% prediction

if NN_type == "MS":
    P = model.predict(test_data)
elif NN_type == "WI":
    P, W = model.predict(test_data)
elif NN_type == "WF":
    P, W = model.predict(test_data)
    
#%% plot results