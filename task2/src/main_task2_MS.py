"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I
Task 3: Hyperelasticity II

==================

Authors: Henrik Hembrock, Jonathan Stollberg

12/2022
"""
import os
import tensorflow as tf
from data import load_data
from data import loc_uniaxial, loc_pure_shear, loc_biaxial
from data import loc_biaxial_test, loc_mixed_test
from models import ModelMS
from utils import weight_L2
from plots import plot_stress_strain, plot_stress_stress

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                      "font.size": 14
                      })

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# load calibration data
F_biaxial, C_biaxial, P_biaxial, W_biaxial = load_data(loc_biaxial[0])
F_uniaxial, C_uniaxial, P_uniaxial, W_uniaxial = load_data(loc_uniaxial[0])
F_shear, C_shear, P_shear, W_shear = load_data(loc_pure_shear[0])

# load test data
(F_biaxial_test, C_biaxial_test, 
 P_biaxial_test, W_biaxial_test) = load_data(loc_biaxial_test[0])
(F_mixed_test, C_mixed_test, 
 P_mixed_test, W_mixed_test) = load_data(loc_mixed_test[0])

#%% setup
training_in = [C_biaxial, C_uniaxial, C_shear]
training_out = [P_biaxial, P_uniaxial, P_shear]
sample_weights = weight_L2(*training_out)
kwargs = {"nlayers": 3, "units": 16}
epochs = 10000

#%% training
training_in = tf.concat(training_in, axis=0)
training_out = tf.concat(training_out, axis=0)

model = ModelMS(**kwargs)
model.compile("adam", "mse")
tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
h = model.fit(training_in, 
              training_out, 
              epochs=epochs, 
              sample_weight=sample_weights,
              verbose=2)

#%% testing
test_data = (C_mixed_test, P_mixed_test)
# test_data = (C_biaxial_test, P_biaxial_test)
# test_data = (C_uniaxial, P_uniaxial)
# test_data = (C_biaxial, P_biaxial)
# test_data = (C_shear, P_shear)

P = model.predict(test_data[0])

#%% plot results

# plot strain vs. stress
components = range(9)
fig1, ax1 = plt.subplots(dpi=600)
handles = plot_stress_strain(ax1, test_data[0], P, 0, components, test_data[1])
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$P_{ij}$")
fig1.tight_layout(pad=0.2)
# plt.savefig("./MS_transiso_PvsF.pdf")

# plot calibration stress vs. model stress
components = range(9)
fig2, ax2 = plt.subplots(dpi=600)
handles = plot_stress_stress(ax2, test_data[1], P, components)
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("normed $P_{ij}$ (calibration data)")
plt.ylabel("normed $P_{ij}$ (model)")
fig2.tight_layout(pad=0.2)
# plt.savefig("./MS_transiso_PvsP.pdf")

# plot training_loss
fig3, ax3 = plt.subplots(dpi=600)
ax3.semilogy(h.history["loss"], label="training loss", color="black")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
# plt.savefig("MS_transiso_loss.pdf")