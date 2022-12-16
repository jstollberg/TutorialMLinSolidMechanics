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
from data import load_data, scale_data
from data import (loc_bcc_uniaxial, loc_bcc_biaxial, loc_bcc_planar, 
                  loc_bcc_shear, loc_bcc_volumetric)
from models import ModelWI
from models import InvariantsCubic
from utils import weight_L2
from plots import plot_stress_strain, plot_stress_stress, plot_energy

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                      "font.size": 14
                      })

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# loadd bcc data
(F_bcc_uniaxial, C_bcc_uniaxial, 
 P_bcc_uniaxial, W_bcc_uniaxial) = load_data(loc_bcc_uniaxial)
(F_bcc_biaxial, C_bcc_biaxial,
 P_bcc_biaxial, W_bcc_biaxial) = load_data(loc_bcc_biaxial)
(F_bcc_planar, C_bcc_planar,
 P_bcc_planar, W_bcc_planar) = load_data(loc_bcc_planar)
(F_bcc_shear, C_bcc_shear,
 P_bcc_shear, W_bcc_shear) = load_data(loc_bcc_shear)
(F_bcc_volumetric, C_bcc_volumetric,
 P_bcc_volumetric, W_bcc_volumetric) = load_data(loc_bcc_volumetric)

#%% setup
training_in = [F_bcc_uniaxial, F_bcc_biaxial, F_bcc_shear, F_bcc_volumetric]
training_out = [[P_bcc_uniaxial, P_bcc_biaxial, P_bcc_shear, P_bcc_volumetric],
                [W_bcc_uniaxial, W_bcc_biaxial, W_bcc_shear, W_bcc_volumetric]]
kwargs = {"nlayers": 3, "units": 16}
epochs = 10000

#%% training
training_in = tf.concat(training_in, axis=0)
training_out = [tf.concat(training_out[0], axis=0),
                tf.concat(training_out[1], axis=0)]
training_out[0], a = scale_data(training_out[0], None)
training_out[1] *= a

loss_weights = [1,1]
model_WI = ModelWI(InvariantsCubic(), **kwargs)
model_WI.compile("adam", "mse", loss_weights=loss_weights)
tf.keras.backend.set_value(model_WI.optimizer.learning_rate, 0.002)
h_WI = model_WI.fit(training_in, 
                    training_out, 
                    epochs=epochs, 
                    sample_weight=weight_L2(training_out[0]),
                    verbose=2)

# %% testing
test_data = (F_bcc_planar, P_bcc_planar, W_bcc_planar)
# test_data = (F_bcc_uniaxial, P_bcc_uniaxial, W_bcc_uniaxial)
# test_data = (F_bcc_shear, P_bcc_shear, W_bcc_shear)
# test_data = (F_bcc_biaxial, P_bcc_biaxial, W_bcc_biaxial)
# test_data = (F_bcc_volumetric, P_bcc_volumetric, W_bcc_volumetric)
P, W = model_WI.predict(test_data[0])
P /= a
W /= a

#%% plot results

# plot deformation gradient vs. stress
components = range(9)
fig1, ax1 = plt.subplots(dpi=600)
handles = plot_stress_strain(ax1, test_data[0], P, 0, components, test_data[1])
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$P_{ij}$")
fig1.tight_layout(pad=0.2)
# plt.savefig("./WI_cubic_PvsF.pdf")

# plot calibration stress vs. model stress
components = range(9)
fig2, ax2 = plt.subplots(dpi=600)
handles = plot_stress_stress(ax2, test_data[1], P, components)
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("normed $P_{ij}$ (calibration data)")
plt.ylabel("normed $P_{ij}$ (model)")
fig2.tight_layout(pad=0.2)
# plt.savefig("./WI_cubic_PvsP.pdf")

# plot energy
fig3, ax3 = plt.subplots(dpi=600)
plot_energy(ax3, test_data[0], W, 0, test_data[2])
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$W$")
fig3.tight_layout(pad=0.2)
# plt.savefig("./WI_cubic_WvsF.pdf")

# plot training_loss
fig4, ax4 = plt.subplots(dpi=600)
ax4.semilogy(h_WI.history["loss"], label="training loss", color="black")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
# plt.savefig("WI_cubic_loss.pdf")