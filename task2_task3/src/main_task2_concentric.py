"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I
Task 3: Hyperelasticity II

==================

Authors: Henrik Hembrock, Jonathan Stollberg

12/2022
"""
import os
import numpy as np
import random
import tensorflow as tf
from data import load_random_gradient_data
from data import loc_concentric
from models import ModelMS, ModelWI
from models import InvariantsTransIso
from models import PiolaKirchhoff, StrainEnergyTransIso
from utils import weight_L2, right_cauchy_green
from plots import plot_stress_strain, plot_stress_stress, plot_energy

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                      "font.size": 14
                      })

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

#%% load concentric data
sample_size = 90
F_samples, F_test = load_random_gradient_data(loc_concentric, sample_size)

P_samples, W_samples = PiolaKirchhoff()(F_samples, 
                                        InvariantsTransIso(), 
                                        StrainEnergyTransIso())

C_samples = right_cauchy_green(F_samples)

#%% setup
sample_weights = weight_L2(P_samples)
kwargs = {"nlayers": 3, "units": 16}
epochs = 10000

#%% training

model_WI = ModelWI(InvariantsTransIso(), **kwargs)
model_WI.compile("adam", "mse", loss_weights=[1,1])
tf.keras.backend.set_value(model_WI.optimizer.learning_rate, 0.002)
h_WI = model_WI.fit(F_samples, 
                    [P_samples, W_samples], 
                    epochs=epochs, 
                    sample_weight=sample_weights,
                    verbose=2)
    
model_MS = ModelMS(**kwargs)
model_MS.compile("adam", "mse")
tf.keras.backend.set_value(model_MS.optimizer.learning_rate, 0.002)
h_MS = model_MS.fit(C_samples, 
                    P_samples, 
                    epochs=epochs, 
                    sample_weight=sample_weights,
                    verbose=2)

#%% testing

# choose random load path to test
all_indices = np.arange(len(F_test))
i = random.sample(all_indices.tolist(), 1)[0]

# define test data
P_test, W_test = PiolaKirchhoff()(F_test[i], 
                                  InvariantsTransIso(), 
                                  StrainEnergyTransIso())
test_data = (F_test[i], P_test, W_test)
C_test = right_cauchy_green(F_test[i])

# predict stresses
P_WI, W_WI = model_WI.predict(test_data[0])
P_MS = model_MS.predict(C_test)

#%% plot results for FFNN

# plot strain vs. stress
components = range(9)
fig1, ax1 = plt.subplots(dpi=600)
handles = plot_stress_strain(ax1, test_data[0], P_MS, 0, components, test_data[1])
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$P_{ij}$")
fig1.tight_layout(pad=0.2)
# plt.savefig(f"./MS_concentric_PvsF_{i}_{sample_size}.pdf")

# plot calibration stress vs. model stress
components = range(9)
fig2, ax2 = plt.subplots(dpi=600)
handles = plot_stress_stress(ax2, test_data[1], P_MS, components)
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("normed $P_{ij}$ (calibration data)")
plt.ylabel("normed $P_{ij}$ (model)")
fig2.tight_layout(pad=0.2)
# plt.savefig(f"./MS_concentric_PvsP_{i}_{sample_size}.pdf")

# plot training_loss
fig3, ax3 = plt.subplots(dpi=600)
ax3.semilogy(h_MS.history["loss"], label="training loss", color="black")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
# plt.savefig(f"MS_concentric_loss_{sample_size}.pdf")

#%% plot results for ICNN

# plot deformation gradient vs. stress
components = range(9)
fig4, ax4 = plt.subplots(dpi=600)
handles = plot_stress_strain(ax4, test_data[0], P_WI, 0, components, test_data[1])
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$P_{ij}$")
fig1.tight_layout(pad=0.2)
# plt.savefig(f"./WI_concentric_PvsF_{i}_{sample_size}.pdf")

# plot calibration stress vs. model stress
components = range(9)
fig5, ax5 = plt.subplots(dpi=600)
handles = plot_stress_stress(ax5, test_data[1], P_WI, components)
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("normed $P_{ij}$ (calibration data)")
plt.ylabel("normed $P_{ij}$ (model)")
fig2.tight_layout(pad=0.2)
# plt.savefig(f"./WI_concentric_PvsP_{i}_{sample_size}.pdf")

# plot energy
fig6, ax6 = plt.subplots(dpi=600)
plot_energy(ax6, test_data[0], W_WI, 0, test_data[2])
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$W$")
fig3.tight_layout(pad=0.2)
# plt.savefig(f"./WI_concentric_WvsF_{i}_{sample_size}.pdf")

# plot training_loss
fig4, ax4 = plt.subplots(dpi=600)
ax4.semilogy(h_WI.history["loss"], label="training loss", color="black")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
# plt.savefig(f"./WI_concentric_loss_{sample_size}.pdf")