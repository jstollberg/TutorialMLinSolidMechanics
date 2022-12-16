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
from models import ModelWI
from models import InvariantsTransIso
from utils import weight_L2, tensor_to_voigt
from plots import plot_stress_strain, plot_stress_stress, plot_energy

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
training_in = [F_biaxial, F_uniaxial, F_shear]
training_out = [[P_biaxial, P_uniaxial, P_shear],
                [W_biaxial, W_uniaxial, W_shear]]

training_in = [F_uniaxial]
training_out = [[P_uniaxial], [W_uniaxial]]

sample_weights = weight_L2(*training_out[0])
kwargs = {"nlayers": 3, "units": 16}
epochs = 10000

#%% training
training_in = tf.concat(training_in, axis=0)
training_out = [tf.concat(training_out[0], axis=0),
                tf.concat(training_out[1], axis=0)]

models = []
h = []
for loss_weights in [[1,0],[0,1],[1,1]]:
    model = ModelWI(InvariantsTransIso(), **kwargs)
    model.compile("adam", "mse", loss_weights=loss_weights)
    tf.keras.backend.set_value(model.optimizer.learning_rate, 0.002)
    h.append(model.fit(training_in, 
                       training_out, 
                       epochs=epochs, 
                       sample_weight=sample_weights,
                       verbose=2))
    models.append(model)

#%% testing
test_data = (F_mixed_test, P_mixed_test, W_mixed_test)
# test_data = (F_biaxial_test, P_biaxial_test, W_biaxial_test)
# test_data = (F_uniaxial, P_uniaxial, W_uniaxial)
# test_data = (F_biaxial, P_biaxial, W_biaxial)
# test_data = (F_shear, P_shear, W_shear)

P = []
W = []
for m in models:
    Pm, Wm = m.predict(test_data[0])
    P.append(Pm)
    W.append(Wm)

#%% plot deformation gradient vs. stress 

# normal components
components = [0,4,8]
fig1, ax1 = plt.subplots(dpi=600)
for Pi in P:
    plot_stress_strain(ax1, test_data[0], Pi, 0, components, test_data[1])

lines = ax1.get_lines()
lines[6].set_linestyle("--")
lines[8].set_linestyle("--")
lines[10].set_linestyle("--")
lines[12].set_linestyle(":")
lines[14].set_linestyle(":")
lines[16].set_linestyle(":")

lines[0].set_label("$P_{11}$ ($\mathbf{P}$)")
lines[2].set_label("$P_{22}$ ($\mathbf{P}$)")
lines[4].set_label("$P_{33}$ ($\mathbf{P}$)")
lines[6].set_label("$P_{11}$ ($W$)")
lines[8].set_label("$P_{22}$ ($W$)")
lines[10].set_label("$P_{33}$ ($W$)")
lines[12].set_label("$P_{11}$ (both)")
lines[14].set_label("$P_{22}$ (both)")
lines[16].set_label("$P_{33}$ (both)")

plt.legend(ncol=3, handlelength=1.2, columnspacing=0.7, loc="upper left")
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$P_{ij}$")
fig1.tight_layout(pad=0.2)
# plt.savefig("./WI_transiso_PvsF_normal.pdf")

# shear components
components = [1,3]
fig2, ax2 = plt.subplots(dpi=600)
for Pi in P:
    plot_stress_strain(ax2, test_data[0], Pi, 1, components, test_data[1])

lines = ax2.get_lines()
lines[4].set_linestyle("--")
lines[6].set_linestyle("--")
lines[8].set_linestyle(":")
lines[10].set_linestyle(":")

lines[0].set_label("$P_{12}$ ($\mathbf{P}$)")
lines[2].set_label("$P_{21}$ ($\mathbf{P}$)")
lines[4].set_label("$P_{12}$ ($W$)")
lines[6].set_label("$P_{21}$ ($W$)")
lines[8].set_label("$P_{12}$ (both)")
lines[10].set_label("$P_{21}$ (both)")

plt.legend(ncol=3, handlelength=1.2, columnspacing=0.7, loc="upper left")
# plt.grid()
plt.xlabel("$F_{12}$")
plt.ylabel("$P_{ij}$")
fig2.tight_layout(pad=0.2)
# plt.savefig("./WI_transiso_PvsF_shear.pdf")

#%% plot calibration stress vs. model stress 

# normal components
components = [0,4,8]
fig3, ax3 = plt.subplots(dpi=600)
for Pi in P:
    plot_stress_stress(ax3, test_data[1], Pi, components)
    
lines = ax3.get_lines()

lines[4].set_linestyle("--")
lines[5].set_linestyle("--")
lines[6].set_linestyle("--")
lines[8].set_linestyle(":")
lines[9].set_linestyle(":")
lines[10].set_linestyle(":")

lines[0].set_label("$P_{11}$ ($\mathbf{P}$)")
lines[1].set_label("$P_{22}$ ($\mathbf{P}$)")
lines[2].set_label("$P_{33}$ ($\mathbf{P}$)")
lines[4].set_label("$P_{11}$ ($W$)")
lines[5].set_label("$P_{22}$ ($W$)")
lines[6].set_label("$P_{33}$ ($W$)")
lines[8].set_label("$P_{11}$ (both)")
lines[9].set_label("$P_{22}$ (both)")
lines[10].set_label("$P_{33}$ (both)")
    
plt.legend(ncol=3, handlelength=1.2, columnspacing=0.7)
# plt.grid()
plt.xlabel("normed $P_{ij}$ (calibration data)")
plt.ylabel("normed $P_{ij}$ (model)")
fig3.tight_layout(pad=0.2)
# plt.savefig("./WI_transiso_PvsP_normal.pdf")

# shear components
components = [1,3]
fig4, ax4 = plt.subplots(dpi=600)
for Pi in P:
    plot_stress_stress(ax4, test_data[1], Pi, components)
    
lines = ax4.get_lines()
lines[3].set_linestyle("--")
lines[4].set_linestyle("--")
lines[6].set_linestyle(":")
lines[7].set_linestyle(":")

lines[0].set_label("$P_{12}$ ($\mathbf{P}$)")
lines[1].set_label("$P_{21}$ ($\mathbf{P}$)")
lines[3].set_label("$P_{12}$ ($W$)")
lines[4].set_label("$P_{21}$ ($W$)")
lines[6].set_label("$P_{12}$ (both)")
lines[7].set_label("$P_{21}$ (both)")
    
plt.legend(ncol=3, handlelength=1.2, columnspacing=0.7)
# plt.grid()
plt.xlabel("normed $P_{ij}$ (calibration data)")
plt.ylabel("normed $P_{ij}$ (model)")
fig4.tight_layout(pad=0.2)
# plt.savefig("./WI_transiso_PvsP_shear.pdf")

#%% plot energy
fig5, ax5 = plt.subplots(dpi=600)
plot_energy(ax5, test_data[0], W[1], 0, test_data[2])
plot_energy(ax5, test_data[0], W[2], 0, test_data[2])

lines = ax5.get_lines()
lines[0].set_linestyle("--")
lines[2].set_linestyle(":")

lines[0].set_label("$W$ ($W$)")
lines[2].set_label("$W$ (both)")

plt.legend(ncol=3, handlelength=1.2, columnspacing=0.7)
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$W$")
fig5.tight_layout(pad=0.2)
# plt.savefig("./WI_transiso_WvsF_1.pdf")

fig6, ax6 = plt.subplots(dpi=600)
plot_energy(ax6, test_data[0], W[0], 0, test_data[2])

lines = ax6.get_lines()
lines[0].set_label("$W$ ($\mathbf{P}$)")

plt.legend(ncol=3, handlelength=1.2, columnspacing=0.7)
# plt.grid()
plt.xlabel("$F_{11}$")
plt.ylabel("$W$")
fig6.tight_layout(pad=0.2)
# plt.savefig("./WI_transiso_WvsF_2.pdf")

#%% plot training_loss
fig7, ax7 = plt.subplots(dpi=600)
ax7.semilogy(h[0].history["loss"], label="training loss", color="black")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
fig7.tight_layout(pad=0.2)
# plt.savefig("WI_transiso_loss_P.pdf")

fig8, ax8 = plt.subplots(dpi=600)
ax8.semilogy(h[1].history["loss"], label="training loss", color="black")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
fig8.tight_layout(pad=0.2)
# plt.savefig("WI_transiso_loss_W.pdf")

fig9, ax9 = plt.subplots(dpi=600)
ax9.semilogy(h[2].history["loss"], label="training loss", color="black")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
fig9.tight_layout(pad=0.2)
# plt.savefig("WI_transiso_loss_both.pdf")

#%% check undeformed state
F = tf.eye(3,3,batch_shape=(1,))
F = tensor_to_voigt(F)
for i, m in enumerate(models):
    Pm, Wm = m.predict(F)
    print(f"i = {i}")
    print("P = ")
    print(Pm)
    print("W = ")
    print(Wm)
    print("")
    

