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

#%% Setup
training_in = [C_biaxial, C_uniaxial, C_shear]
training_out = [P_biaxial, P_uniaxial, P_shear]
sample_weights = weight_L2(*training_out)
kwargs = {"nlayers": 3, "units": 16}
epochs = 4000

#%% Training
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

#%% Testing
test_data = (C_mixed_test, P_mixed_test)
# test_data = (C_biaxial_test, P_biaxial_test)

P = model.predict(test_data[0])

#%% Plot results
color_map = {0: "#0083CC", 1: "gray"}
label_map = {0: "11", 1: "12"}

# visualize training loss
fig1, ax1 = plt.subplots(dpi=600)
ax1.semilogy(h.history["loss"], label="training loss")
plt.grid(which="both")
plt.xlabel("calibration epoch", labelpad=-0.1)
plt.ylabel(r"log$_{10}$ MSE")
plt.savefig("FFNN_MS_loss.pdf")

# visualize stress prediction
fig2, ax2 = plt.subplots(dpi=600)
components = [0,1]
for i in components:
    ax2.plot(test_data[0][:,i], P[:,i],
             label=label_map[i],
             linestyle="-",
             linewidth=2,
             color=color_map[i]
             )

for i in components:
    ax2.plot(test_data[0][:,i], test_data[1][:,i], 
             linewidth=0, 
             markevery=5, 
             markersize=2.5, 
             markerfacecolor="black",
             color="black",
             marker="o")
    
plt.xlabel(r"$C_{ij}$", labelpad=-0.1)
plt.ylabel(r"$P_{ij}$")
plt.grid()
plt.legend(ncol=3, columnspacing=0.5)
plt.savefig("FFNN_MS_stress.pdf")

# compare exact stress with interpolated stress
i = 0  # 11-component
fig3, ax3 = plt.subplots(dpi=600)
ax3.plot(test_data[1][:,i], P[:,i],
         label=label_map[i],
         linestyle="-",
         linewidth=2,
         color=color_map[i]
         )
ax3.plot(test_data[1][:,i], test_data[1][:,i], 
         linewidth=1.5, 
         color="black",
         linestyle="--"
         )
    
plt.xlabel(r"$P_{11}$ (data)", labelpad=-0.1)
plt.ylabel(r"$P_{11}$ (model)")
plt.grid()
plt.savefig("FFNN_MS_PvsP.pdf")

plt.show()

