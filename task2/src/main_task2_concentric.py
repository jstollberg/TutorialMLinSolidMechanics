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

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                      "font.size": 14
                      })

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

#%% Load concentric data
sample_size = 10
F_samples, F_test = load_random_gradient_data(loc_concentric, sample_size)

P_samples, W_samples = PiolaKirchhoff()(F_samples, 
                                        InvariantsTransIso(), 
                                        StrainEnergyTransIso())

C_samples = right_cauchy_green(F_samples)

#%% Setup
sample_weights = weight_L2(P_samples)
kwargs = {"nlayers": 3, "units": 16}
epochs = 4000

#%% Training

model_WI = ModelWI(InvariantsTransIso(), **kwargs)
model_WI.compile("adam", "mse", loss_weights=[1,0])
tf.keras.backend.set_value(model_WI.optimizer.learning_rate, 0.002)
h_WI = model_WI.fit(F_samples, 
                    P_samples, 
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

#%% Testing

# choose load path to test
all_indices = np.arange(len(F_test))
load_i = random.sample(all_indices.tolist(), 1)[0]

# define test data
P_test, W_test = PiolaKirchhoff()(F_test[load_i], 
                                  InvariantsTransIso(), 
                                  StrainEnergyTransIso())
test_data = (F_test[load_i], P_test, W_test)
C_test = right_cauchy_green(F_test[load_i])

# predict stresses
P_WI, W_WI = model_WI.predict(test_data[0])
P_MS = model_MS.predict(C_test)

#%% Plot results for FFNN
color_map = {0: "#0083CC", 1: "gray"}
label_map = {0: "11", 1: "12"}

# visualize training loss
fig1, ax1 = plt.subplots(dpi=600)
ax1.semilogy(h_MS.history["loss"], label="training loss")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel(r"log$_{10}$ MSE")
plt.savefig(f"FFNN_MS_loss_{sample_size}.pdf")

# visualize stress prediction
fig2, ax2 = plt.subplots(dpi=600)
components = [0,1]
for i in components:
    ax2.plot(C_test[:,i], P_MS[:,i],
              label=label_map[i],
              linestyle="-",
              linewidth=2,
              color=color_map[i]
              )

for i in components:
    ax2.plot(C_test[:,i], test_data[1][:,i], 
              linewidth=0, 
              markevery=5, 
              markersize=2.5, 
              markerfacecolor="black",
              color="black",
              marker="o")
    
plt.xlabel(r"$C_{ij}$", labelpad=-1)
plt.ylabel(r"$P_{ij}$", labelpad=-1)
plt.grid()
plt.legend(ncol=3, columnspacing=0.5)
plt.savefig(f"FFNN_MS_stress_{load_i}_{sample_size}.pdf")

# compare exact stress with interpolated stress
i = 1  # 12-component
fig3, ax3 = plt.subplots(dpi=600)
ax3.plot(test_data[1][:,1], P_MS[:,1],
          label=label_map[i],
          linestyle="-",
          linewidth=2,
          color=color_map[i]
          )
ax3.plot(test_data[1][:,1], test_data[1][:,1], 
          linewidth=1.5, 
          color="black",
          linestyle="--"
          )
    
plt.xlabel(r"$P_{12}$ (data)", labelpad=-1)
plt.ylabel(r"$P_{12}$ (model)", labelpad=-1)
plt.grid()
plt.savefig(f"FFNN_MS_PvsP_{load_i}__{sample_size}.pdf")

#%% Plot results for ICNN

# visualize training loss
fig4, ax4 = plt.subplots(dpi=600)
ax4.semilogy(h_WI.history["loss"], label="training loss")
plt.grid(which="both")
plt.xlabel("calibration epoch", labelpad=-1)
plt.ylabel(r"log$_{10}$ MSE", labelpad=-1)
plt.savefig(f"ICNN_WI_loss_{sample_size}.pdf")

# visualize stress prediction
fig5, ax5 = plt.subplots(dpi=600)
components = [0,1]
ax5.plot(test_data[0][:,0], P_WI[:,0],
         label=label_map[0],
         linestyle="-",
         linewidth=2,
         color=color_map[0]
         )
ax5.plot(test_data[0][:,1], P_WI[:,1],
         label=label_map[1],
         linestyle="-",
         linewidth=2,
         color=color_map[1]
         )

for i in components:
    ax5.plot(test_data[0][:,i], test_data[1][:,i], 
             linewidth=0, 
             markevery=5, 
             markersize=2.5, 
             markerfacecolor="black",
             color="black",
             marker="o")
    
plt.xlabel(r"$F_{ij}$", labelpad=-1)
plt.ylabel(r"$P_{ij}$", labelpad=-1)
plt.grid()
plt.legend(ncol=3, columnspacing=0.5)
plt.savefig(f"ICNN_WI_stress_{load_i}_{sample_size}.pdf")

# compare exact stress with interpolated stress
i = 1  # 12-component
fig6, ax6 = plt.subplots(dpi=600)
ax6.plot(test_data[1][:,i], P_WI[:,i],
         linestyle="-",
         linewidth=2,
         color=color_map[i]
         )
ax6.plot(test_data[1][:,i], test_data[1][:,i], 
         linewidth=1.5, 
         color="black",
         linestyle="--"
         )

plt.xlabel(r"$P_{12}$ (data)", labelpad=-1)
plt.ylabel(r"$P_{12}$ (model)", labelpad=-1)
plt.grid()
plt.savefig(f"ICNN_WI_PvsP_{load_i}_{sample_size}.pdf")

plt.show()