"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I
Task 3: Hyperelasticity II

==================

Authors: Henrik Hembrock, Jonathan Stollberg

12/2022
"""
import time
import os
import random
import numpy as np
import tensorflow as tf
from data import load_data, scale_data
from data import (loc_bcc_uniaxial, loc_bcc_biaxial, loc_bcc_planar, 
                  loc_bcc_shear, loc_bcc_volumetric)
from models import ModelWF
from utils import weight_L2, rotate, cubic_symmetries
from plots import plot_stress_strain, plot_energy

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
# training_in = [F_bcc_shear]
# training_out = [[P_bcc_shear], [W_bcc_shear]]
kwargs = {"nlayers": 3, "units": 16}
epochs = 2
observers = 64  # 1 means only the initial data

pre_train = False
checkpoint_path = os.path.join(".", f"checkpoints_{observers}", "cp.ckpt")
checkpoint_path_pre = os.path.join(".", "checkpoints_pre", "cp.ckpt")
# checkpoint_path = os.path.join(".", f"checkpoints_{observers}_shear", "cp.ckpt")
# checkpoint_path_pre = os.path.join(".", "checkpoints_pre_shear", "cp.ckpt")

#%% augment data
n_initial = sum(len(f) for f in training_in)*24
F = [f for f in training_in]
P = [p for p in training_out[0]]
W = [w for w in training_out[1]]

for i in range(observers - 1):
    angle = random.uniform(0, 2*np.pi)
    axis = [random.uniform(-1, 1), 
            random.uniform(-1, 1), 
            random.uniform(-1, 1)]
    
    for f, p, w in zip(training_in, *training_out):
        F.append(rotate(f, angle, axis, True))
        P.append(rotate(p, angle, axis, True))
        W.append(w)
        
aug_F = []
aug_P = []
aug_W = []
for i in range(len(F)):
    aug_F.extend(cubic_symmetries(F[i], False))
    aug_P.extend(cubic_symmetries(P[i], False))
    aug_W.extend([W[i] for j in range(24)])
        
# compute sample weight from stresses
sample_weight = weight_L2(*aug_P)
    
# concatenate all observers
training_in = tf.concat(aug_F, axis=0)
training_out[0] = tf.concat(aug_P, axis=0)
training_out[1] = tf.concat(aug_W, axis=0)

# scale training data
training_out[0], a = scale_data(training_out[0], 1.0)
training_out[1] *= a

#%% pre-train
loss_weights = [1,1]
model_WF = ModelWF(**kwargs)
model_WF.compile("adam", "mse", loss_weights=loss_weights)
tf.keras.backend.set_value(model_WF.optimizer.learning_rate, 0.002)
if pre_train:
    try:
        model_WF.load_weights(checkpoint_path_pre)
    except:
        pass
    cp_callback_pre = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_pre,
                                                         save_weights_only=True,
                                                         verbose=0)
    pre_start = time.time()
    h_pre = model_WF.fit(training_in[0:n_initial], 
                         [training_out[0][0:n_initial], 
                          training_out[1][0:n_initial]], 
                         epochs=15000, 
                         sample_weight=sample_weight[0:n_initial],
                         verbose=2,
                         callbacks=[cp_callback_pre])
    pre_time = time.time() - pre_start
else:
    try:
        model_WF.load_weights(checkpoint_path)
        print("Load weights from last checkpoint...")
    except:
        model_WF.load_weights(checkpoint_path_pre)
        print("Load weights from pre-training...")
            
#%% training
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=0)

train_start = time.time()
h = model_WF.fit(training_in[n_initial::], 
                 [training_out[0][n_initial::], training_out[1][n_initial::]], 
                 epochs=epochs, 
                 sample_weight=sample_weight[n_initial::],
                 verbose=2,
                 callbacks=[cp_callback])
train_time = time.time() - train_start

#%% testing
test_observers = 100
# test_data = (F_bcc_planar, P_bcc_planar, W_bcc_planar)
# test_data = (F_bcc_uniaxial, P_bcc_uniaxial, W_bcc_uniaxial)
test_data = (F_bcc_shear, P_bcc_shear, W_bcc_shear)
# test_data = (F_bcc_biaxial, P_bcc_biaxial, W_bcc_biaxial)
# test_data = (F_bcc_volumetric, P_bcc_volumetric, W_bcc_volumetric)

# create oberserver test data
F_test = []
P_test = []
W_test = []
for i in range(test_observers):
    angle = random.uniform(0, 2*np.pi)
    axis = [random.uniform(-1, 1), 
            random.uniform(-1, 1), 
            random.uniform(-1, 1)]
    
    QF = rotate(test_data[0], angle, axis, True)
    QP, W = model_WF.predict(QF)
    P = rotate(tf.convert_to_tensor(QP), angle, axis, True, True)
    P = P.numpy()
    
    P /= a
    W /= a
    
    F_test.append(QF)
    P_test.append(P)
    W_test.append(W)
    
# create symmetry test data  
QF = cubic_symmetries(test_data[0], False, False)
for i, qf in enumerate(QF):
    QP, W = model_WF.predict(qf)
    P = cubic_symmetries(tf.convert_to_tensor(QP), False, True)[i]
    F_test.append(qf)
    P_test.append(P)
    W_test.append(W)

# find max and min values for plots
max_P = P_test[0]
min_P = P_test[0]
for i in range(len(P_test) - 1):
    max_P = np.maximum(max_P, P_test[i+1])
    min_P = np.minimum(min_P, P_test[i+1])
    
max_W = W_test[0]
min_W = W_test[0]
for i in range(len(W_test) - 1):
    max_W = np.maximum(max_W, W_test[i+1])
    min_W = np.minimum(min_W, W_test[i+1])

#%% plot results
color_map = {0: "#ecbc00", 1: "#324379", 2: "#dedede",
              3: "#5c86c4", 4: "#f99c00", 5: "#dedede",
              6: "#dedede", 7: "#dedede", 8: "#bf3c3c"}

# plot deformation gradient vs. stress
components = range(9)
fig1, ax1 = plt.subplots(dpi=600)
handles = plot_stress_strain(ax1, test_data[0], max_P, 1, components, test_data[1])
handles = plot_stress_strain(ax1, test_data[0], min_P, 1, components, test_data[1])
for i in components:
    ax1.fill_between(test_data[0][:,1], max_P[:,i], min_P[:,i], 
                     alpha=0.2, color=color_map[i])
plt.legend(ncol=3, handles=handles, handlelength=1, columnspacing=0.7)
# plt.grid()
plt.xlabel("$F_{12}$")
plt.ylabel("$P_{ij}$")
fig1.tight_layout(pad=0.2)
# plt.savefig(f"./WF_cubic_PvsF_{observers}.pdf")

# plot energy
fig2, ax2 = plt.subplots(dpi=600)
plot_energy(ax2, test_data[0], max_W, 1, test_data[2])
plot_energy(ax2, test_data[0], min_W, 1, test_data[2])
ax2.fill_between(test_data[0][:,1], max_W.flatten(), min_W.flatten(), 
                 alpha=0.2, color="black")
# plt.grid()
plt.xlabel("$F_{12}$")
plt.ylabel("$W$")
fig2.tight_layout(pad=0.2)
# plt.savefig(f"./WF_cubic_WvsF_{observers}.pdf")

# plot training_loss
fig3, ax3 = plt.subplots(dpi=600)
ax3.semilogy(h.history["loss"], label="training loss", color="black")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
# plt.savefig(f"WF_cubic_loss_{observers}.pdf")