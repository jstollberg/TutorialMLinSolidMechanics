"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Henrik Hembrock, Jonathan Stollberg, Dominik K. Klein
         
01/2023
"""
import tensorflow as tf
from matplotlib import pyplot as plt
import data as ld
import plots as lp
from models import MaxwellModel, build_naive_RNN, build_maxwell_RNN, build_GSM_RNN
import datetime
now = datetime.datetime.now

#%% data visualization

# material constants
E_infty = 0.5
E = 2
eta = 1

# load parameters
n = 100
omegas = [1,1,2]
As = [1,2,3]

# harmonic data
eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
lp.plot_data(eps, eps_dot, sig, omegas, As, "data_cyclic.pdf")

# relaxation data
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
lp.plot_data(eps, eps_dot, sig, omegas, As, "data_relaxation.pdf")

#%% analytical Maxwell model
omegas = [1]
As = [1]

# build model
maxwell_analytical = MaxwellModel(E_infty, E, eta)

# check model prediction for harmonic data (should be exact)
eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = maxwell_analytical([eps, dts])
# lp.plot_model_pred(eps, sig, sig_m, omegas, As)

# check model prediction for relaxation data (should be exact)
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = maxwell_analytical([eps, dts])
# lp.plot_model_pred(eps, sig, sig_m, omegas, As)

#%% training data
omegas = [1,2]
As = [1,3]

eps_train1, _, sig_train1, dts_train1 = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
eps_train2, _, sig_train2, dts_train2 = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)

# train on both data sets
eps_train = tf.concat([eps_train1, eps_train2], 0)
sig_train = tf.concat([sig_train1, sig_train2], 0)
dts_train = tf.concat([dts_train1, dts_train2], 0)

# train only on cyclic data
# eps_train = eps_train1
# sig_train = sig_train1
# dts_train = dts_train1

# train only on relaxation data
# eps_train = eps_train2
# sig_train = sig_train2
# dts_train = dts_train2

#%% naive RNN
omegas = [1,1,2]
As = [1,2,3]

RNN = build_naive_RNN()

t1 = now()
print(t1)
tf.keras.backend.set_value(RNN.optimizer.learning_rate, 0.002)
h = RNN.fit([eps_train, dts_train], [sig_train], epochs=4000, verbose=2)
t2 = now()
print(f"took {t2 - t1} sec to calibrate the model")

# plot loss
fig = plt.figure(1, dpi=600)
fig_size = fig.get_size_inches()
fig.set_figwidth(fig_size[1]/2)
plt.semilogy(h.history["loss"], label="training loss")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
plt.legend()
plt.savefig("naive_loss.pdf", bbox_inches="tight")

# check the model prediction for harmonic data (mainly interpolation)
eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As, "naive_cyclic.pdf")

# check the model prediction for relaxation data (extrapolation)
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As, "naive_relaxation.pdf")

#%% Maxwell RNN
omegas = [1,1,2]
As = [1,2,3]

maxwell_RNN = build_maxwell_RNN(E_infty, E)

t1 = now()
print(t1)
tf.keras.backend.set_value(maxwell_RNN.optimizer.learning_rate, 0.002)
h = maxwell_RNN.fit([eps_train, dts_train], [sig_train], epochs=4000, verbose=2)
t2 = now()
print(f"took {t2 - t1} sec to calibrate the model")

# plot loss
fig = plt.figure(1, dpi=600)
fig_size = fig.get_size_inches()
fig.set_figwidth(fig_size[1]/2)
plt.semilogy(h.history["loss"], label="training loss")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
plt.legend()
plt.savefig("RNN_loss.pdf", bbox_inches="tight")

# check the model prediction for harmonic data (mainly interpolation)
eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = maxwell_RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As, "RNN_cyclic.pdf")

# check the model prediction for relaxation data (extrapolation)
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = maxwell_RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As, "RNN_relaxation.pdf")

#%% generalized standard model
omegas = [1,1,2]
As = [1,2,3]

GSM_RNN = build_GSM_RNN(eta)

t1 = now()
print(t1)
tf.keras.backend.set_value(GSM_RNN.optimizer.learning_rate, 0.002)
h = GSM_RNN.fit([eps_train, dts_train], [sig_train], epochs=4000,  verbose=2)
t2 = now()
print(f"took {t2 - t1} sec to calibrate the model")

# plot loss
fig = plt.figure(1, dpi=600)
fig_size = fig.get_size_inches()
fig.set_figwidth(fig_size[1]/2)
plt.semilogy(h.history["loss"], label="training loss")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
plt.legend()
plt.savefig("GSM_loss.pdf", bbox_inches="tight")

# check the model prediction for harmonic data (mainly interpolation)
eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = GSM_RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As, "GSM_cyclic.pdf")

# check the model prediction for relaxation data (extrapolation)
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = GSM_RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As, "GSM_relaxation.pdf")
