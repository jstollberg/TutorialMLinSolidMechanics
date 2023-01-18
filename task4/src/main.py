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
from models import MaxwellModel, build_naive_RNN, build_maxwell_RNN
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
lp.plot_data(eps, eps_dot, sig, omegas, As)

# relaxation data
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
lp.plot_data(eps, eps_dot, sig, omegas, As)

#%% analytical Maxwell model
omegas = [1,1,2]
As = [1,2,3]

# build model
maxwell_analytical = MaxwellModel(E_infty, E, eta)

# check model prediction for harmonic data (should be exact)
eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = maxwell_analytical([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As)

# check model prediction for relaxation data (should be exact)
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = maxwell_analytical([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As)

#%% training data
omegas = [1]
As = [1]

eps_train, _, sig_train, dts_train = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
# eps_train, _, sig_train, dts_train = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)

#%% naive RNN
omegas = [1,1,2]
As = [1,2,3]

RNN = build_naive_RNN()

t1 = now()
print(t1)
tf.keras.backend.set_value(RNN.optimizer.learning_rate, 0.002)
h = RNN.fit([eps_train, dts_train], [sig_train], epochs = 2000,  verbose = 2)
t2 = now()
print(f"took {t2 - t1} sec to calibrate the model")

# plot loss
plt.figure(1, dpi=600)
plt.semilogy(h.history["loss"], label="training loss")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
plt.legend()

# check the model prediction for harmonic data (mainly interpolation)
eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As)

# check the model prediction for relaxation data (extrapolation)
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As)

#%% Maxwell RNN
omegas = [1,1,2]
As = [1,2,3]

maxwell_RNN = build_maxwell_RNN()

t1 = now()
print(t1)
tf.keras.backend.set_value(maxwell_RNN.optimizer.learning_rate, 0.002)
h = maxwell_RNN.fit([eps_train, dts_train], [sig_train], epochs=2000,  verbose=2)
t2 = now()
print(f"took {t2 - t1} sec to calibrate the model")

# plot loss
plt.figure(1, dpi=600)
plt.semilogy(h.history["loss"], label="training loss")
plt.grid(which="both")
plt.xlabel("calibration epoch")
plt.ylabel("log$_{10}$ MSE")
plt.legend()

# check the model prediction for harmonic data (mainly interpolation)
eps, eps_dot, sig, dts = ld.generate_data_harmonic(E_infty, E, eta, n, omegas, As)
sig_m = maxwell_RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As)

# check the model prediction for relaxation data (extrapolation)
eps, eps_dot, sig, dts = ld.generate_data_relaxation(E_infty, E, eta, n, omegas, As)
sig_m = maxwell_RNN([eps, dts])
lp.plot_model_pred(eps, sig, sig_m, omegas, As)

#%% generalized standard model