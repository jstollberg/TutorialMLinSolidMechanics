"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Henrik Hembrock, Jonathan Stollberg, Dominik K. Klein
         
01/2023
"""
from matplotlib import pyplot as plt
import numpy as np

colors = np.array(["#ecbc00", "#324379", "#bf3c3c",
        # 'tab:blue', 'tab:orange', 'tab:green',
        # 'tab:red', 'tab:purple', 'tab:brown',
        # 'tab:pink', 'tab:gray', 'tab:olive'
        ])

def plot_data(eps, eps_dot, sig, omegas, As, file=None):
    n = len(eps[0])
    ns = np.linspace(0, 2*np.pi, n)
    
    plt.figure(dpi=600, figsize=(10, 8))
    # plt.title('Data')
    
    plt.subplot(2,2,1)
    for i in range(len(eps)):
        plt.plot(ns, sig[i], label = '$\\omega$: %.2f, $A$: %.2f' \
                 %(omegas[i], As[i]), color=colors[i], linestyle='--')
        plt.xlim([0, 2*np.pi])
        plt.ylabel('stress $\\sigma$')
        plt.xlabel('time $t$')
        plt.legend()
        
    plt.subplot(2,2,2)
    for i in range(len(eps)):
        plt.plot(eps[i], sig[i], color=colors[i], linestyle='--')
        plt.xlabel('strain $\\varepsilon$')
        plt.ylabel('stress $\\sigma$')
        
    plt.subplot(2,2,3)
    for i in range(len(eps)):
        plt.plot(ns, eps[i], color=colors[i], linestyle='--')
        plt.xlim([0, 2*np.pi])
        plt.xlabel('time $t$')
        plt.ylabel('strain $\\varepsilon$')
        
    plt.subplot(2,2,4)
    for i in range(len(eps)):
        plt.plot(ns, eps_dot[i], color=colors[i], linestyle='--')
        plt.xlim([0, 2*np.pi])
        plt.xlabel('time $t$')
        plt.ylabel(r'strain rate $\.{\varepsilon}$')
       
    if file is not None:
        plt.savefig(file, bbox_inches="tight")
    plt.show()
    
def plot_model_pred(eps, sig, sig_m, omegas, As, file=None):
    n = len(eps[0])
    ns = np.linspace(0, 2*np.pi, n)
    
    plt.figure(dpi = 600, figsize = (10,4))
    plt.suptitle('Data: dashed line, model prediction: continuous line')
    
    plt.subplot(1,2,1)
    for i in range(len(eps)):
        plt.plot(ns, sig[i], label = '$\\omega$: %.2f, $A$: %.2f'%(omegas[i], As[i]),linestyle='--', color=colors[i])
        plt.plot(ns, sig_m[i], color=colors[i])
        plt.xlim([0, 2*np.pi])
        plt.ylabel('stress $\\sigma$')
        plt.xlabel('time $t$')
        plt.legend()
        
        
    plt.subplot(1,2,2)
    for i in range(len(eps)):
        plt.plot(eps[i], sig[i], linestyle='--', color=colors[i])
        plt.plot(eps[i], sig_m[i], color=colors[i])
        plt.xlabel('strain $\\varepsilon$')
        plt.ylabel('stress $\\sigma$')

    if file is not None:
        plt.savefig(file, bbox_inches="tight")
    plt.show()