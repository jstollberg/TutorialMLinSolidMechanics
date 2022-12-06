# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:23:34 2022

@author: jonat
"""

# def plot_load_path(F, P):
#     # plot stress and strain in normal direction
#     F11, F22, F33 = F[:,0,0], F[:,1,1], F[:,2,2]
#     P11, P22, P33 = P[:,0,0], P[:,1,1], P[:,2,2]

#     fig1, ax1 = plt.subplots(dpi=600)
#     ax1.plot(F11, P11, label="11")
#     ax1.plot(F22, P22, label="22")
#     ax1.plot(F33, P33, label="33")
#     ax1.set(xlabel="deformation gradient",
#             ylabel="first Piola-Kirchhoff stress")
#     ax1.legend()
#     ax1.grid()

#     # plot stress and strain in shear direction
#     F12, F13, F21, F23, F31, F32 = (F[:,0,1], F[:,0,2], F[:,1,0], F[:,1,2],
#                                     F[:,2,0], F[:,2,1])
#     P12, P13, P21, P23, P31, P32 = (P[:,0,1], P[:,0,2], P[:,1,0], P[:,1,2],
#                                     P[:,2,0], P[:,2,1])

#     fig2, ax2 = plt.subplots(dpi=600)
#     ax2.plot(F12, P12, label="12")
#     ax2.plot(F13, P13, label="13")
#     ax2.plot(F21, P21, label="21")
#     ax2.plot(F23, P23, label="23")
#     ax2.plot(F31, P31, label="31")
#     ax2.plot(F32, P32, label="32")
#     ax2.set(xlabel="deformation gradient",
#             ylabel="first Piola-Kirchhoff stress")
#     ax2.legend()
#     ax2.grid()

#     plt.show()
    
# def plot_equivalent(F, P):
#     # compute equivalent quantities
#     F_eq = equivalent(F, "strain")
#     P_eq = equivalent(P, "stress")

#     fig, ax = plt.subplots(dpi=600)
#     ax.plot(F_eq, P_eq)
#     ax.set(xlabel="equivalent deformation gradient",
#            ylabel="equivalent first Piola-Kirchhoff stress")
#     ax.grid()
#     plt.show()