"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I
Task 3: Hyperelasticity II

==================

Authors: Henrik Hembrock, Jonathan Stollberg

12/2022
"""
import tensorflow as tf
import matplotlib as plt
import matplotlib.patches as mpatches

label_map = {0: "$P_{11}$", 1: "$P_{12}$", 2: "$P_{13}$",
             3: "$P_{21}$", 4: "$P_{22}$", 5: "$P_{23}$",
             6: "$P_{31}$", 7: "$P_{32}$", 8: "$P_{33}$"}
color_map = {0: "#ecbc00", 1: "#324379", 2: "#dedede",
             3: "#5c86c4", 4: "#f99c00", 5: "#dedede",
             6: "#dedede", 7: "#dedede", 8: "#bf3c3c"}

def plot_stress_strain(ax, 
                       strain, 
                       stress, 
                       strain_component, 
                       stress_components, 
                       reference_stress=None):
    handles = []
    for i in stress_components:
        ax.plot(strain[:,strain_component], stress[:,i],
                linestyle="-",
                linewidth=2,
                color=color_map[i]
                )
        handles.append(mpatches.Patch(color=color_map[i], label=label_map[i]))
        
        if reference_stress is not None:
            ax.plot(strain[:,strain_component], reference_stress[:,i], 
                    linewidth=0, 
                    markevery=10, 
                    markersize=4, 
                    markerfacecolor=color_map[i],
                    color=color_map[i],
                    marker="o")
            
    return handles

def plot_stress_stress(ax,
                       reference_stress,
                       stress,
                       components):
    handles = []
    max_stresses = []
    min_stresses = []
    for i in components:
        maximum = tf.math.reduce_max(tf.math.abs(reference_stress[:,i]))
        ax.plot(reference_stress[:,i]/maximum, stress[:,i]/maximum,
                linestyle="-",
                linewidth=2,
                color=color_map[i]
                )
        handles.append(mpatches.Patch(color=color_map[i], label=label_map[i]))
        
        max_stresses.append(tf.math.reduce_max(reference_stress[:,i])/maximum)
        min_stresses.append(tf.math.reduce_min(reference_stress[:,i])/maximum)
        
    maximum = max(max_stresses)
    minimum = min(min_stresses)
    ax.plot([minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            linewidth=1.5,
            color="black"
            )
    
    return handles

def plot_energy():
    pass