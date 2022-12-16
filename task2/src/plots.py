"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 2: Hyperelasticity I
Task 3: Hyperelasticity II

==================

Authors: Henrik Hembrock, Jonathan Stollberg

12/2022
"""
import numpy as np
import tensorflow as tf
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
    """
    Plot a stress-strain curve.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot to contain the curves.
    strain : tensorflow.Tensor
        The strain tensor in Voigt notation.
    stress : tensorflow.tensor
        The stress tensor in Voigt notation.
    strain_component : int
        The strain component to use on the x-axis.
    stress_components : list
        The components of the stresses to plot.
    reference_stress : tensorflow.Tensor, optional
        A reference stress data set to plot as markers. The default is None.

    Returns
    -------
    handles : list
        The handle objects to create a legend.

    """
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
                    markevery=3, 
                    markersize=4.5, 
                    markerfacecolor=color_map[i],
                    color=color_map[i],
                    marker="o")
            
    return handles

def plot_stress_stress(ax,
                       reference_stress,
                       stress,
                       components):
    """
    Plot stress components over the same component from a reference data set.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot to contain the curves.
    reference_stress : tensorflow.Tensor
        The reference stress tensor in Voigt notation.
    stress : tensorflow.Tensor
        The stress tensor in Voigt notation.
    components : list
        The components to plot.

    Returns
    -------
    handles : list
        The handle objects to create a legend.

    """
    handles = []
    max_stresses = []
    min_stresses = []
    for i in components:
        maximum = tf.math.reduce_max(tf.math.abs(reference_stress[:,i]))
        if np.allclose(maximum.numpy(), 0.0, rtol=1e-4, atol=1e-4):
            continue
        
        ax.plot(reference_stress[:,i]/maximum, stress[:,i]/maximum,
                linestyle="-",
                linewidth=2,
                color=color_map[i]
                )
        handles.append(mpatches.Patch(color=color_map[i], label=label_map[i]))
        
        max_stresses.append(tf.math.reduce_max(reference_stress[:,i])/maximum)
        min_stresses.append(tf.math.reduce_min(reference_stress[:,i])/maximum)
    
    if len(max_stresses) != 0 and len(min_stresses) != 0:
        maximum = max(max_stresses)
        minimum = min(min_stresses)
        ax.plot([minimum, maximum],
                [minimum, maximum],
                linestyle="--",
                linewidth=1.5,
                color="black"
                )
    
    return handles

def plot_energy(ax, 
                strain, 
                energy, 
                strain_component, 
                reference_energy=None):
    """
    Plot the energy over a strain component.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The subplot to contain the curves.
    strain : tensorflow.Tensor
        The strain tensor in Voigt notation.
    energy : tensorflow.Tensor
        The energy.
    strain_component : int
        The strain component used on the x-axis.
    reference_energy : tensorflow.Tensor, optional
        A reference data set for the energy. The default is None.

    Returns
    -------
    None.

    """
    ax.plot(strain[:,strain_component], energy,
            linestyle="-",
            linewidth=2,
            color="black"
            )
        
    if reference_energy is not None:
        ax.plot(strain[:,strain_component], reference_energy, 
                linewidth=0, 
                markevery=3, 
                markersize=4.5, 
                markerfacecolor="black",
                color="black",
                marker="o")