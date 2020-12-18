"""
Tools for circuit analysis.
"""

import numpy as np
import math
from matplotlib import pyplot as plt

from parliament.mccay import u_dp_correlations


def find_vdotdot(time, vdot):
    """Finds 2nd derivative of volume as a function of time from
    vdot.
    """

    dt = time[1] - time[0]

    vdotdot = np.gradient(vdot, dt)

    return vdotdot

def find_volume(time, vdot, volume0):
    """Finds volume as a function of time from vdot."""

    volume = np.zeros(len(vdot))

    for i in range(len(vdot)):

        volume[i] = np.trapz(vdot[0:i + 1], time[0:i + 1]) + volume0

    return volume

def pc_NRMSD(vdot, pcirc_vent, pcirc_recon, n_avoid=3):
    """Finds Normalized Root Mean Square Difference (NRMSD) between two
    pressure signals for a pressure control breath. Normalized by the
    difference between the max ventilator pressure and the ventilator
    pressure at the start of the breath. Avoids `n_avoid`points after
    max flow rate and before and after start of exhalation.

    pcirc_vent and pcirc_recon need the same units. Units of vdot do
    not matter.

    Parameters
    ----------
    vdot : list (or list-like) of floats
        Flow rate signal for breath.
    pcirc_vent : list (or list-like) of floats
        Ventilator pressure signal. Treated as true-value.
    pcirc_recon : list (or list-like) of floats
        Reconstructed pressure signal. Treated as experimental value.
    n_avoid : int
        Number of points to avoid after max flow rate and around exhale
        start. Almost always three.

    Returns
    -------
    nrmsd : float
        Normalized Root Mean Square Difference (NRMSD) between
        pcirc_vent and pcirc_recon
    """


    # find max point
    vdot_max = max(vdot)
    i = 0
    while vdot[i] < vdot_max:
        i = i + 1
    vdot_max_index = i

    # find exhale point
    i = vdot_max_index
    while vdot[i] >= 0:
        i = i + 1
    exhale_start_index = i

    # slice out avoid regions
    inhale_avoid = [vdot_max_index + i for i in range(n_avoid)]
    inhale_avoid_end = inhale_avoid[-1] + 1

    exhale_avoid = [exhale_start_index + i for i in range(-n_avoid, n_avoid)]
    exhale_avoid_start = exhale_avoid[0]
    exhale_avoid_end = exhale_avoid[-1] + 1

    # loop through pressures and calculate NRMSD
    nrmsd = 0
    n_points = 0 # points counted in NRMSD
    for i in range(len(pcirc_vent)):
        if any(i==x for x in inhale_avoid) or any(i==x for x in exhale_avoid):
            pass
        else:
            n_points = n_points + 1
            nrmsd = nrmsd + (pcirc_vent[i] - pcirc_recon[i])**2.

    nrmsd = math.sqrt(nrmsd/n_points)/(max(pcirc_vent) - pcirc_vent[0])

    return nrmsd

def find_Rpipe(vdot, pipe_info):
    """Finds resistance of a pipe given a flow rate. Designed as helper
    function for pressure_control_RRC().

    Always returns Rpipe > 0.
    """

    u_temp = vdot/(np.pi*(pipe_info['D']/2)**2)

    if u_temp < 0:
        u_temp = u_temp*-1.0

    dp_temp = u_dp_correlations.pipe_dp_of_u(u_temp,
                                             pipe_info['D'],
                                             pipe_info['L'],
                                             pipe_info['nu'],
                                             pipe_info['roughness'])

    Rpipe = dp_temp/vdot

    return Rpipe

def pressure_control_regions(vdot, make_plots):
    """Finds inhalation and exhalation points for a standard PC breath.
    """

    # flow rate ramp starts at beginning of breath if pre-processed properly.
    ramp_start = 0

    # flow rate ramp ends at maximum vdot
    vdot_max = max(vdot)

    i = 0

    while vdot[i] < vdot_max:
        i = i + 1

    ramp_end = i

    # fall region starts after ramp ends and goes until exhalation
    fall_start = ramp_end + 1

    i = fall_start

    while vdot[i] >= 0:
        i = i + 1

    fall_end = i - 1

    # exhale starts after fall region and goes until the end of the breath
    exhale_start = fall_end + 1

    exhale_end = len(vdot) - 1

    # another region is the neutral region, where the vent is not controlling
    # it's the last part of exhalation
    # not looking for it now but adding support
    neutral_start = 0
    neutral_end = 0

    if make_plots == True:
        plt.plot(vdot, 'b-')
        plt.plot(range(ramp_start, ramp_end+1), vdot[ramp_start:ramp_end+1], \
                 'ro', label='ramp')
        plt.plot(range(fall_start, fall_end+1), vdot[fall_start:fall_end+1], \
                 'go', label='fall')
        plt.plot(range(exhale_start, exhale_end+1), \
                 vdot[exhale_start:exhale_end+1],'yo', label ='exhale')
        plt.title('Breath Regions')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

        plt.plot(vdot, 'bo-')
        plt.plot(ramp_start, vdot[ramp_start], 'yo')
        plt.plot(ramp_end, vdot[ramp_end], 'ro')
        plt.plot(fall_start, vdot[fall_start], 'yo')
        plt.plot(fall_end, vdot[fall_end], 'ro')
        plt.plot(exhale_start, vdot[exhale_start], 'yo')
        plt.plot(exhale_end, vdot[exhale_end], 'ro')
        plt.title('Breath Region Start and End Points')
        plt.grid()
        plt.show()

    return  {'ramp_start' : ramp_start,
             'ramp_end' : ramp_end,
             'fall_start' : fall_start,
             'fall_end' : fall_end,
             'exhale_start' : exhale_start,
             'exhale_end' : exhale_end,
             'neutral_start' : neutral_start,
             'neutral_end' : neutral_end
            }
