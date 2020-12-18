'''
script to find inspiratory volume of a breath
primary purpose is to compare volumes to see how that effects expiration

NOTE:
    * currently assumes 50 Hz signal
    * input to vdot is in L/min
    * outputs vdot in L and exhale_start_index as tuple
'''
import numpy as np
from matplotlib import pyplot as plt


def find_inspiratory_volume(vdot):

    # convert to numpy array
    vdot = np.array(vdot)

    # convert vdot to SI
    vdot = vdot/(1000.0*60.0)

    # Find beginning of exhalation
    i = 5       # skips the first few points in the breath

    while (vdot[i] > 0) and vdot[i+1] > 0:


        i = i + 1

    exhale_start_index = i + 1

    vdot_inhale = vdot[0:exhale_start_index]

    # Find time
    # TODO: currently assumes 50 Hz signal, would be nice to add time support
    time_inhale = np.arange(0,len(vdot_inhale))/50.0

    # Now find volume
    volume  = np.trapz(vdot_inhale, time_inhale)

    # return volume in L/min
    return (volume*1000.0, exhale_start_index)
