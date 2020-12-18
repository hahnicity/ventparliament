"""
Another circuit analysis function. Has own module because I'm trying to
be a little more systematic about it.
"""
import numpy as np
from importlib import reload as reload
from matplotlib import pyplot as plt

from parliament.mccay import circuit_tools

from parliament.mccay import u_dp_correlations


def pressure_control_RRC(vdot, pcirc, PEEP, volume0, signal_freq, pipe_info):
    """Analysis of a pressure control waveform based on a circuit model
    with resistor in series with a parallel resistor-capacitor unit.

    Flow rate and pressure should be in clinical units (vdot in L/min,
    volume0 in L, and pcirc and PEEP in cmH2O) and are converted to SI.
    Other data should be in SI (signal_freq in Hz). Outputs are in SI.

    Currently assumes upper airway resistance modeled by a pipe of a
    certain length.
    """

    vdot = np.array(vdot)
    pcirc = np.array(pcirc)

    # go to SI units
    vdot = vdot/(60.0*1000.0)           # L/min to m**3/s
    pcirc = pcirc*98.07                 # cmH2O to Pa
    PEEP = PEEP*98.07                # cmH2O to Pa
    volume0 = volume0/1000.0          # L to m**3

    # Find time in breath frame
    time = np.arange(0,len(vdot))/signal_freq

    # Get volume as a function of time
    volume = circuit_tools.find_volume(time, vdot, volume0)

    # Find inhalation and exhalation regions
    regions = circuit_tools.pressure_control_regions(vdot, make_plots=False)

    # Arrays for values in ramp region
    vdot_ramp = vdot[regions['ramp_start']:regions['ramp_end'] + 1]
    Rlung_ramp = np.zeros_like(vdot_ramp)
    Rpipe_ramp = np.zeros_like(vdot_ramp)

    i_ramp = 0  # index for ramp-region arrays

    # solve for resistances in the ramp region
    for i in range(regions['ramp_start'], regions['ramp_end'] + 1):

        # plt.scatter(i, vdot[i])

        # Find Rpipe_ramp with helper function
        Rpipe_ramp[i_ramp] = circuit_tools.find_Rpipe(vdot[i], pipe_info)

        Rlung_ramp[i_ramp] = (pcirc[i] - PEEP)/vdot[i] - Rpipe_ramp[i_ramp]

        # TODO: early values very sensitive to PEEP
        #   end up with large negative Rlung's
        #   TEMPORARILY if that happens, set Rlung_ramp = 0.0
        if Rlung_ramp[i_ramp] < 0:
            Rlung_ramp[i_ramp] = 0.0

        i_ramp = i_ramp + 1

    # plt.scatter(i+1, vdot[i+1],color='red')
    # plt.grid()
    # plt.plot(vdot_ramp)
    # plt.show()

    # plt.plot(Rlung_ramp, 'o-', label='Rlung')
    # plt.plot(Rpipe_ramp, 'o-', label='Rpipe')
    # plt.plot(Rlung_ramp + Rpipe_ramp, 'o-', label='Rlung + Rpipe')
    # plt.grid()
    # plt.legend(loc='best')
    # plt.show()

    # solve for compliances in exhalation
    vdot_exhale = vdot[regions['exhale_start']:regions['exhale_end'] + 1]
    volume_exhale = volume[regions['exhale_start']:regions['exhale_end'] + 1]
    time_exhale = time[regions['exhale_start']:regions['exhale_end'] + 1]
    vdot2_exhale = np.zeros_like(vdot_exhale)
    Rpipe_exhale = np.zeros_like(vdot_exhale)
    Rlung_exhale = np.zeros_like(vdot_exhale)
    volume2_exhale = np.zeros_like(vdot_exhale)

    i_exhale = 0

    for i in range(regions['exhale_start'], regions['exhale_end'] + 1):

        Rpipe_exhale[i_exhale] = circuit_tools.find_Rpipe(vdot[i],
                                                          pipe_info)

        Rlung_exhale[i_exhale] = np.interp(-vdot[i], vdot_ramp, Rlung_ramp)

        # TODO: requires a pretty large Rpipe in order to have vdot2 < 0
        #   for some of the region (and very large for all of it)
        #   is this an issue?? (just go with it ftm)
        vdot2_exhale[i_exhale] = (pcirc[i] \
                                  - PEEP \
                                  - Rpipe_exhale[i_exhale]*vdot[i]) \
                                  /Rlung_exhale[i_exhale]

        i_exhale = i_exhale + 1

    # TODO: can't find volume2_exhale, don't know constant of integration
    #   might be okay, still thinking about it
    #   leaning towards not okay
    volume2_exhale = circuit_tools.find_volume(time_exhale,
                                               vdot2_exhale,
                                               0.0)
    volume3_exhale = volume_exhale - volume2_exhale

    Clung_exhale = np.zeros_like(vdot_exhale)

    i_exhale = 0

    for i in range(regions['exhale_start'], regions['exhale_end'] + 1):

        Clung_exhale[i_exhale] = volume3_exhale[i_exhale] \
                                 /(pcirc[i] - PEEP - Rpipe_exhale[i_exhale])


        i_exhale = i_exhale + 1

    plt.plot(Clung_exhale)
    plt.title('Clung_exhale')
    plt.show()

    plt.plot(pcirc[regions['exhale_start']:regions['exhale_end'] + 1]/Rlung_exhale,
             'o-', label='pcirc')
    plt.plot(-PEEP*np.ones_like(vdot_exhale)/Rlung_exhale, 'o-', label='PEEP')
    plt.plot(-Rpipe_exhale*vdot_exhale/Rlung_exhale, 'o-', label='Rpipe')
    plt.plot(vdot2_exhale, 'o-', label='vdot2')
    plt.title('exhale quantities')
    plt.legend()
    plt.grid()
    plt.show()

    return 0
