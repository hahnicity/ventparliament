"""
predator
~~~~~~~~

Follows Redmond's 2014 paper "Pressure reconstruction by eliminating the demand effect
of spontaneous respiration (PREDATOR) method for assessing respiratory mechanics of
reverse-triggered breathing cycles"

According to Redmond:

"I believe we used 5 consecutive previous breaths - 4 prior, and the breath itself."

But this would stand in contrast to his paper where they say they only use 1 breath, which
would have to be the prior breath and the current.

Presumably PREDATOR could be used for either flow or pressure waveforms.
"""
import numpy as np

from parliament.other_calcs import least_squares_method


def perform_pressure_reconstruction(pressure_waveforms):
    reconstructed = []
    longest = max([len(wave) for wave in pressure_waveforms])
    for i in range(longest):
        vals = []
        for wave in pressure_waveforms:
            try:
                vals.append(wave[i])
            except IndexError:
                continue
        reconstructed.append(np.percentile(vals, 90))
    return reconstructed


def perform_predator_algo(pressure_waveforms, flow, x0_index, dt, peep, tvi):
    """
    Implement PREDATOR as mentioned in Redmond et al. 2014.

    :param pressure_waveforms:
    :param flow: flow waveform in L/s
    :param x0_index: index at which inhalation ends, exhalation begins
    :param dt: time between observations
    :param peep: baseline pressure setting for the vent (PEEP)
    :param tvi: tidal volume inhaled on breath
    """
    # it's totally possible that PREDATOR could screw up the x0 by lengthening pressure curve
    # longer than it should be. This is very clearly seen in the case where there is a plateau
    # pressure in prior breaths.
    reconstructed_pressure = perform_pressure_reconstruction(pressure_waveforms)
    reconstructed_pressure = reconstructed_pressure[:len(flow)]
    return least_squares_method(flow, reconstructed_pressure, x0_index, dt, peep, tvi)
