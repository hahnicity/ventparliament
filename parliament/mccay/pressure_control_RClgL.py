"""
Single function module for evaluating a different way of doing an RLC model
"""
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib import pyplot as plt

from parliament.mccay import circuit_tools


def pressure_control_RClgL(vdot, pcirc, PEEP, volume0, signal_freq,
                           plot_type):
    """Different way of doing an LRC analysis."""

    vdot = np.array(vdot)
    pcirc = np.array(pcirc)

    # go to SI units
    vdot = vdot/(60.0*1000.0)           # L/min to m**3/s
    pcirc = pcirc*98.07                 # cmH2O to Pa
    PEEP = PEEP*98.07                   # cmH2O to Pa
    volume0 = volume0/1000.0            # L to m**3

    # Find time in breath frame
    time = np.arange(0,len(vdot))/signal_freq

    # Find volume as a function of time
    volume = circuit_tools.find_volume(time, vdot, volume0)

    # Find max flow rate
    vdot_max = max(vdot)

    # TODO: use circuit_tools to find region indices as in RRC

    # Find ramp resistances
    vdot_ramp = []
    R_ramp = []

    i = 0

    # Find resistances up flow rate ramp
    while vdot[i] < vdot_max:

        vdot_ramp.append(vdot[i])
        R_ramp.append((pcirc[i] - PEEP)/vdot[i])

        # R < 0 doesn't make sense so set R < 0 to R = 0
        if R_ramp[i] < 0:
            R_ramp[i] = 0.0

        i = i+1

    # Find resistance at max (the last step)
    vdot_ramp.append(vdot[i])
    R_ramp.append((pcirc[i] - PEEP)/vdot[i])

    vdot_ramp = np.array(vdot_ramp)
    R_ramp = np.array(R_ramp)

    # also store that index, useful later
    vdot_max_index = i

    # Find start of exhalation
    i = vdot_max_index

    while vdot[i] >= 0:
        i = i + 1

    exhale_start_index = i

    # Find compliances
    C_exhale = []
    vdot_exhale = []
    volume_exhale = []

    print(np.all(np.diff(vdot_ramp) > 0))

    # TODO: limit compliance deriviation to exhale, stop before neutral region
    for i in range(exhale_start_index, len(vdot)):

        # Interpolate to find R
        # vdot < 0 in exhale but > 0 in ramp so multiply vdot[i] by -1.0
        R_temp = np.interp(-vdot[i], vdot_ramp, R_ramp)

        C_exhale.append((volume[i] - volume0) \
                         / (pcirc[i] - PEEP - R_temp*vdot[i]))
        vdot_exhale.append(vdot[i])
        volume_exhale.append(volume[i])

    # Process compliances
    # TODO: smoothing parameters?
    smooth_frac = 0.1
    lowess_out = lowess(C_exhale, volume_exhale, \
                        frac=smooth_frac, is_sorted=False)
    volume_smooth =lowess_out[:,0]
    C_smooth = lowess_out[:,1]

    # Look at fall-off region to find L

    vdot_fall = vdot[vdot_max_index:exhale_start_index]
    volume_fall = volume[vdot_max_index:exhale_start_index]
    pcirc_fall = pcirc[vdot_max_index:exhale_start_index]
    time_fall = time[vdot_max_index:exhale_start_index]

    polyfit_coeffs = np.polyfit(time_fall, vdot_fall, 1)
    print(polyfit_coeffs)

    vdot_fit = time_fall*polyfit_coeffs[0] + polyfit_coeffs[1]

    # numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)

    plt.plot(time, vdot)
    plt.plot(time_fall, vdot_fall,'ro-')
    plt.plot(time_fall, vdot_fit, 'k')
    plt.grid()
    plt.show()

    C_fall = []
    R_fall = []

    # Find an average value for L based on fall region
    for i in range(vdot_max_index + 1, exhale_start_index-1):
        C_fall.append(np.interp(volume[i], volume_smooth, C_smooth))
        R_fall.append(np.interp(vdot[i], vdot_ramp, R_ramp))

    C_fall_mean = np.mean(C_fall)
    R_fall_mean = np.mean(R_fall)
    vdot_fall_mean = np.mean(vdot_fall)
    volume_fall_mean = np.mean(volume_fall)
    pcirc_fall_mean = np.mean(pcirc_fall)

    L_approx = (pcirc_fall_mean - PEEP - R_fall_mean*vdot_fall_mean \
                - (volume_fall_mean - volume0)/C_fall_mean) \
                /polyfit_coeffs[0]

    print(L_approx)

    # Try to reconstruct pcirc
    pcirc_recon = []

    vdotdot = circuit_tools.find_vdotdot(time, vdot)

    for i in range(len(vdot)):

        C_temp = np.interp(volume[i], volume_smooth, C_smooth)

        # as in compliance derivation, make negative vdots positive for interp.
        if vdot[i] < 0:
            R_temp = np.interp(-vdot[i], vdot_ramp, R_ramp)
        elif vdot[i] > 0:
            R_temp = np.interp(vdot[i], vdot_ramp, R_ramp)
        else:
            R_temp = 0.0

        pcirc_recon.append(PEEP + L_approx*vdotdot[i] + 2.0*R_temp*vdot[i] \
                           + 2.0*(volume[i] - volume0)/C_temp)

    plt.plot(pcirc, 'bo-')
    plt.plot(pcirc_recon, 'ro-')
    plt.show()
