"""
Pressure control LRC functions. Original versions ported from
circuit_analysis.RC_functions. Some functions in the old location may
still be useful.
"""
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from parliament.mccay import circuit_tools


def pc_LRC(FLR, vdot, pcirc, PEEP, volume0=0., signal_freq=50.,
           input_units='clinical', smooth_vdot=False,
           smooth_vdotdot=False, average_c=False,
           avoid_jump=False, error_if_nan=True):
    """Derives inductance (L), resistance (R), and compliance
    coefficients for a pressure control breath. Coefficients are
    nonlinear. In particular, inductance is a function of flow rate
    (vdot), resistance is a function of vdot, and compliance is either
    a function of volume or constant.

    Clinical units (L, min, cmH2O) and SI units (m**3, s, Pa) are
    supported.

    Parameters
    ----------
    FLR : two element list [FL (float), FR (float)]
        Estimated fractions of (pcirc - PEEP) due to resistance and
        inductance, respectively, in the first part of the breath.
        Usually found iteratively with pc_LRC_iterate.
    vdot : list of floats
        Flow rates in order of time
    pcirc : list of floats
        Circulation pressure in order of time
    PEEP : float
        Positive end-expiratory pressure (usually first value in pcirc)
    volume0 : float
        Initial volume (default 0)
    signal_freq : float
        Frequency of vdot and pcirc data (default 50 Hz)
    input_units : string
        Units for vdot, pcirc, and PEEP (default 'clinical' (L, min,
        cmH2O)). Otherwise assumed to be SI (m**3, s, Pa).
    smooth_vdot : bool
        If true, smooth vdot early in script (default False).
    smooth_vdotdot : bool
        If true, smooth vdotdot early in script (default False).
    average_c : bool
        If true, use an average compliance to reconstruct pcirc
        (default False).
    avoid_jump : bool
        If true, avoid three point after max(vdot) and three points
        before and after exhalation start (default False)
    error_if_nan : bool
        If true, raise an exception if a value in pcirc_recon is nan
        (default True)

    Returns
    -------
    pcirc_recon : array of floats
        Reconstructed pcirc (same length as pcirc)
    component_dict : dictionary
        Dictionary with arrays containing components of pcirc_recon.
        Keys : 'PEEP', 'L', 'R', 'C'
    interp_dict : dictionary
        Dictionary with data related to respiratory parameters.
        Keys :
        'vdot_L' : list of floats
            Flow rates for L coefficient table.
        'L' : list of floats
            Inductance coefficient (function of flow rates in 'vdot_L')
        'vdot_R' : list of floats
            Flow rates for R coefficient table.
        'R : list of floats
            Resistance coefficient (function of flow rates in 'vdot_R)
        'volume_C' : array of floats
            Volumes for C coefficient table.
        'C' : array of floats
            Compliance coefficient (function of volumes in 'volume_C)
        'C_mean' : float
            Average compliance for middle 50 percent of exhaled volumes
        'vdot_max_index' : int
            Index of max flow rate point
        'exhale_start_index' : int
            Index of the first point in exhalation
    """

    # preprocessing
    # -------------

    vdot = np.array(vdot)
    pcirc = np.array(pcirc)

    if input_units == 'clinical':
        vdot = vdot/(60.0*1000.0)           # L/min to m**3/s
        pcirc = pcirc*98.07                 # cmH2O to Pa
        PEEP = PEEP*98.07                   # cmH2O to Pa
        volume0 = volume0/1000.0            # L to m**3

    # get internal parameters: time, volume, vdotdot, max vdot
    # --------------------------------------------------------

    time = np.arange(0,len(vdot))/signal_freq

    if smooth_vdot:
        smooth_frac = 0.05
        lowess_out = lowess(vdot, time, \
                            frac=smooth_frac, is_sorted=True)
        vdot = lowess_out[:,1]

    volume = circuit_tools.find_volume(time, vdot, volume0)

    vdotdot = circuit_tools.find_vdotdot(time, vdot)

    if smooth_vdotdot:
        smooth_frac = 0.05
        lowess_out = lowess(vdotdot, time, \
                            frac=smooth_frac, is_sorted=True)
        vdotdot = lowess_out[:,1]

    vdot_max = max(vdot)

    # find ramp resistance and inductance based on vdot ramp values
    # -------------------------------------------------------------

    # fractions of pcirc - PEEP assigned to L and R (iterate externally)
    F_L = FLR[0]
    F_R = FLR[1]

    vdot_ramp = []
    vdotdot_ramp = []
    R_ramp = []
    L_ramp = []

    # iterate up flow rate ramp and calculate R(vdot) and L(vdot)
    i = 0
    while vdot[i] < vdot_max:

        vdot_ramp.append(vdot[i])
        vdotdot_ramp.append(vdotdot[i])

        R_ramp.append(F_R*(pcirc[i] - PEEP)/vdot[i])
        L_ramp.append(F_L*(pcirc[i] - PEEP)/vdotdot[i])

        # negative R's to zero
        if R_ramp[i] < 0:
            R_ramp[i] = 0.0

        i = i+1

    vdotdot_ramp = np.array(vdotdot_ramp)
    vdot_ramp = np.array(vdot_ramp)

    # also store that index, useful later
    vdot_max_index = i

    # find compliance by looking at exhalation
    # ----------------------------------------

    # Find start of exhalation
    i = vdot_max_index
    while vdot[i] >= 0:
        i = i + 1

    exhale_start_index = i

    # adjust vdotdot near exhale point if avoid_jump==true
    n_avoid = 3
    if avoid_jump:
        for i in range(exhale_start_index - n_avoid,
                        exhale_start_index + 1):
            vdotdot[i] = vdotdot[exhale_start_index - n_avoid - 1]
        for i in range(exhale_start_index, \
                        exhale_start_index + n_avoid + 1):
            vdotdot[i] = vdotdot[exhale_start_index + n_avoid + 2]

    C_exhale = []
    vdot_exhale = []
    volume_exhale = []
    vdotdot_exhale = []

    # iterate through exhalation to find C(volume)
    for i in range(exhale_start_index, len(vdot)):

        # interpolate to find R and L
        # TODO: make sign conventions the same
        R_temp = np.interp(-vdot[i], vdot_ramp, R_ramp)
        L_temp = np.interp(vdot[i], -vdot_ramp, L_ramp)

        C_exhale.append((volume[i] - volume0) \
                         / (pcirc[i] - PEEP - R_temp*vdot[i] - \
                         L_temp*vdotdot[i]))


        vdot_exhale.append(vdot[i])
        volume_exhale.append(volume[i])
        vdotdot_exhale.append(vdotdot[i])

    # smooth compliances
    smooth_frac = 0.1
    lowess_out = lowess(C_exhale, volume_exhale, \
                        frac=smooth_frac, is_sorted=False)
    volume_smooth =lowess_out[:,0]
    C_smooth = lowess_out[:,1]

    # average compliance for output and/or use in reconstruction
    # ----------------------------------------------------------

    # drop first and last 25% of volumes in exhale
    max_volume = max(volume_smooth)
    drop_volume = 0.25*max_volume
    high_volume = 0.75*max_volume

    i = 0
    while volume_smooth[i] <= drop_volume:
        i = i + 1
    drop_point = i

    i = 0
    while volume_smooth[i] <= high_volume:
        i = i + 1
    high_point = i

    C_mean = np.mean(C_smooth[drop_point:high_point])

    # reconstruct pcirc
    # -----------------

    # NOTE: might be better to switch these to appends rather than
    # preallocated arrays

    pcirc_recon = np.zeros(len(vdot))

    PEEP_component = np.zeros(len(vdot))
    L_component = np.zeros(len(vdot))
    R_component = np.zeros(len(vdot))
    C_component = np.zeros(len(vdot))

    for i in range(len(vdot)):

        # if average_c==true, use average compliance
        if average_c:
            C_temp = C_mean
        else:
            C_temp = np.interp(volume[i], volume_smooth, C_smooth)

        # interpolate for R and L
        if vdot[i] < 0:
            R_temp = np.interp(-vdot[i], vdot_ramp, R_ramp)
            L_temp = np.interp(-vdot[i], vdot_ramp, L_ramp)
        elif vdot[i] > 0:
            R_temp = np.interp(vdot[i], vdot_ramp, R_ramp)
            L_temp = np.interp(vdot[i], vdot_ramp, L_ramp)
        else:
            R_temp = 0.0
            L_temp = 0.0

        pcirc_recon[i] =  PEEP + (volume[i] - volume0)/C_temp \
                          + R_temp*vdot[i] + L_temp*vdotdot[i]

        PEEP_component[i] = PEEP
        L_component[i] = L_temp*vdotdot[i]
        R_component[i] = R_temp*vdot[i]
        C_component[i] = (volume[i] - volume0)/C_temp

    # Error handling
    # --------------

    if error_if_nan:
        for i, value in enumerate(pcirc_recon):
            assert(str(value) != 'nan'), 'nan at ' + str(i)

    # Output
    # ------

    component_dict = { 'PEEP' : PEEP_component,
                       'L' : L_component,
                       'R' : R_component,
                       'C' : C_component
                    }

    interp_dict = { 'vdot_L' : vdot_ramp,
                    'L' : np.array(L_ramp),
                    'vdot_R' : vdot_ramp,
                    'R' : np.array(R_ramp),
                    'volume_C' : volume_smooth,
                    'C' : C_smooth,
                    'C_mean' : C_mean,
                    'vdot_max_index' : vdot_max_index,
                    'exhale_start_index' : exhale_start_index
                    }

    return (pcirc_recon, component_dict, interp_dict)

def pc_LRC_iterate(FR_range, recon_tolerance, vdot, pcirc, PEEP, volume0=0,
                    signal_freq=50., input_units='clinical',
                    smooth_vdot=False, smooth_vdotdot=False, average_c=False,
                    avoid_jump=False):
    """Iterates pc_LRC on FR (and FL) to minimize error
    between ventilator pcirc and LRC reconstructed pcirc halfway
    through the fall-off region. The fall-off region is the time in
    inhalation between the max flow rate and the start of exhalation.

    FR and FL are the fractions of (pcirc - PEEP) due to resistance and
    inductance, respectively. The iteration is a standard bisection
    algorithm on FR starting from the low value in the allowed range.

    Parameters
    ----------
    FR_range : two-element list [FR_min (float), FR_max (float)]
        The range over which to iterate for ramp-resistance fraction.
    recon_tolerance : float
        Tolerance of abs(pcirc - pcirc_recon)/pcirc to iterate for
    All Other Parameters :
        All other parameters continue to internal instance of pc_LRC().
        Look at doc string of pc_LRC() in LRC_pc_functions.py or try
        help(pc_LRC) or help(LRC_pc_functions.pc_LRC).

    Returns
    -------
    FLR_final : two element list [FL (float), FR (float)]
        Final inductance and resistance fractions.
    """

    # TODO: still working on error handling a little
    class Error(Exception):
        """Base class for exceptions"""
        pass

    class LRCFailedError(Error):
        """Error for failure of LRC in iteration loop."""
        def __init__(self, cause):
            pass

    # Set up FR for iteration
    FR_min = FR_range[0]
    FR_max = FR_range[1]
    FR = FR_min

    # Make error > tolerance to ensure at least one step
    median_pcirc_error = 1. + recon_tolerance

    # Iteration counter
    i = 0

    # Iterate FLR via bisection to minimize fall-off median point error
    while np.abs(median_pcirc_error) >= recon_tolerance and i <= 20:
        i = i + 1

        FLR_iter = [1 - FR, FR]

        # LRC analysis
        try:
            (pcirc_recon, component_dict, interp_dict) = \
                pc_LRC(FLR_iter, vdot, pcirc, PEEP, \
                        volume0=volume0, signal_freq=signal_freq, \
                        input_units=input_units, \
                        smooth_vdot=smooth_vdot, \
                        smooth_vdotdot=smooth_vdotdot, average_c=average_c, \
                        avoid_jump=avoid_jump)
        except Exception as e:
            raise LRCFailedError(str(e) + " in iteration " + str(i))

        # Find middle (time median) of fall-off region
        median_point = int((interp_dict['exhale_start_index'] \
                            - interp_dict['vdot_max_index'])/2.0 \
                            + interp_dict['vdot_max_index'])

        # Error between pcirc and reconstructed pcirc at that point
        median_pcirc_error = (pcirc[median_point] \
                                    - pcirc_recon[median_point]/98.07) \
                                    /pcirc[median_point]

        FR_old = FR

        # Find new FR range
        # TODO: what about error == 0. ?
        if median_pcirc_error < 0.0:
            FR_min = FR_min
            FR_max = FR
            FR = (FR + FR_min)/2.0
        else:
            FR_max = FR_max
            FR_min = FR
            FR = (FR + FR_max)/2.0

    # FR_old satisfied tolerance so return that
    FLR_final = [1 - FR_old, FR_old]

    # Error for max iterations
    assert (i <= 20), "reached max iterations (20)"

    # Error for no change in iteration.
    assert(FLR_final[0] != FR_range[0] and FLR_final[1] != 1 - FR_range[0]),  \
            "iteration stalled, FR and FL are unchanged from initial values"

    return FLR_final



def pc_LRC_external(vdot, pcirc, PEEP, volume0, volume_smooth, C_smooth, vdot_ramp, R_ramp, L_ramp, C_mean, average_c=False, avoid_jump=False):

    """LRC model for a pressure control waveform using previous data.
    A BIT OF A MESS: parameters inconsistent with pc_LRC
    just wanted to get it going quickly
    """
    # convert to numpy arrays
    vdot = np.array(vdot)
    pcirc = np.array(pcirc)

    # go to SI units
    vdot = vdot/(60.0*1000.0)           # L/min to m**3/s
    pcirc = pcirc*98.07                 # cmH2O to Pa
    PEEP = PEEP*98.07                # cmH2O to Pa
    volume0 = volume0/1000.0          # L to m**3

    # Find time in breath frame
    # TODO: add signal frequency as a function argument
    time = np.arange(0,len(vdot))/50.0


    # Get volume as a function of time
    volume = circuit_tools.find_volume(time, vdot, volume0)

    # Get second derivative of volume as a function of time
    vdotdot = circuit_tools.find_vdotdot(time, vdot)

    # Find max flow rate
    vdot_max = max(vdot)

    i = 0

    # Find resistances up flow rate ramp
    while vdot[i] < vdot_max:

        i = i+1


    # also store that index, useful later
    vdot_max_index = i

    # Find start of exhalation
    i = vdot_max_index

    while vdot[i] >= 0:
        i = i + 1

    exhale_start_index = i

    # TODO: n_avoid?
    # TODO: placement compared to internal function?
    n_avoid = 3

    if avoid_jump:
        for i in range(exhale_start_index - n_avoid, exhale_start_index + 1):
            vdotdot[i] = vdotdot[exhale_start_index - n_avoid - 1]
        for i in range(exhale_start_index, \
                        exhale_start_index + n_avoid + 1):
            vdotdot[i] = vdotdot[exhale_start_index + n_avoid + 2]

    # Reconstruct pcirc

    pcirc_recon = np.zeros(len(vdot))

    PEEP_component = np.zeros(len(vdot))
    L_component = np.zeros(len(vdot))
    R_component = np.zeros(len(vdot))
    C_component = np.zeros(len(vdot))


    for i in range(len(vdot)):

        if average_c:
            C_temp = C_mean
        else:
            C_temp = np.interp(volume[i], volume_smooth, C_smooth)

        # C_temp = np.interp(volume[i], volume_smooth, C_smooth)

        if vdot[i] < 0:
            # R_temp = np.interp(vdot[i], -vdot_ramp, R_ramp)
            R_temp = np.interp(-vdot[i], vdot_ramp, R_ramp)
            # L_temp = np.interp(vdot[i], -vdot_ramp, L_ramp)
            L_temp = np.interp(-vdot[i], vdot_ramp, L_ramp)
        elif vdot[i] > 0:
            R_temp = np.interp(vdot[i], vdot_ramp, R_ramp)
            L_temp = np.interp(vdot[i], vdot_ramp, L_ramp)
        else:
            R_temp = 0.0
            L_temp = 0.0

        # TODO: pcirc + PEEP
        pcirc_recon[i] =  PEEP + (volume[i] - volume0)/C_temp \
                          + R_temp*vdot[i] + L_temp*vdotdot[i]

        # # TODO: abs(L*vdotdot)
        # pcirc_recon[i] =  PEEP + (volume[i] - volume0)/C_temp \
        #                   + R_temp*vdot[i] + np.abs(L_temp*vdotdot[i])

        PEEP_component[i] = PEEP
        L_component[i] = np.abs(L_temp*vdotdot[i])
        R_component[i] = R_temp*vdot[i]
        C_component[i] = (volume[i] - volume0)/C_temp

    return pcirc_recon
