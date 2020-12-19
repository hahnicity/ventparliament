"""
polynomial_model
~~~~~~~~~~~~~~~~

From Redmond's 2015 paper "A polynomial model of patient-specific breathing effort
during controlled mechanical ventilation"

They start with the equation

    Paw = E*V + R*V^dot + P0 + Pe

And then they give some cases where Pe would not be 0, and how to model it
like a second order polynomial equation.
"""
import numpy as np
from scipy.integrate import simps

from parliament.other_calcs import inspiratory_least_squares as least_squares_method


def get_predicted_pressure_waveform(flow, pressure, peep, x0, tvi, vols):
    """
    Get the pressure waveform as predicted by the least squares model

        Paw = E*V + R*V^dot + P0

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param peep: positive end expiratory pressure
    :param x0: index where flow crosses 0
    :param tvi: TVi in L
    :param vols: calculated volumes inspired over time
    """
    plat, comp, resist, K, residual = least_squares_method(flow, vols, pressure, x0, 0.02, peep, tvi)
    modeled_pressure = (1 / comp) * vols[:x0-1] + resist * flow[:x0-1] + K
    return modeled_pressure, comp, resist, residual


def find_pressure_regions(actual_pressure, modeled_pressure):
    """
    Find regions where the actual pressure curve is above/below the modeled curve. This
    function will find these regions by their start and end indexes, and return information
    about how large those regions were

    :returns list: [[<region start idx>, <region end idx>, <magnitude diff model-actual>], ...]
    """
    actual = actual_pressure[:len(modeled_pressure)]
    diff = modeled_pressure - actual
    aucs = np.array([0] + [simps(diff[:i], dx=0.02) for i in range(2, len(diff))])
    is_endpoint = False

    regions = [[0, None, None]]
    if aucs[1] >= aucs[0]:
        prev_grad = 1
    else:
        prev_grad = -1

    # The logic behind some of this is that regions with positive auc is more important
    # because that means the Pmodel > Pdata. So ensure that they are contiguous with
    # each other even if there is a 0 gradient somewhere in series.
    for idx, auc in enumerate(aucs[2:]):
        # region is growing in size and Pmodel > Pdata
        if auc - aucs[idx-1] >= 0 and prev_grad == -1:
            prev_grad = 1
            is_endpoint = True
        # region is declining in size and Pmodel < Pdata
        elif auc - aucs[idx-1] < 0 and prev_grad == 1:
            prev_grad = -1
            is_endpoint = True

        if is_endpoint:
            if abs(modeled_pressure[idx-1] - actual_pressure[idx-1]) < abs(modeled_pressure[idx] - actual_pressure[idx]):
                endpoint = idx - 1
            else:
                endpoint = idx
            regions[-1][1] = endpoint
            regions[-1][2] = aucs[endpoint] - aucs[regions[-1][0]]
            regions.append([idx, None, None])
            is_endpoint = False
    else:
        regions[-1][1] = idx
        regions[-1][2] = aucs[idx] - aucs[regions[-1][0]]
    return regions


def find_ts_tf(actual_pressure, modeled_pressure):
    regions = find_pressure_regions(actual_pressure, modeled_pressure)
    largest = max(regions, key=lambda x: x[2])
    if largest[2] > 0:
        return largest[0], largest[1]
    else:
        return -1, -1


def perform_least_squares_fit(flow, pressure, vols, peep, x0, ts, tf):
    dt = 0.02
    a_coef = np.zeros(x0-1)
    b_coef = np.zeros(x0-1)
    c_coef = np.zeros(x0-1)
    a_coef[ts:tf+1] = (np.arange(ts, tf+1) * dt) ** 2
    b_coef[ts:tf+1] = np.arange(ts, tf+1) * dt
    c_coef[ts:tf+1] = np.ones(tf+1-ts)
    arr = np.array([vols[:x0-1], flow[:x0-1], a_coef, b_coef, c_coef]).transpose()
    result = np.linalg.lstsq(arr, pressure[:x0-1] - peep)
    return result


def perform_polynomial_model(flow, vols, pressure, x0, peep, tvi):
    """
    Perform the Redmond algorithm for estimating patient effort using a polynomial
    eq. This algorithm has some drawbacks but it is a good proof of concept
    for estimating effort.

    :param flow: array vals of flow measurements in L/s
    :param vols: calculated volumes inspired over time
    :param pressure: array vals of pressure obs
    :param peep: positive end expiratory pressure
    :param x0: index where flow crosses 0 and expiration starts
    :param tvi: TVi in L

    :returns tuple: (compliance, resistance, residual, response code)

    Response codes:
        0: Used algorithm and was successful
        2: Used least squares as fallback and was successful
    """
    flow = np.array(flow)
    pressure = np.array(pressure)
    modeled_pressure, compliance, resistance, _ = get_predicted_pressure_waveform(flow, pressure, peep, x0, tvi, vols)
    ts, tf = find_ts_tf(pressure, modeled_pressure)
    # Perform least squares modeling
    result = perform_least_squares_fit(flow, pressure, vols, peep, x0, ts, tf)
    solns = result[0]

    # Check roots of the polynomial eq
    roots = np.roots(solns[2:])
    for root in roots:
        if np.imag(root) != 0:
            all_roots_real = False
            break
    else:
        all_roots_real = True

    if not ((0 <= solns[0] <= 500) and (0 <= solns[1] <= 500) and solns[2] > 0 and all_roots_real):
        _, comp, resist, K, residual = least_squares_method(flow, vols, pressure, x0, 0.02, peep, tvi)
        return comp, resist, residual[0], 2
    else:
        return 1 / solns[0], solns[1], result[1][0], 0
