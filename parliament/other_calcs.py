import math

import numpy as np
from scipy.integrate import simps
from scipy.stats import linregress


def al_rawas_expiratory_const(flow, x0_index, dt, tvi, linregress_tol):
    """
    Calculate the Al Rawas time constant (tau_E)

    Al-Rawas N, Banner MJ, Euliano NR, Tams CG, Brown J, Martin AD, Gabrielli A. Expiratory
    time constant for determinations of plateau pressure, respiratory system compliance,
    and total resistance. Critical care. 2013 Feb 1;17(1):R23.

    :param flow: array vals of flow measurements in L/s
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param tvi: TVi in L
    :param linregress_tol: tolerance for the residual on linear regression
    """
    if len(flow[x0_index:]) <= int(.5 / dt):
        return np.nan
    flow = flow[x0_index:]
    start_idx = int(.1 / dt)
    end_idx = int(.5 / dt)
    vols = [tvi + simps(flow[0:i], dx=dt) for i in range(2,len(flow[:end_idx+1]))]
    x = [abs(val) for val in flow[start_idx:end_idx]]
    regress = linregress(x, vols[start_idx-1:end_idx-1])
    if regress.rvalue < linregress_tol:
        return np.nan
    return regress.slope


def al_rawas_calcs(flow, pressure, x0_index, dt, pip, peep, tvi, compliance_idx, linregress_tol=.98):
    """
    Calculate compliance using al-rawas methodology

    Al-Rawas N, Banner MJ, Euliano NR, Tams CG, Brown J, Martin AD, Gabrielli A. Expiratory
    time constant for determinations of plateau pressure, respiratory system compliance,
    and total resistance. Critical care. 2013 Feb 1;17(1):R23.

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure measurements from vent
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param pip: peak insp pressure
    :param peep: positive end expiratory pressure
    :param tvi: TVi in L
    :param compliance_idx: After computing the compliance curve, choose which indexwe want to use
    :param linregress_tol: tolerance for the residual on linear regression

    :returns tuple: tau, plat, compliance, resistance
    """
    if len(flow[:x0_index-1]) <= compliance_idx:
        return np.nan, np.nan, np.nan, np.nan
    tau = al_rawas_expiratory_const(flow, x0_index, dt, tvi, linregress_tol)
    vols = [0] + [simps(flow[0:i], dx=dt) for i in range(2, len(flow)+1)]
    compliance_curve = [
        (vol + tau * flow[i]) / (max(pressure[:i+1]) - peep)
        for i, vol in enumerate(vols[:x0_index-1])
    ]
    compliance = compliance_curve[compliance_idx]
    resistance = tau / compliance
    plat = tvi * (1 / compliance) + peep
    return round(tau, 4), round(plat, 2), round(compliance, 4), round(resistance, 2)


def brunner(tve, e_time, f_min, iters):
    """
    Perform the recursive brunner function. This provides an estimate of the
    expiratory time constant tau. tau is defined as the time it takes the lungs to
    inflate to 63% of volume or deflate by 63%. tau is the product of resistance and compliance.

    Brunner JX, Laubscher TP, Banner MJ, Iotti G, Braschi A. Simple method to measure total
    expiratory time constant based on the passive expiratory flow-volume curve. Critical
    care medicine. 1995 Jun 1;23(6):1117-22.

    :param tve: expiratory volume in liters (L)
    :param e_time: expiratory time in seconds
    :param f_min: abs value of maximum expiratory flow rate in L/s
    :param iters: number iterations to perform on the algorithm
    """
    if f_min == 0 or tve == 0 or e_time == 0:
        return np.nan

    bru = tve / f_min
    for i in range(iters):
        # this should only explode if (1 - e^(-e_time/bru)) gets very small. This would happen in
        # the case that e^(-e_time/bru) -> 1 => -e_time/bru -> 0. This could presumably happen
        # as bru outgrows e_time significantly, which is kinda a chicken and the egg thing. It
        # will only explode if it explodes.
        #
        # The main thing to ask is whether this grows or decreases over time tho. If it grows,
        # then additional iterations should only hurt us.
        #
        # OK I did the math and this function will increase in the case that
        # (t_e * e^(t_e/bru)) / bru * (t_e + bru) < 1. So we might be able to use this to guide
        # iterative bounds. For now though I'm going to stick to the paper.
        bru = bru * (1 / (1 - math.exp(-e_time/bru)))
    return bru


def perform_least_squares(a, pressure, tvi, peep):
    """
    Helper function for performing least squares regression with single chamber model
    """
    least_square_result = np.linalg.lstsq(a, np.array(pressure))
    solution = least_square_result[0]
    elastance = solution[0]
    plat = tvi * elastance + peep
    return plat, 1 / elastance, solution[1], solution[2], least_square_result[1]


def expiratory_least_squares(flow, pressure, x0_index, dt, peep, tvi):
    """
    Calculate compliance, resistance, and K via standard single chamber
    model equation. Only looks at expiratory part of breath:

    This method was exemplified by:

    Van Drunen EJ, Chiew YS, Chase JG, Shaw GM, Lambermont B, Janssen N, Damanhuri NS, Desaive T.
    Expiratory model-based method to monitor ARDS disease state. Biomedical engineering online.
    2013 Dec 1;12(1):57.

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param peep: positive end expiratory pressure
    :param tvi: TVi in L

    :returns tuple: plateau pressure, compliance, resistance, K, residual
    """
    # there was no identifiable expiratory location
    if x0_index > len(flow):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    vols = [0] + [simps(flow[:i], dx=dt) for i in range(2, len(flow)+1)]
    vols = vols[x0_index-1:]
    flow = flow[x0_index-1:]
    pressure = pressure[x0_index-1:]
    a = np.array([vols, flow, [1]*len(flow)]).transpose()
    return perform_least_squares(a, pressure, tvi, peep)


def howe_expiratory_least_squares(flow, pressure, x0_index, dt, peep, tvi):
    """
    Calculate compliance, resistance, and K via standard single chamber
    model equation. Only looks at expiratory section of breath and perform
    additional modifications suggested by Howe et al. 2020

    Howe SL, Chase JG, Redmond DP, Morton SE, Kim KT, Pretty C, Shaw GM, Tawhai MH, Desaive T.
    Inspiratory respiratory mechanics estimation by using expiratory data for reverse-triggered
    breathing cycles. Computer methods and programs in biomedicine. 2020 Apr 1;186:105184.

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param peep: positive end expiratory pressure
    :param tvi: TVi in L

    :returns tuple: plateau pressure, compliance, resistance, K, residual
    """
    # there was no identifiable expiratory location
    if x0_index > len(flow):
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    start_idx = x0_index - 1
    vols = [0] + [simps(flow[start_idx:i], dx=dt) for i in range(start_idx+2, len(flow)+1)]
    flow = flow[start_idx:]
    pressure = np.array(pressure[start_idx:])
    pressure = pressure - pressure[0]
    a = np.array([vols, flow, [1]*len(flow)]).transpose()
    return perform_least_squares(a, pressure, tvi, peep)


def inspiratory_least_squares(flow, pressure, x0_index, dt, peep, tvi):
    """
    Calculate compliance, resistance, and K via standard single chamber
    model equation. Only looks at inspiratory part of breath:

    P_vent = V(t)E + V_dot(t)R + K

    where K is some constant that is defined by PEEP + PEEPi + P_0

    This method is will work in more situations than other methods will,
    but it's also going to be the least accurate, especially when a patient
    is exerting force of their muscles

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param peep: positive end expiratory pressure
    :param tvi: TVi in L

    :returns tuple: plateau pressure, compliance, resistance, K, residual
    """
    end_idx = x0_index if x0_index <= len(flow) else len(flow)
    vols = [0] + [simps(flow[:i], dx=dt) for i in range(2, end_idx+1)]
    a = np.array([vols, flow[:end_idx], [1]*end_idx]).transpose()
    return perform_least_squares(a, pressure[:end_idx], tvi, peep)


def lourens_time_const(flow, x0_index, dt):
    """
    Calculate Lourens time constant for patients. Lourens performed her
    calculations on paralyzed COPD patients. So it might be helpful in
    this case.

    Lourens M, van den Berg B, Aerts JG, Verbraak A, Hoogsteden H, Bogaard J. Expiratory
    time constants in mechanically ventilated patients with and without COPD. Intensive
    care medicine. 2000 Nov 1;26(11):1612-8.

    :param flow: array vals of flow measurements in L/s
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs

    :returns tuple: tau_.75, tau_.5, tau_.25
    """
    min_idx = flow.index(min(flow))
    flow = [abs(v) for v in flow[min_idx:]]
    vols = [0] + [simps(flow[0:i], dx=dt) for i in range(2, len(flow)+1)]
    tve = vols[-1]
    one_quarter, half, three_quarter = tve / 4.0, tve / 2.0, tve * (3/4.0)
    quarter_idx, half_idx, three_quarter_idx = None, None, None
    for idx, v in enumerate(vols[1:]):
        if v > one_quarter and vols[idx-1] <= one_quarter:
            quarter_idx = idx
        elif v > half and vols[idx-1] <= half:
            half_idx = idx
        elif v > three_quarter and vols[idx-1] <= three_quarter:
            three_quarter_idx = idx
    tau_1 = (tve * .75) / (flow[quarter_idx] - flow[-1])
    tau_2 = (tve * .5) / (flow[half_idx] - flow[-1])
    tau_3 = (tve * .25) / (flow[three_quarter_idx] - flow[-1])
    return tau_1, tau_2, tau_3


def vicario_nieap(flow, pressure, x0_index, peep, tvi, tau):
    """
    Calculate plateau pressure, compliance, and resistance via Vicario's
    method for NonInvasive Estimation of Alveolar Pressure (NIEAP).

    Vicario F, Buizza R, Truschel WA, Chbat NW. Noninvasive estimation of alveolar pressure.
    In2016 38th Annual International Conference of the IEEE Engineering in Medicine and
    Biology Society (EMBC) 2016 Aug 16 (pp. 2721-2724). IEEE.

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param x0_index: index where flow crosses 0
    :param peep: positive end expiratory pressure
    :param tvi: TVi in L
    :param tau: The expiratory time constant

    :returns tuple: plateau pressure, compliance, resistance
    """
    # return a nan if we are not going to get a sensible response from the algo.
    if (tvi/tau) + flow[x0_index-1] - flow[0] < 0.0:
        return np.nan
    resistance = (pressure[x0_index-1] - pressure[0]) / ((tvi/tau) + flow[x0_index-1] - flow[0])
    elastance = resistance / tau
    compliance = 1 / elastance
    plat = tvi * elastance + peep
    return plat, compliance, resistance
