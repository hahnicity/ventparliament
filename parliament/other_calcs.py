import math

import numpy as np
from scipy.integrate import simps
from scipy.stats import linregress


def calc_volumes(flow, dt):
    """
    Calculate volume for flow wave using simpsons rule. Here we use a streaming version of simpsons
    that is faster to compute than just calling the simps method for each observation in the
    flow array

    :param flow: array vals of flow measurements in L/s
    :param dt: time delta between obs
    """
    vols = np.zeros(len(flow))
    # this is a pretty gross logic block, but it works for now
    if len(flow) >= 2:
        vols[1] = (flow[0] + flow[1]) / 2 * dt
    if len(flow) >= 3:
        vols[2] = (dt / 3) * (flow[0] + 4*flow[1] + flow[2])
    if len(flow) >= 4:
        evens = dt/3 * (flow[1] + 4*flow[2] + flow[3])
        vols[3] = ((vols[2] + (flow[2] + flow[3])/2*dt) + (evens + vols[1])) / 2

    # fill in odd vals first
    for i in range(4, len(flow), 2):
        vols[i] = (vols[i-2]) + (flow[i-2] + 4*flow[i-1] + flow[i]) * (dt/3)

    # then fill evens.
    for i in range(5, len(flow), 2):
        # this would be the first trap plus n-1 obs of simpsons
        evens = evens + (flow[i-2] + 4*flow[i-1] + flow[i]) * (dt/3)
        first = evens + (flow[0]+flow[1])/2*dt
        # this is the last trap plus first n-1 obs of simpsons
        second = vols[i-1] + (flow[i-1] + flow[i]) / 2 * dt
        vols[i] = (first + second) / 2
    return vols


def al_rawas_expiratory_const(flow, vols, x0_index, dt, tvi, linregress_tol):
    """
    Calculate the Al Rawas time constant (tau_E)

    Al-Rawas N, Banner MJ, Euliano NR, Tams CG, Brown J, Martin AD, Gabrielli A. Expiratory
    time constant for determinations of plateau pressure, respiratory system compliance,
    and total resistance. Critical care. 2013 Feb 1;17(1):R23.

    :param flow: array vals of flow measurements in L/s
    :param vols: breath volume in L per observation of the flow array
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
    # XXX I dunno why we're cutting off the 0 here, but this is what I wrote originally
    #
    # XXX this is weird...originally we cut off volumes at the end index, but then we
    # cut them off again at the end index. I dont get this.
    vols = tvi + vols[:end_idx]
    x = [abs(val) for val in flow[start_idx:end_idx]]
    regress = linregress(x, vols[start_idx-1:end_idx-1])
    if regress.rvalue < linregress_tol:
        return np.nan
    return regress.slope


def al_rawas_calcs(flow, vols, pressure, x0_index, dt, pip, peep, tvi, compliance_idx, linregress_tol=.98):
    """
    Calculate compliance using al-rawas methodology

    Al-Rawas N, Banner MJ, Euliano NR, Tams CG, Brown J, Martin AD, Gabrielli A. Expiratory
    time constant for determinations of plateau pressure, respiratory system compliance,
    and total resistance. Critical care. 2013 Feb 1;17(1):R23.

    :param flow: array vals of flow measurements in L/s
    :param vols: breath volume in L per observation of the flow array
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
    tau = al_rawas_expiratory_const(flow, vols, x0_index, dt, tvi, linregress_tol)
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
    expiratory time constant tau.

    Brunner JX, Laubscher TP, Banner MJ, Iotti G, Braschi A. Simple method to measure total
    expiratory time constant based on the passive expiratory flow-volume curve. Critical
    care medicine. 1995 Jun 1;23(6):1117-22.

    :param tve: expiratory volume in liters (L)
    :param e_time: expiratory time in seconds
    :param f_min: abs value of maximum expiratory flow rate in L/s
    :param iters: number iterations to perform on the algorithm. We suggest sticking to 2 iters
    """
    if f_min == 0 or tve == 0 or e_time == 0:
        return np.nan

    bru = tve / f_min
    for i in range(iters):
        # If you graph out the derivative of the function, it shows there is never
        # a convergence when bru>0 and e_time>0, unless of course bru = inf, then it will converge,
        # but by then your tc is infinity... So you need to limit the number of times you
        # iterate on this function.
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


def expiratory_least_squares(flow, vols, pressure, x0_index, dt, peep, tvi):
    """
    Calculate compliance, resistance, and K via standard single chamber
    model equation. Only looks at expiratory part of breath:

    This method was exemplified by:

    Van Drunen EJ, Chiew YS, Chase JG, Shaw GM, Lambermont B, Janssen N, Damanhuri NS, Desaive T.
    Expiratory model-based method to monitor ARDS disease state. Biomedical engineering online.
    2013 Dec 1;12(1):57.

    :param flow: array vals of flow measurements in L/s
    :param vols: breath volume in L per observation of the flow array
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

    vols = vols[x0_index-1:]
    flow = flow[x0_index-1:]
    pressure = pressure[x0_index-1:]
    a = np.array([vols, flow, [1]*len(flow)]).transpose()
    return perform_least_squares(a, pressure, tvi, peep)


def howe_expiratory_least_squares(flow, vols, pressure, x0_index, dt, peep, tvi):
    """
    Calculate compliance, resistance, and K via standard single chamber
    model equation. Only looks at expiratory section of breath and perform
    additional modifications suggested by Howe et al. 2020

    Howe SL, Chase JG, Redmond DP, Morton SE, Kim KT, Pretty C, Shaw GM, Tawhai MH, Desaive T.
    Inspiratory respiratory mechanics estimation by using expiratory data for reverse-triggered
    breathing cycles. Computer methods and programs in biomedicine. 2020 Apr 1;186:105184.

    :param flow: array vals of flow measurements in L/s
    :param vols: technically an unused param here. and exists for compatibility
                 across least squares methods you can set to None if you want.
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
    vols = calc_volumes(flow[start_idx:], dt)
    flow = flow[start_idx:]
    pressure = np.array(pressure[start_idx:])
    pressure = pressure - pressure[0]
    a = np.array([vols, flow, [1]*len(flow)]).transpose()
    return perform_least_squares(a, pressure, tvi, peep)


def inspiratory_least_squares(flow, vols, pressure, x0_index, dt, peep, tvi):
    """
    Calculate compliance, resistance, and K via standard single chamber
    model equation. Only looks at inspiratory part of breath:

    P_vent = V(t)E + V_dot(t)R + K

    where K is some constant that is defined by PEEP + PEEPi + P_0

    This method is will work in more situations than other methods will,
    but it's also going to be the least accurate, especially when a patient
    is exerting force of their muscles

    :param flow: array vals of flow measurements in L/s
    :param vols: breath volume in L per observation of the flow array
    :param pressure: array vals of pressure obs
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param peep: positive end expiratory pressure
    :param tvi: TVi in L

    :returns tuple: plateau pressure, compliance, resistance, K, residual
    """
    end_idx = x0_index if x0_index <= len(flow) else len(flow)
    a = np.array([vols[:end_idx], flow[:end_idx], [1]*end_idx]).transpose()
    return perform_least_squares(a, pressure[:end_idx], tvi, peep)


def lourens_time_const(flow, vols, tve, x0_index, dt):
    """
    Calculate Lourens time constant for patients. Lourens performed her
    calculations on paralyzed COPD patients. So it might be helpful in
    this case.

    Lourens M, van den Berg B, Aerts JG, Verbraak A, Hoogsteden H, Bogaard J. Expiratory
    time constants in mechanically ventilated patients with and without COPD. Intensive
    care medicine. 2000 Nov 1;26(11):1612-8.

    :param flow: numpy array vals of flow measurements in L/s
    :param vols: breath volume in L per observation of the flow array
    :param tve: expiratory tidal volume in L
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs

    :returns tuple: RCfv25, RCFv50, RCfv75, RCFv100
    """
    # there was no identifiable expiratory location
    if x0_index >= len(flow):
        return (np.nan, np.nan, np.nan, np.nan)

    min_idx = np.argmin(flow)
    flow = np.abs(flow)

    v25, v50, v75 = tve/4.0, tve/2.0, tve*(3/4.0)
    v25_idx, v50_idx, v75_idx, v100_idx = None, None, None, None
    for idx, v in enumerate(vols[1:]):
        if v >= v25 and vols[idx-1] < v25:
            v25_idx = idx
        elif v >= v50 and vols[idx-1] < v50:
            v50_idx = idx
        elif v >= v75 and vols[idx-1] < v75:
            v75_idx = idx
        elif v >= tve and vols[idx-1] < tve:
            v100_idx = idx

    rcfv25 = (tve*.25) / (flow[v25_idx]-flow[-1])
    rcfv50 = (tve*.5) / (flow[v50_idx]-flow[-1])
    rcfv75 = (tve*.75) / (flow[v75_idx]-flow[-1])
    rcfv100 = tve / (flow[v100_idx]-flow[-1]) if v100_idx is not None else np.nan
    return rcfv25, rcfv50, rcfv75, rcfv100


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
