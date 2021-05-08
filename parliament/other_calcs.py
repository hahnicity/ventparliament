import math

import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.stats import linregress


def _perform_least_squares(a, pressure, tvi, peep):
    """
    Helper function for performing least squares regression with single chamber model
    """
    least_square_result = np.linalg.lstsq(a, np.array(pressure))
    solution = least_square_result[0]
    elastance = solution[0]
    plat = tvi * elastance + peep
    return plat, 1 / elastance, solution[1], solution[2], least_square_result[1]


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


def al_rawas_expiratory_const(flow, x0_index, dt, linregress_tol):
    """
    Calculate the Al Rawas time constant (tau_E)

    Al-Rawas N, Banner MJ, Euliano NR, Tams CG, Brown J, Martin AD, Gabrielli A. Expiratory
    time constant for determinations of plateau pressure, respiratory system compliance,
    and total resistance. Critical care. 2013 Feb 1;17(1):R23.

    :param flow: array vals of flow measurements in L/s
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param linregress_tol: tolerance for the residual on linear regression
    """
    if len(flow[x0_index:]) <= int(.5 / dt):
        return np.nan
    start_idx = int(.1 / dt)
    end_idx = int(.5 / dt)
    # method deviates slightly from al-rawas in that the volumes we have are negative. This is
    # necessary to maintain a positive slope while we just focus exclusively on tidal volume
    # exhaled, and not the tidal volume exhaled plus any residual tvi.
    vols = np.abs(calc_volumes(flow[x0_index:x0_index+end_idx], dt))
    x = np.abs(flow[x0_index+start_idx:x0_index+end_idx])
    regress = linregress(x, vols[start_idx:])
    # XXX I'm still unsure about some questions I had between the al-rawas paper
    # and my own implementation. especially for the first linregress check
    if np.abs(regress.rvalue) < linregress_tol:
        return np.nan
    return linregress(np.arange(len(x)), vols[start_idx:]/x).slope


def al_rawas_calcs(flow, vols, pressure, x0_index, dt, pip, peep, tvi, compliance_idx, linregress_tol=.98, tau=None):
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
    :param compliance_idx: After computing the compliance curve, choose which index we want to use.
                           Allows choice of "max", "median", "mean', or a specific idx
    :param linregress_tol: tolerance for the residual on linear regression
    :param tau: input time const. optional, if not set defaults to al-rawas time const

    :returns tuple: tau, plat, compliance, resistance
    """
    if isinstance(compliance_idx, int) and len(flow[:x0_index-1]) <= compliance_idx:
        return np.nan, np.nan, np.nan, np.nan
    if tau is None:
        tau = al_rawas_expiratory_const(flow, x0_index, dt, linregress_tol)
    compliance_curve = [
        (vol + tau * flow[i]) / (max(pressure[:i+1]) - peep)
        for i, vol in enumerate(vols[:x0_index-1])
    ]
    if compliance_idx == 'max':
        compliance = max(compliance_curve)
    elif compliance_idx == 'mean':
        compliance = np.mean(compliance_curve)
    elif compliance_idx == 'median':
        compliance = np.median(compliance_curve)
    elif isinstance(compliance_idx, int):
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
    if f_min <= 0 or tve == 0 or e_time == 0:
        return np.nan

    bru = tve / f_min
    for i in range(iters):
        # If you graph out the derivative of the function, it shows there is never
        # a convergence when bru>0 and e_time>0, unless of course bru = inf, then it will converge,
        # but by then your tc is infinity... So you need to limit the number of times you
        # iterate on this function.
        bru = bru * (1 / (1 - math.exp(-e_time/bru)))
    return bru


def ft_inspiratory_least_squares(flow, vols, pressure, x0_index, peep, dt, tvi):
    """
    Volume targeted least squares. Perform least squares approximation of R and E the way
    that Kannangara does it in his paper. Only use the inspiratory limb

    Kannangara DO, Newberry F, Howe S, Major V, Redmond D, Szlavecs A,
    Chiew YS, Pretty C, Benyó B, Shaw GM, Chase JG. Estimating the true
    respiratory mechanics during asynchronous pressure controlled ventilation.
    Biomedical Signal Processing and Control. 2016 Sep 1;30:70-8.

    V_dot = (pressure - peep) / R - E*V / R

    :param flow: array vals of flow measurements in L/s
    :param vols: volumes of air inspired in L
    :param pressure: array vals of pressure obs
    :param x0_index: Index where flow crosses to expiratory phase
    :param peep: positive end expiratory pressure
    :param dt: time delta between obs

    :returns tuple: plat, compliance, resistance, None, residual

    Returns None in 4th arg to maintain compatibility with other least squares methods
    """
    end_idx = x0_index if x0_index <= len(flow) else len(flow)
    a = np.array([pressure[:end_idx] - peep, vols[:end_idx]]).transpose()
    least_square_result = np.linalg.lstsq(a, flow[:end_idx])
    solution = least_square_result[0]
    resistance = 1 / solution[0]
    elastance = -1 * solution[1] * resistance
    try:
        residual = least_square_result[1][0]
    except IndexError:
        residual = np.nan
    plat = tvi * elastance + peep
    return plat, 1 / elastance, resistance, None, residual


def fuzzy_clustering_time_const(flow, vols, x0_index, min_n_obs, alpha, gk_cls):
    """
    Get exp. time const using fuzzy clustering and least squares

    Babuška R, Alic L, Lourens MS, Verbraak AF, Bogaard J. Estimation of respiratory parameters
    via fuzzy clustering. Artificial Intelligence in Medicine. 2001 Jan 1;21(1-3):91-105.

    :param flow: array vals of flow measurements in L/s
    :param vols: array of volumes in lung for the breath
    :param x0_index: index where flow crosses 0
    :param min_n_obs: minimum number obs we need in expiratory flow array to analyze breath
    :param alpha: The alpha param used to make an alpha-cut. In the paper this is 0.8
    :param gk_cls: Trained GK algo object.
    """
    n_clust = gk_cls.u.shape[0]
    # if flow never crosses 0 then quit
    if x0_index >= len(flow)-1:
        return [np.nan] * n_clust

    # the paper never states this, but our impl takes flow from min flow to end. Otherwise
    # the time const may be lengthened.
    min_f_idx = np.argmin(flow)
    if len(flow[min_f_idx:]) < min_n_obs:
        return [np.nan] * n_clust

    z = np.array([vols[min_f_idx:], flow[min_f_idx:]]).T
    u = gk_cls.predict_proba(z)
    tau_clust = []
    for c_idx in range(u.shape[0]):
        clust_members = z[u[c_idx] >= alpha]
        # the eq in the paper is V + \tau*V_dot = 0
        # so we shape it to be
        # \tau*V_dot = -V
        target = clust_members[:, 0]
        a = np.expand_dims(clust_members[:, 1], axis=0).T
        least_square_result = np.linalg.lstsq(a, -target)
        tau_clust.append(least_square_result[0][0])
    return tau_clust


def howe_expiratory_least_squares(flow, vols, pressure, x0_index, dt, peep, tvi):
    """
    Calculate compliance, resistance, and K via standard single chamber
    model equation. Only looks at expiratory section of breath and perform
    additional modifications suggested by Howe et al. 2020

    Howe SL, Chase JG, Redmond DP, Morton SE, Kim KT, Pretty C, Shaw GM, Tawhai MH, Desaive T.
    Inspiratory respiratory mechanics estimation by using expiratory data for reverse-triggered
    breathing cycles. Computer methods and programs in biomedicine. 2020 Apr 1;186:105184.

    This algorithm uses pressure-targeted expiratory least squares model. Model is
    changed though by ensuring both pressure and volume values are initially set to
    0.

    :param flow: array vals of flow measurements in L/s
    :param vols: technically an unused param here. and exists for compatibility
                 across least squares methods you can set to None if you want.
    :param pressure: array vals of pressure obs
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param peep: positive end expiratory pressure
    :param tvi: TVi in L

    :returns tuple: plateau pressure, compliance, resistance, peep, residual
    """
    # there was no identifiable expiratory location
    if x0_index >= len(flow)-1:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    start_idx = x0_index - 1
    vols = calc_volumes(flow[start_idx:], dt)
    f_new = flow[start_idx:]
    p_new = np.array(pressure[start_idx:])
    p_new = p_new - p_new[0]
    a = np.array([vols, f_new]).transpose()
    least_square_result = np.linalg.lstsq(a, np.array(p_new)-peep)
    solution = least_square_result[0]
    elastance = solution[0]
    plat = tvi * elastance + peep
    return plat, 1 / elastance, solution[1], peep, least_square_result[1]


def ikeda_time_const(flow, tvi, tve, dt):
    """
    Calculate tau_e based on Ikeda's 2019 paper:

    Ikeda T, Yamauchi Y, Uchida K, Oba K, Nagase T, Yamada Y. Reference value for expiratory
    time constant calculated from the maximal expiratory flow-volume curve.
    BMC pulmonary medicine. 2019 Dec;19(1):1-9.

    :param flow: numpy array vals of flow measurements in L/s
    :param tvi: inspiratory tidal volume in L
    :param tve: expiratory tidal volume in L
    :param dt: time delta between obs
    """
    vols = calc_volumes(flow, dt)
    mef50 = None
    mef25 = None
    # just uses a derivative of lourens tc logic
    for idx, v in enumerate(vols):
        # Note we dont use the idx+1 indexing here because we assume
        # that the volume expired was passed in between the last obs and
        # the current obs
        if v >= tvi * .5 and mef50 is None:
            # use +1 because we start from index 1 on vols
            mef50 = flow[idx]
        # mef25 is a bit misleadingly labeled, but it really corresponds with 75% volume
        if v >= tvi * .75 and mef25 is None:
            mef25 = flow[idx]
            # if we've made it here then break because mef50 will already be done
            break

    if mef50 == mef25:
        # breath is too short to get a reasonable result
        return np.nan

    return 0.25 * tve /  (mef50 - mef25)


def lourens_time_const(flow, tve, x0_index, dt, percentage_target):
    """
    Calculate Lourens time constant for patients. Although Lourens used 4 fixed percentage
    targets in their paper, you are free to set whatever target you wish.

    Lourens M, van den Berg B, Aerts JG, Verbraak A, Hoogsteden H, Bogaard J. Expiratory
    time constants in mechanically ventilated patients with and without COPD. Intensive
    care medicine. 2000 Nov 1;26(11):1612-8.

    :param flow: numpy array vals of flow measurements in L/s
    :param tve: expiratory tidal volume in L
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    :param percentage_target: specific percentage of exhaled tidal volume you want to target for the time constant. Can be between 1 and 100

    :returns tuple: RCfvN (where N is your percentage_target)
    """
    # there was no identifiable expiratory location
    if x0_index >= len(flow):
        return (np.nan, np.nan, np.nan, np.nan)

    if not 1<=percentage_target<=100:
        raise Exception('Lourens time constant percentage_target should be set between 1 and 100!')

    flow = np.abs(flow)
    vols = calc_volumes(flow[x0_index-1:], dt)

    volume_target = tve * (percentage_target/100)
    volume_idx = None
    for idx, v in enumerate(vols):
        if v >= volume_target:
            volume_idx = idx
            break

    return (tve*(percentage_target/100)) / (flow[volume_idx]-flow[-1]) if volume_idx is not None else np.nan


def pt_expiratory_least_squares(flow, vols, pressure, x0_index, dt, peep, tvi):
    """
    Pressure targeted least squares. Calculate compliance and resistance via standard
    single chamber model equation. Only looks at expiratory part of breath:

    This method was discussed, but not directly used in:

    Van Drunen EJ, Chiew YS, Chase JG, Shaw GM, Lambermont B, Janssen N, Damanhuri NS, Desaive T.
    Expiratory model-based method to monitor ARDS disease state. Biomedical engineering online.
    2013 Dec 1;12(1):57.

    Howe later mentions in 2020 that it's not a very good method, so we implement to show
    improvements of Howe upon this baseline.

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
    if x0_index >= len(flow)-1:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    vols = vols[x0_index-1:]
    flow = flow[x0_index-1:]
    pressure = pressure[x0_index-1:]
    a = np.array([vols, flow, [1]*len(flow)]).transpose()
    return _perform_least_squares(a, pressure, tvi, peep)


def pt_inspiratory_least_squares(flow, vols, pressure, x0_index, dt, peep, tvi):
    """
    Pressure targeted least squares. Calculate compliance, resistance, and K via standard
    single chamber model equation. Only looks at inspiratory part of breath:

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
    return _perform_least_squares(a, pressure[:end_idx], tvi, peep)


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
    if (tvi/tau) + flow[x0_index-1] - flow[0] <= 0.0:
        return np.nan, np.nan, np.nan

    resistance = (pressure[x0_index-1] - pressure[0]) / ((tvi/tau) + flow[x0_index-1] - flow[0])
    elastance = resistance / tau
    compliance = 1 / elastance
    plat = tvi * elastance + peep
    return plat, compliance, resistance


def vicario_nieap_tau(flow, pressure, vols, x0_index, delta_tol=.1):
    """
    Perform method of calculating tau as explained in Vicario's 2016
    paper "Noninvasive estimation of alveolar pressure."

    Vicario F, Buizza R, Truschel WA, Chbat NW. Noninvasive estimation
    of alveolar pressure. In2016 38th Annual International Conference
    of the IEEE Engineering in Medicine and Biology Society (EMBC) 2016
    Aug 16 (pp. 2721-2724). IEEE.

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param vols: breath volume in L per observation of the flow array
    :param x0_index: index where flow crosses 0
    :param delta_tol: fraction tolerance to approach PEEP for delta.
                      (expressed in eq. 4 in paper)
    """
    # transient abnormality. just return NaN.
    if x0_index >= len(flow)-1 or x0_index<5:
        return np.nan

    # calculate a per-breath peep because peep can fluctuate based on
    # efforting and other factors like PRVC. Since the delta factor is
    # so clearly based on getting as close to PEEP as possible then doing
    # this on a breath-by-breath basis is probably best
    peep = np.mean(pressure[-5:])

    # find \delta where \delta is the "time the ventilator requires
    # to reduce the pressure down to set PEEP." This phrasing is
    # nebulous so what I am doing is declaring a tolerance factor to
    # where we are close to PEEP. This maintains the intention of the
    # paper's method.
    delta_thresh = (peep * delta_tol) + peep
    delta_idx = np.argmin(np.logical_not((pressure[x0_index:]<delta_thresh))) + x0_index

    a = np.array([[1]*len(flow[delta_idx:])]).transpose()
    target = -np.array(vols[delta_idx:]) / (flow[delta_idx:]-flow[0])
    result = np.linalg.lstsq(a, target)
    tau = result[0][0]
    return tau


def wiriyaporn_time_const_exp(flow, x0_index, dt):
    """
    Calculate the expiratory time constant based on Wiriyaporn's 2016 paper:

    Wiriyaporn D, Wang L, Aboussouan LS. Expiratory time constant and sleep apnea severity
    in the overlap syndrome. Journal of Clinical Sleep Medicine. 2016 Mar 15;12(3):327-32.

    Solve using scipy.optimize.curve_fit.

    :param flow: array vals of flow measurements in L/s
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    """
    min_flow_idx = np.argmin(flow)
    # second cond means exp flow <= 0.22 seconds
    if x0_index >= len(flow)-1 or len(flow[min_flow_idx:]) <= int(0.22 / dt):
        return np.nan

    exp_flow = np.abs(flow[min_flow_idx:])
    t = np.arange(dt, dt*len(exp_flow), .2)
    ydata = np.abs([exp_flow[int(v/dt)] for v in t])
    f = lambda x, tau: exp_flow[0]*np.exp(-x/tau)
    result = curve_fit(f, t, ydata)
    tau = result[0][0]
    pred = f(t, tau)
    r2 = 1 - np.sum((ydata-pred)**2) / np.sum((ydata - np.mean(ydata))**2)
    if r2 < 0.95:
        return np.nan
    return tau


def wiriyaporn_time_const_linear(flow, x0_index, dt):
    """
    Calculate the expiratory time constant based on Wiriyaporn's 2016 paper:

    Wiriyaporn D, Wang L, Aboussouan LS. Expiratory time constant and sleep apnea severity
    in the overlap syndrome. Journal of Clinical Sleep Medicine. 2016 Mar 15;12(3):327-32.

    Here we solve their exponential fit eq by linearizing the equation and then solving with
    least squares fit. Just wanted to do this to provide alternative method in case exponential
    solver fails

    :param flow: array vals of flow measurements in L/s
    :param x0_index: index where flow crosses 0
    :param dt: time delta between obs
    """
    min_flow_idx = np.argmin(flow)
    # second cond means exp flow <= 0.22 seconds
    if x0_index >= len(flow)-1 or len(flow[min_flow_idx:]) <= int(0.22 / dt):
        return np.nan

    min_flow_idx = np.argmin(flow)
    exp_flow = np.abs(flow[min_flow_idx:])
    t = np.arange(dt, dt*len(exp_flow), .2)
    # just linearize the eq instead of fitting an exponential decay
    target = [-v / np.log(exp_flow[int(v/dt)]/exp_flow[0]) for v in t]
    a = np.array([[1] * len(target)]).T
    result = np.linalg.lstsq(a, target)

    # now figure out the residual using the original eq.
    tau = result[0][0]
    f = lambda x, tau: exp_flow[0]*np.exp(-x/tau)
    ydata = np.abs([exp_flow[int(v/dt)] for v in t])
    r2 = 1 - np.sum((ydata-f(t, tau))**2)/np.sum((ydata-np.mean(ydata))**2)
    if r2 > 0.05:
        return np.nan
    return tau
