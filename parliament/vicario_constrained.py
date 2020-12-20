"""
vicario_constrained
~~~~~~~~~~~~~~~~~~~

Based on:

Vicario F, Albanese A, Karamolegkos N, Wang D, Seiver A, Chbat NW. Noninvasive
estimation of respiratory mechanics in spontaneously breathing ventilated patients:
a constrained optimization approach. IEEE Transactions on Biomedical Engineering.
2015 Aug 20;63(4):775-87.
"""
import numpy as np
from scipy.optimize import fmin_slsqp


def construct_U(flow, vols):
    """
    This is eq 8a in the paper, with some minor simplifications.

    This works by creating a 2 dim matrix (N+2,N+2) in size. provided we multiply this using
    x.dot(U).dot(x), the upper left resolves out to be: R^2*sum(V_dot^2) + 2REsum(V_dotV)
    + E^2sum(V^2). Then with the way we setup the matrix the rest of the multiplications resolve
    to be 2R*sum(V_dotP) + 2E*sum(VP) + sum(P^2).

    :param flow: numpy array of flow obs
    :param vols: numpy array of volume obs
    """
    U = np.zeros([len(flow)+2, len(flow)+2])
    U[0][0] = np.sum(flow ** 2)
    U[0][1] = np.sum(flow * vols)
    U[1][0] = np.sum(flow * vols)
    U[1][1] = np.sum(vols ** 2)
    for i in range(2, len(flow)+2):
        U[0][i] = flow[i-2]
        U[i][0] = flow[i-2]
        U[1][i] = vols[i-2]
        U[i][1] = vols[i-2]
        U[i][i] = 1

    return U


def construct_bound_mats_ineq(n_obs, x0_idx, m_idx):
    # the dims of x0_idx-1 just prevents us from performing unnecessary calculations when
    # performing the dot product a.dot(P_mus).
    #
    # We use +2 on the number of columns because we are also optimizing for elastance and resistance
    bm_ineq = np.zeros([x0_idx-1, n_obs+2])

    # 5a. this stands for P_mus(k+1) - P_mus(k) <= 0  k=1,2,...,m-1
    # which is converted to -P_mus(k+1) + P_mus(k) >= 0, because
    # fmin_slsq utilizes positive inequalities
    for i in range(0, m_idx):
        bm_ineq[i][i+2] = 1
        bm_ineq[i][i+3] = -1

    # 5b. this stands for P_mus(k+1) - P_mus(k) >= 0  k=m,m+1,...,q-1
    for i in range(m_idx, x0_idx-1):
        bm_ineq[i][i+2] = -1
        bm_ineq[i][i+3] = 1
    return bm_ineq


def construct_bound_mats(n_obs, x0_idx, m_idx):
    """
    Construct our constraining matrix A for our inequalities and
    equalities defined in 5a, 5b, and 5c.
    """
    bm_ineq = construct_bound_mats_ineq(n_obs, x0_idx, m_idx)
    bm_eq = np.zeros([n_obs-x0_idx-1, n_obs+2])
    # 3c. this stands for P_mus(k+1) - P_mus(k) = 0  k=q,q+1,...,N
    for i in range(x0_idx, n_obs-1):
        bm_eq[i-x0_idx][i+2] = -1
        bm_eq[i-x0_idx][i+3] = 1

    return bm_ineq, bm_eq


def objective(S, T, U, x):
    return S + T.dot(x) + x.dot(U).dot(x)


def _setup_constrained_optimization(flow, vols, pressure, x0, m_idx, r_max, r_min, e_max, e_min, p_min, p_max, initial_guess):

    if isinstance(initial_guess, type(None)):
        initial_guess = np.random.rand(len(flow)+2)
    # all code come from the optimization of eq 4. in the paper
    #
    # This is just sum(P_ao(k)^2), just call this S
    S = np.sum(pressure ** 2)
    # this is sum((RV(k)+EV(k) + P_mus(k)*P_ao(k)), call this T
    T = -2 * np.append([np.sum(flow * pressure), np.sum(vols * pressure)], pressure)
    U = construct_U(flow, vols)

    try:
        bm_ineq, bm_eq = construct_bound_mats(len(flow), x0, m_idx)
    except:
        import IPython; IPython.embed()

    bounds = [(r_min, r_max), (e_min, e_max)] + [(p_min, p_max) for i in range(len(flow))]
    obj = lambda x: objective(S, T, U, x)
    ieq_con = lambda p_mus: bm_ineq.dot(p_mus)
    eq_con = lambda p_mus: bm_eq.dot(p_mus)
    return obj, initial_guess, bounds, ieq_con, eq_con


def perform_constrained_optimization(flow, vols, pressure, x0, m_idx, r_max=100, r_min=0, e_max=100, e_min=0, p_min=-15, p_max=20, initial_guess=None):
    """
    :param flow: numpy array. Units denoted in L/s
    :param vols: numpy array. Units denoted in L
    :param pressure: numpy array. Units in cm H20.
    :param x0: index at which the breath begins expiratory phase
    :param m_idx: index to use for t_m. m itself is the location defined at which muscular effort of
                  breathing changes from decreasing to increasing over time. For fast algorithmic
                  purpose this is a guess. Unless you have more instrumentation or can make
                  observations to make this more than a guess. If you want to try to be clever about
                  it on VC you might be able to look at pressure curve deflection, or flow deflection
                  of PC/PS.

                  For slow algorithmic purposes, Vicario describes in his paper that you can use all
                  points m between 0 and x0. Then you can determine which value m gives you the
                  lowest residual and then use that value for you final result.
    """
    if m_idx >= len(flow)-1 or m_idx >= x0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    obj, initial_guess, bounds, ieq_con, eq_con = _setup_constrained_optimization(
        flow, vols, pressure, x0, m_idx, r_max, r_min, e_max, e_min, p_min, p_max, initial_guess
    )
    # this can get an error "x0 violates bound constraints if you have scipy 1.5.X. So as a result,
    # we've set scipy=1.4.1 in environment.yml
    result = fmin_slsqp(obj, initial_guess, f_eqcons=eq_con, f_ieqcons=ieq_con, bounds=bounds)
    R = result[0]
    E = result[1]
    P_mus = result[2:]
    pao_preds = R * flow + E * vols + P_mus
    residual = (1.0 / len(flow)) * np.sum((pao_preds - pressure) ** 2)
    return E, R, P_mus, pao_preds, residual


def optimize_insp_lim_only(flow, vols, pressure, x0, m_idx, r_max=100, r_min=0, e_max=100, e_min=0, p_min=-15, p_max=20, initial_guess=None):
    """
    Only run vicario constrained algo for the inspiratory lim. This has the advantage of removing
    any pesky problems if the patient is efforting on the exhale.

    :param flow: numpy array. Units denoted in L/s
    :param vols: numpy array. Units denoted in L
    :param pressure: numpy array. Units in cm H20.
    :param x0: index at which the breath begins expiratory phase
    :param m_idx: index to use for t_m. m itself is the location defined at which muscular effort of
                  breathing changes from decreasing to increasing over time. For algorithmic purposes
                  this is essentially a guess. Unless you have more instrumentation or can make
                  observations to make this more than a guess. If you want to try to be clever about
                  it on VC you might be able to look at pressure curve deflection, or flow deflection
                  of PC/PS.
    """
    if m_idx >= len(flow)-1 or m_idx >= x0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    flow, pressure, vols = flow[:x0], pressure[:x0], vols[:x0]
    obj, initial_guess, bounds, ieq_con, eq_con = _setup_constrained_optimization(
        flow, vols, pressure, x0, m_idx, r_max, r_min, e_max, e_min, p_min, p_max, initial_guess
    )
    result = fmin_slsqp(obj, initial_guess, f_ieqcons=ieq_con, bounds=bounds)
    R = result[0]
    E = result[1]
    P_mus = result[2:]
    pao_preds = R * flow + E * vols + P_mus
    residual = (1.0 / len(flow)) * np.sum((pao_preds - pressure) ** 2)
    return E, R, P_mus, pao_preds, residual
