"""
pressure_ctrl_correction
~~~~~~~~~~~~~~~~~~~~~~~~

Details algorithm that Kannangara et al. 2016 developed for correcting for
effort in PC waveforms.
"""
from copy import copy
import math

import numpy as np
from scipy.integrate import simps
from scipy.ndimage.filters import gaussian_filter1d


def shear_transform(pressure, flow, x0_index, dt=0.02):
    """
    Follows shear transform discussed in Stevenson et al. 2012.

    The goal here is to find the left and right shear locations for the pressure curve
    """
    # flow min idx is included in the shear calculation
    max_p_idx = np.argmax(pressure)
    min_f_idx = np.argmin(flow)
    m1 = (pressure[0] - pressure[max_p_idx]) / (max_p_idx * dt)
    m2 = (pressure[max_p_idx] - pressure[min_f_idx]) / ((min_f_idx - max_p_idx) * dt)
    c1 = 0
    c2 = -m2 * (max_p_idx * dt)
    t1 = np.arange(max_p_idx+1) * dt
    t2 = np.arange(max_p_idx, min_f_idx+1) * dt
    # A first approximation of the shear transform is found by the maximum of
    # the shear transform between the first data point and the point of maximum
    # pressure
    shear_left = np.argmax(pressure[:max_p_idx+1] + m1 * t1 + c1)
    # The second shoulder is found by taking the maximum of the shear transform
    # between the point of maximum pressure and the point of minimum flow
    #
    # A weird thing that Kannangara does, that they don't talk about in paper, is
    # that they say that x0 is the right shoulder, but then they subtract 3 off the
    # index. I dunno if I should do that here or not.
    try:
        shear_right = max_p_idx + np.argmax(pressure[max_p_idx:min_f_idx+1] + m2 * t2 + c2)
    except ValueError:
        return None, None
    quarter_way = shear_left + int((x0_index - shear_left) / 4)
    m3 = (pressure[0] - pressure[quarter_way]) / (quarter_way * dt)
    c3 = -m3 * (quarter_way * dt)
    t3 = np.arange(quarter_way+1) * dt
    shear_left = np.argmax(pressure[:quarter_way+1] + m3 * t3 + c3)
    return shear_left, shear_right


def get_predicted_flow_waveform(flow, pressure, peep, vols, dt=0.02):
    comp, resist, residual = least_squares_flow_fit(flow, vols, pressure, peep, dt=dt)
    v_dot_predicted = (pressure - peep) / resist - (1/comp) * vols / resist
    return v_dot_predicted, comp, resist, residual


def least_squares_flow_fit(flow, vols, pressure, peep, dt=0.02):
    """
    V_dot = (pressure - peep) / R - E*V / R

    :param flow: array vals of flow measurements in L/s
    :param vols: volumes of air inspired in L
    :param pressure: array vals of pressure obs
    :param peep: positive end expiratory pressure
    :param dt: time delta between obs

    :returns tuple: compliance, resistance, residual
    """
    a = np.array([pressure - peep, vols]).transpose()
    least_square_result = np.linalg.lstsq(a, flow)
    solution = least_square_result[0]
    resistance = 1 / solution[0]
    elastance = -1 * solution[1] * resistance
    try:
        residual = least_square_result[1][0]
    except IndexError:
        residual = np.nan
    return 1 / elastance, resistance, residual


def perform_algo(flow, pressure, x0, peep, auc_thresh, debug=False):
    """
    Perform the Kannangara algorithm to improve fit of least squares algo in pressure
    modes for asynchronously breathing patients.

    :param flow: flow in L/min, the normal way we get it from the vent
    :param vols: calculated volumes inspired over time
    :param pressure: pressure recordings
    :param x0: the point at which flow crosses 0
    :param peep: the PEEP set for the breath
    :param auc_thresh: The AUC threshold to determine whether the breath is async or not.
                       Anything under a certain thresh is believed to be synchronous so
                       we don't bother analyzing these breaths
    :param debug: Set to true if you want to plot every step of this algo

    :returns tuple: compliance, resistance, residual, response code

    I don't know what a good AUC threshold is yet, but it seems that something
    around .1 might be useful.

    Response codes:
        0: successful run
        1: auc threshold not breached
        2: shear transform error
        3: step 7ab no intercept
        4: step 7c no intercept
        5: unknown error
        6: step 7ab bad intercept
    """
    dt = 0.02
    # step 1, perform light noise filtering
    flow = gaussian_filter1d(np.array(flow) / 60.0, 1)
    pressure = gaussian_filter1d(np.array(pressure), 1)

    # step 2, find shear transforms
    shear_left, shear_right = shear_transform(pressure, flow, x0)
    if not shear_left and not shear_right:
        return np.nan, np.nan, np.nan, 2
    elif shear_right - shear_left <= 0:
        return np.nan, np.nan, np.nan, 2

    # step 3 perform flow reconstruction
    pred, comp, res, resid = get_predicted_flow_waveform(flow[shear_left:shear_right+1], pressure[shear_left:shear_right+1], peep, vols[shear_left:shear_right+1])
    pred = np.append([np.nan] * shear_left, pred)
    diff_pred_real = 0
    for i in range(shear_left+2, shear_right):
        pred_auc = simps(pred[i-2:i], dx=dt)
        real_auc = simps(flow[i-2:i], dx=dt)
        diff_pred_real += abs(pred_auc - real_auc)

    if diff_pred_real < auc_thresh:
        return comp, res, resid, 1

    if debug:
        import matplotlib.pyplot as plt
        plt.plot([shear_left], [flow[shear_left]], marker='x', markersize=7, label='shear left')
        plt.plot([shear_right], [flow[shear_right]], marker='x', markersize=7, label='shear right')
        plt.plot(pred)
        plt.plot(flow)
        plt.legend()
        plt.title('flow and correction')
        plt.show()

    # step 4 identify patient effort.
    flow_grad = np.array([0] + [(flow[i] - flow[i-1]) / dt*2 for i in range(1, shear_right+1)])
    intercept = None
    for i in range(shear_left, shear_right+1):
        f = flow[i]
        p = pred[i]
        # The positive gradient requirement for the intercept is in the paper
        if ((flow[i] >= pred[i] and flow[i-1] < pred[i-1]) or (flow[i] <= pred[i] and flow[i-1] > pred[i-1])) and flow_grad[i] > 0:
            # which flow obs are closer to each other?
            if abs(flow[i] - pred[i]) > abs(flow[i-1] - pred[i-1]):
                intercept = i - 1
            else:
                intercept = i
            break

    if intercept and debug:
        plt.plot(pred)
        plt.plot([shear_left], [flow[shear_left]], marker='x', markersize=7, label='shear left')
        plt.plot([shear_right], [flow[shear_right]], marker='x', markersize=7, label='shear right')
        plt.plot([intercept], [pred[intercept]], marker='x', markersize=10, label='pred intercept')
        plt.plot([intercept], [flow[intercept]], marker='x', markersize=10, label='flow intercept')
        plt.plot(flow)
        plt.title('flow and correction with intercept')
        plt.legend()
        plt.show()
        print("shear left: {}, intercept: {}".format(shear_left, intercept))

    if intercept and intercept > shear_left + 8:

        # step 5 reduction of patient induced effort
        first_recon_pt = shear_left + 2
        min_up_to_intercept = np.argmin(flow[shear_left:intercept+1]) + shear_left
        second_recon_pt = int(math.floor((intercept - min_up_to_intercept) / 2.0)) + min_up_to_intercept

        if debug:
            plt.plot(pred)
            plt.plot([shear_left], [flow[shear_left]], marker='x', markersize=7, label='shear left')
            plt.plot([first_recon_pt], [pred[first_recon_pt]], marker='x', markersize=7, label='first recon pt')
            plt.plot([second_recon_pt], [flow[second_recon_pt]], marker='x', markersize=7, label='second recon pt')
            plt.plot(flow)
            plt.legend()
            plt.title('flow and correction with intercept')
            plt.show()

        m = (flow[second_recon_pt] - flow[first_recon_pt]) / ((second_recon_pt-first_recon_pt)*dt)
        x = np.arange(shear_right-first_recon_pt) * 0.02
        flow_recon = np.append(flow[:first_recon_pt], m*x + flow[first_recon_pt])
        flow_recon = np.append(flow_recon, flow[len(flow_recon):])

        if debug:
            plt.plot([first_recon_pt], [pred[first_recon_pt]], marker='x', markersize=7)
            plt.plot([second_recon_pt], [flow[second_recon_pt]], marker='x', markersize=7)
            plt.plot(flow, label='original flow')
            plt.plot(flow_recon, label='reconstructed flow')
            plt.title('flow and correction with intercept')
            plt.legend()
            plt.show()

        # step 6 single compartment refitting
        vols_recon = np.array([0] + [simps(flow_recon[:i], dx=dt) for i in range(2, len(flow_recon)+1)])
        comp, res, resid = least_squares_flow_fit(flow_recon, vols_recon, pressure, peep)
        return comp, res, resid, 0

    # Steps 7a-b breaths with early asynchrony
    #
    # This logic follows page 4 where it mentions "therefore if the first asynchronous crossing
    # detected is within 8 data points of the first fit flow point..."
    elif intercept and intercept <= shear_left + 8:
        # perform an iterative reconstruction of flow
        for i in range(3):
            intercepts = []
            for i in range(intercept+1, len(pred)):
                f = flow[i]
                p = pred[i]
                if (flow[i] >= pred[i] and flow[i-1] < pred[i-1]) or (flow[i] <= pred[i] and flow[i-1] > pred[i-1]):
                    if abs(flow[i] - pred[i]) > abs(flow[i-1] - pred[i-1]):
                        intercepts.append(i - 1)
                    else:
                        intercepts.append(i)

            if len(intercepts) > 0:
                new_intercept = intercepts[-1]
            else:
                return np.nan, np.nan, np.nan, 3

            m = (flow[new_intercept]-flow[shear_left]) / ((new_intercept-shear_left)*dt)
            x = np.arange(new_intercept-shear_left) * 0.02
            flow_recon = np.append(flow[:shear_left], m*x + flow[shear_left])
            flow_recon = np.append(flow_recon, flow[len(flow_recon):])

            if debug:
                plt.plot([new_intercept], [flow[new_intercept]], marker='x', markersize=7, label='new intercept loc')
                plt.plot(pred, label='predicted')
                plt.plot(flow, label='original flow')
                plt.plot(flow_recon, label='reconstructed flow')
                plt.title('step 7a-b special case')
                plt.legend()
                plt.show()

            vols_recon = np.array([0] + [simps(flow_recon[:i], dx=dt) for i in range(2, len(flow_recon)+1)])
            if len(flow_recon[shear_left:new_intercept+1]) < 3:
                return np.nan, np.nan, np.nan, 6
            pred, comp, res, resid = get_predicted_flow_waveform(flow_recon[shear_left:shear_right+1], pressure[shear_left:shear_right+1], peep, vols_recon[shear_left:shear_right+1])
            pred = np.append([np.nan] * shear_left, pred)
            flow = flow_recon
        comp, res, resid = least_squares_flow_fit(flow_recon, vols_recon, pressure, peep)
        return comp, res, resid, 0

    # Step 7c breaths where linear extrapolation not possible
    elif not intercept or (pred < 0).any():
        intercepts = []
        for i in range(shear_left, shear_right+1):
            f = flow[i]
            p = pred[i]
            if (flow[i] >= pred[i] and flow[i-1] < pred[i-1]) or (flow[i] <= pred[i] and flow[i-1] > pred[i-1]):
            # We want the intercept where flow goes below the prediction
                if abs(flow[i] - pred[i]) > abs(flow[i-1] - pred[i-1]):
                    intercepts.append(i - 1)
                else:
                    intercepts.append(i)

        if not intercepts:
            return np.nan, np.nan, np.nan, 4

        if len(intercepts) == 1:
            intercept = intercepts[0]
        else:
            intercept = intercepts[1]

        flow_recon = copy(flow)
        for i in range(intercept, len(pred)):
            flow_recon[i] = pred[i]

        if debug:
            plt.plot([shear_left], [flow[shear_left]], marker='x', markersize=7, label='shear left')
            plt.plot([intercept], [flow[intercept]], marker='x', markersize=7, label='intercept')
            plt.plot(flow, label='original flow')
            plt.plot(flow_recon, label='reconstructed flow')
            plt.title('step 7c special case')
            plt.legend()
            plt.show()

        vols_recon = np.array([0] + [simps(flow_recon[:i], dx=dt) for i in range(2, len(flow_recon)+1)])
        comp, res, resid = least_squares_flow_fit(flow_recon, vols_recon, pressure[:len(flow_recon)], peep)
        return comp, res, resid, 0

    # XXX This corner case has happened in the following scenarios
    #
    #   1. premature exhalation, but predicted line does not go negative
    #   2. auc is set too low and the breath wasn't asynchronous
    return np.nan, np.nan, np.nan, 5
