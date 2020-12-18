"""
iipr
~~~~

From Newberry's paper "Iterative Interpolative Pressure Reconstruction for Improved
Respiratory Mechanics Estimation During Asynchronous Volume Controlled Ventilation"
"""
from copy import copy

import numpy as np
from scipy.integrate import simps

from parliament.other_calcs import calc_volumes, inspiratory_least_squares

IMPLAUSIBLE_PRESSURE = -1000


def shear_transform(waveform, dt=0.02, time_offset=0):
    """
    Return shear transform function to do with what you want
    """
    m = (waveform[0] - waveform[-1]) / (dt * (len(waveform)-1))
    c = -m * (time_offset * dt)
    t = np.arange(len(waveform)) * dt
    return waveform + m * t + c


def find_shoulders(flow, pressure, dt):
    """
    Follows shear transform discussed in Stevenson et al. 2012.

    The goal here is to find the left and right shear locations for the pressure curve
    """
    # flow min idx is included in the shear calculation
    max_p_idx = np.argmax(pressure)
    min_f_idx = np.argmin(flow)
    # A first approximation of the shear transform is found by the maximum of
    # the shear transform between the first data point and the point of maximum
    # pressure
    shear_left = np.argmax(shear_transform(pressure[:max_p_idx+1], dt=dt))
    # The second shoulder is found by taking the maximum of the shear transform
    # between the point of maximum pressure and the point of minimum flow
    try:
        shear_right_func = shear_transform(pressure[max_p_idx:min_f_idx+1], dt=dt, time_offset=max_p_idx)
    except (IndexError, ValueError):
        return None, None
    # just use first max val instead of the global max. It should be good enough
    for idx, val in enumerate(shear_right_func[::-1][1:]):
        if shear_right_func[idx-1] <= val >= shear_right_func[idx+1]:
            # -2 is because we are advancing index by 1 above, and that the len gives
            # a length that is 1 greater than the final index. So its -1 - 1 = -2
            shear_right = (len(shear_right_func) - 2 - idx) + max_p_idx
            break
    else:  # shear max unable to be found
        return None, None

    return shear_left, shear_right


def get_least_squares_preds(flow, pressure, vols):
    """
    Get elastance, resistance, K, and residual as defined by least squares eq.

        Paw = E*V + R*V^dot + P0
    """
    a = np.array([vols, flow, [1] * len(pressure)]).transpose()
    try:
        least_square_result = np.linalg.lstsq(a, pressure)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    elastance, resist, K = least_square_result[0]
    try:
        residual = least_square_result[1][0]
    except IndexError:
        residual = None
    return elastance, resist, K, residual


def get_predicted_pressure_waveform(flow, pressure, vols):
    """
    Get the pressure waveform as predicted by the least squares model

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param vols: calculated volumes inspired over time

    :returns tuple: the modeled pressure and the residual
    """
    elastance, resist, K, residual = get_least_squares_preds(flow, pressure, vols)
    modeled_pressure = elastance * vols + resist * flow + K
    return modeled_pressure, residual


def get_predicted_pressure_waveform_with_shears(flow, pressure, vols, shear_left, shear_right):
    """
    Get the pressure waveform as predicted by the least squares model

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param vols: calculated volumes inspired over time
    :param shear_left: Left shear point
    :param shear_right: Right shear point

    :returns tuple: the modeled pressure and the residual
    """
    orig_len = len(flow)
    flow = flow[shear_left:shear_right+1]
    vols = vols[shear_left:shear_right+1]
    pressure = pressure[shear_left:shear_right+1]
    modeled_pressure, residual = get_predicted_pressure_waveform(flow, pressure, vols)
    modeled_pressure = np.append([np.nan] * shear_left, modeled_pressure)
    return np.append(modeled_pressure, [np.nan] * (orig_len - shear_right - 1)), residual


def find_intercept_via_points(a1, a2, b1, b2):
    """
    Determine if intersection between two lines (a and b) is found.

    :param a1: first point on line a
    :param a2: second point on line a
    :param b1: first point on line b
    :param b2: second point on line b

    :returns tuple: bool on whether intercept found,
                    and offset of closest point to intercept
    """
    intercept_found = ((a1 >= b1 and a2 < b2) or (a1 <= b1 and a2 > b2) or (a1 == b1))
    if intercept_found and abs(a1-b1) < abs(a2-b2):
        return True, 0
    elif intercept_found:
        return True, 1
    else:
        return False, None


def find_intercepts(a, b):
    """
    Find all intersections between two lines (a and b).
    """
    intercepts = []
    for i in range(1, len(a)):
        intercept_found, offset = find_intercept_via_points(a[i], a[i-1], b[i], b[i-1])
        if intercept_found:
            intercepts.append(i-offset)
    if not intercepts:
        return False, []
    return True, intercepts


def is_fit(flow, pressure, vols, peep, dt):
    """
    Evaluate newly reconstructed pressure if it needs to be fit again (step 8).

    If it is not fit, return False. If it is fit return True
    """
    # perform steps 1-2 again
    shear_left, shear_right = find_shoulders(flow, pressure, dt)
    if not shear_left or not shear_right:
        return False
    modeled_pressure, residual = get_predicted_pressure_waveform_with_shears(flow, pressure, vols, shear_left, shear_right)
    diff = abs(pressure[shear_left:shear_right+1] - modeled_pressure[shear_left:shear_right+1])
    auc = simps(pressure[shear_left:shear_right+1] - peep)

    # If the pressure is above an auc threshold then return False
    if simps(diff) > .045 * auc:
        return False

    return True


def find_best_left_right_shoulders(pressure, left_shldrs, right_shldrs, shear_min):
    """
    According to Newberry: The points on the pressure curve which are a maximum
    orthogonal distance above these lines are identified as the asynchrony shoulders.

    This function accomplishes the role as specified above. We find which two shoulders
    when connected, form a line of maximal orthogonal distance from the shear min.
    """
    combos = []
    for l_elem in left_shldrs:
        for r_elem in right_shldrs:
            combos.append((l_elem, r_elem))

    # now calculate orthogonal distance above shear_min
    max_ = IMPLAUSIBLE_PRESSURE
    max_idx = 0
    for idx, (l, r) in enumerate(combos):
        m = (pressure[r] - pressure[l]) / (r - l)
        b = pressure[l]
        y = m * (shear_min-l) + b
        if y - pressure[shear_min] > max_:
            max_idx = idx
            max_ = y - pressure[shear_min]

    return combos[max_idx]


def perform_single_iter_reconstruction(flow, pressure, vols, dt):
    """
    Perform a single iteration of pressure reconstruction following the first 7 steps of the
    IIPR algorithm

    :returns tuple: reconstructed pressure, response code
    """
    # step 1
    shear_left, shear_right = find_shoulders(flow, pressure, dt)
    # no shear transform found
    if not shear_left or not shear_right:
        return None, 2

    # step 2
    #
    # One idea that I have is to just stop analysis on breaths where the residual is
    # sufficiently low, but this is not discussed in paper at all.
    modeled_pressure, residual = get_predicted_pressure_waveform_with_shears(flow, pressure, vols, shear_left, shear_right)

    # Step 3. Paper says intersections are identified based on gradient of pressure
    # curve at the crossings. If grad_pressure<0 or grad_pressure<grad_fit_pressure
    # these are two circumstances that an intersection could be considered valid.
    #
    # First figure out how many intercepts we actually have.
    intercepts = []
    crossings = []
    cur_crossing = []
    for idx, val in enumerate(modeled_pressure):
        if val is np.nan or modeled_pressure[idx-1] is np.nan:
            continue
        pressure_grad = pressure[idx] - pressure[idx-1]
        modeled_pressure_grad = modeled_pressure[idx] - modeled_pressure[idx-1]
        intercept_found, offset = find_intercept_via_points(pressure[idx], pressure[idx-1], modeled_pressure[idx], modeled_pressure[idx-1])

        # found intercept
        if intercept_found and (pressure_grad < 0 or pressure_grad < modeled_pressure_grad):
            intercepts.append(idx-offset)
            if len(cur_crossing) == 0:
                cur_crossing.append(idx-offset)
            else:
                cur_crossing.append(idx-offset)
                crossings.append(cur_crossing)
                cur_crossing = []

    if len(crossings) == 0:
        return None, 3

    shear_mins = []
    for crossing in crossings:
        left_async_cross, right_async_cross = crossing
        shear_min = np.argmin(shear_transform(pressure[left_async_cross:right_async_cross+1], time_offset=left_async_cross, dt=dt)) + left_async_cross
        shear_mins.append([shear_min, left_async_cross, right_async_cross])

    # step 4
    for idx, crossing in enumerate(shear_mins):
        shear_min, left_async_cross, right_async_cross = crossing
        # create intercept line from min to first crossing
        #
        # we use shear_min +/- 1 per instructions: "To reduce the gradient of these lines slightly,
        # providing more reliable intersections, these lines begin from the points on either
        # side of the pressure minimum rather than from the minimum point itself."
        mod = 1 if shear_min - left_async_cross != 1 else 0
        if left_async_cross - shear_min == 0:
            left_async_shldr_found, left_async_shldrs = True, np.array([shear_min])
        else:
            m = (pressure[left_async_cross] - pressure[shear_min-mod]) / (left_async_cross - (shear_min-mod))
            b = pressure[shear_min-mod]
            arr_left = m * np.arange(-shear_min+1, 1) + b
            left_async_shldr_found, left_async_shldrs = find_intercepts(arr_left[::-1], pressure[:shear_min][::-1])
            left_async_shldrs = shear_min - 1 - np.array(left_async_shldrs)

        # create intercept line from min to second crossing
        mod = 1 if right_async_cross - shear_min != 1 else 0
        if right_async_cross - shear_min == 0:
            right_async_shldr_found, right_async_shldrs = True, np.array([shear_min])
        else:
            m = (pressure[right_async_cross] - pressure[shear_min+mod]) / (right_async_cross - (shear_min+mod))
            b = pressure[shear_min+mod]
            arr_right = m * np.arange(0, len(pressure)-shear_min-1) + b
            right_async_shldr_found, right_async_shldrs = find_intercepts(arr_right, pressure[shear_min+1:])
            right_async_shldrs = np.array(right_async_shldrs) + shear_min + 1

        # If we didn't find intercepts then fallback to uing least squares. I don't think
        # this can really happen but I'm being paranoid
        if not left_async_shldr_found or not right_async_shldr_found:
            return None, 4

        # Importantly: The points on the pressure curve which are a maximum
        # orthogonal distance above these lines are identified as the
        # asynchrony shoulders,
        best_left, best_right = find_best_left_right_shoulders(pressure, left_async_shldrs, right_async_shldrs, shear_min)
        shear_mins[idx].extend([best_left, best_right])

    # step 5
    #
    # Join asynchrony shoulders together so that we can fill in the space between them
    for shear_min_idx, left_async_cross, right_async_cross, left_async_shldr, right_async_shldr in shear_mins:
        m = (pressure[right_async_shldr] - pressure[left_async_shldr]) / (right_async_shldr - left_async_shldr)
        b = pressure[left_async_shldr]
        recon_line = np.arange(0, right_async_shldr-left_async_shldr+1) * m + b
        recon_line = np.concatenate([
            [IMPLAUSIBLE_PRESSURE] * (left_async_shldr),
            recon_line,
            [IMPLAUSIBLE_PRESSURE] * (len(pressure)-1-right_async_shldr)
        ])
        recon = np.array([pressure, recon_line]).max(axis=0)

    # step 6
    for idx, (shear_min_idx, left_async_cross, right_async_cross, left_async_shldr, right_async_shldr) in enumerate(shear_mins):

        # Do this for right hand side, but don't evaluate prior to final asynchrony.
        if idx != len(shear_mins) - 1:
            continue
        m = (pressure[right_async_shldr] - pressure[left_async_shldr]) / (right_async_shldr - left_async_shldr)
        # This doesn't make sense to perform if the slope is going up, only if it's going down
        if m > 0:
            continue

        # shears can be slightly off. Might need to run this for shear_left as well.
        if len(pressure)-1 == shear_right + 1:
            right_grad_idx = shear_right + 1
        else:
            right_grad_idx = shear_right + 2

        exp_grad = (pressure[right_grad_idx] - pressure[shear_right]) / 2
        b = pressure[right_grad_idx]
        line = []
        # First we just construct the vertical part of the line
        for i in range((right_grad_idx)-left_async_cross+1):
            i = -i
            y = exp_grad * i + b
            if y < pressure[left_async_cross]:
                line.append(y)
            else:
                line.append(pressure[left_async_cross])
        line = np.concatenate([
            [IMPLAUSIBLE_PRESSURE] * left_async_cross,
            line,
            [IMPLAUSIBLE_PRESSURE] * (len(pressure) - 1 - (right_grad_idx))
        ])
        recon = np.array([recon, line]).max(axis=0)

        # determine if we need to do a left-hand reconstruction
        #
        # It says this only happens if the first identified async crossing is lower (less than)
        # than half way between left pressure shoulder and point immediately previous.
        #
        # There are a few that happen in my example file:
        # cvc-rpi13-2018-07-05-16-29-18.98460; rel bns: 27, 29, 30, 31
        if shear_left >= 2 and pressure[left_async_shldr] < (pressure[shear_left] - pressure[shear_left-1]) + pressure[shear_left-1]:
            insp_grad = (pressure[shear_left] - pressure[shear_left - 2]) / 2
            y0 = pressure[shear_left-2]
            line = []
            # Construct the vertical and horizontal parts of the line
            for i in range(right_async_cross-(shear_left-2)+1):
                y = insp_grad * i + y0
                if y < pressure[right_async_cross]:
                    line.append(y)
                else:
                    line.append(pressure[right_async_cross])
            # now attach to the horizonal part
            line = np.concatenate([
                [IMPLAUSIBLE_PRESSURE] * (shear_left-2),
                line,
                [IMPLAUSIBLE_PRESSURE] * (len(pressure) - 1 - right_async_cross)
            ])
            recon = np.array([recon, line]).max(axis=0)

    # We do step 7 throughout the code previously. no need to redo it.
    return recon, 0


def preprocess_flow_pressure(flow, pressure, dt):
    vols = calc_volumes(flow, dt)
    pressure = np.array(pressure)
    flow = np.array(flow)
    return flow, pressure, vols


def perform_iipr_pressure_reconstruction(flow, pressure, x0, peep, dt):
    """
    Perform IIPR pressure reconstruction on a waveform. Can be used with a variety of algorithms
    such as PREDATOR or MIPR.

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param x0: index where flow crosses 0
    :param peep: positive end expiratory pressure
    :param dt: time in between observations

    :returns tuple: reconstructed pressure, code

    See below for code explanations
    """
    max_iter = 10
    flow, pressure, vols = preprocess_flow_pressure(flow, pressure, dt)

    # perform first iter (steps 1-7)
    recon, code = perform_single_iter_reconstruction(flow, pressure, vols, dt)
    if code != 0:
        return None, code

    # step 8-9
    iters = 1
    while not is_fit(flow, recon, vols, peep, dt) and iters < max_iter:
        recon_copy = copy(recon)
        recon, code = perform_single_iter_reconstruction(flow, recon, vols, dt)
        # if the algo wasn't successful then just use the last successful
        # reconstructed pressure curve
        if code != 0:
            return recon_copy, 5
        iters += 1
    return recon, 0


def perform_iipr_algo(flow, pressure, x0, peep, dt):
    """
    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param x0: index where flow crosses 0
    :param peep: positive end expiratory pressure
    :param dt: time in between observations

    :returns tuple: compliance, resistance, residual, response code

    Response codes:
        0: Used algorithm and was successful
        2: No shear points found
        3: No crossings found
        4: Used least squares as fallback because no pressure intercepts found
           when performing step 4 of algorithm.
        5: Failed on step 9, so fallback to least squares for last found reconstruction
    """
    # my first iteration of this function used the whole breath for insp least squares. I
    # still wonder if this is a better way to go. We can probably test this.
    recon, code = perform_iipr_pressure_reconstruction(flow, pressure, x0, peep, dt)
    if code in [2, 3, 4]:
        # just supply 0 as tvi because it doesnt matter for this func
        plat, comp, resist, K, residual = inspiratory_least_squares(flow, pressure, x0, dt, peep, 0)
    if code in [0, 5]:
        plat, comp, resist, K, residual = inspiratory_least_squares(flow, recon, x0, dt, peep, 0)
    return comp, resist, residual, code
