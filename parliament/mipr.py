"""
mipr
~~~~~

From Chiew's paper "Assessing mechanical ventilation asynchrony through iterative
airway pressure reconstruction."
"""
import numpy as np

from parliament.iipr import get_least_squares_preds, IMPLAUSIBLE_PRESSURE, preprocess_flow_pressure


def get_predicted_pressure_waveform(flow, pressure, vols):
    """
    Get the pressure waveform as predicted by the least squares model

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param vols: calculated volumes inspired over time

    :returns tuple: the modeled pressure, elastance, resistance, and the residual
    """
    elastance, resist, K, residual = get_least_squares_preds(flow, pressure, vols)
    modeled_pressure = elastance * vols + resist * flow + K
    return modeled_pressure, elastance, resist, residual


def perform_mipr(flow, pressure, x0, peep, dt, iters):
    """
    Run MIPR algorithm that successively tries to reconstruct deformed pressure in volume control
    modes in the case of flow asynchrony.

    :param flow: array vals of flow measurements in L/s
    :param pressure: array vals of pressure obs
    :param x0: index where flow crosses 0
    :param peep: positive end expiratory pressure
    :param dt: time between observations on x axis
    :param iters: number iterations to run mipr algo

    :returns tuple: compliance, resistance, residual, response code

    Response codes:
        0: Used algorithm and was successful
    """
    flow, pressure, vols = preprocess_flow_pressure(flow, pressure, dt)

    # I have seen that this can fail because least squares doesnt converge. Need to be able to
    # handle this.
    elastance, resist, K, residual = get_least_squares_preds(flow, pressure, vols)
    peak_p = np.argmax(pressure)
    # According to Chiew: "Late asynchronies, detected when the peak pressure is not located
    # at the end of a breathing cycle." This is a bit counter-intuitive, but normally the peak
    # pressure is supposed to occur toward the end of inspiration. So if the peak pressure occurs
    # early in inspiration then that means the patient is pulling hard toward the end of inspiration.
    #
    # Chiew doesnt provide discrete implementation instructions though.
    # So we divide the inspiratory section into half and then say if the peak pressure is not in
    # the last half then its a late asynchrony
    if 0 <= peak_p < x0/2:
        recon_line = np.concatenate([
            pressure[:peak_p+1], [pressure[peak_p]] * (x0-peak_p-1), pressure[x0:]
        ])
        pressure = np.array([pressure, recon_line]).max(axis=0)

    for i in range(iters):
        modeled_pressure, elastance, resistance, residual = get_predicted_pressure_waveform(
            flow[:x0], pressure[:x0], vols[:x0]
        )
        recon_line = np.concatenate([
            modeled_pressure, [IMPLAUSIBLE_PRESSURE] * (len(pressure) - x0)
        ])
        pressure = np.array([pressure, recon_line]).max(axis=0)
    modeled_pressure, elastance, resistance, residual = get_predicted_pressure_waveform(
        flow[:x0], pressure[:x0], vols[:x0]
    )

    # Option to quantify asynchrony using Chiew's AUC calcs.
    # MAsyn = (AUCrec − AUCori) / AUCrec × 100%. For now we don't need this.
    return 1/elastance, resistance, residual, 0
