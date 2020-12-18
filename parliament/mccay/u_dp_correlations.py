"""
Module containing functions that use correlations to go from velocity
to pressure drop and vice-versa.
"""
import numpy as np
from scipy import optimize


def pipe_u_of_dp(dp, D, L, nu, roughness):
    """Finds average fluid velocity in a clean, circular pipe given
    pressure drop, dimensions, and pipe/fluid properties. Units as
    specified below are required.

    Switches between laminar and turbulent flow around Re = 2300 based
    on the laminar calculation of Reynolds number. This creates a
    discontinuity at this point.

    The dp input must be greater than zero and the returned velocity
    will be greater than zero. This is opposite the usual sign
    convention because the algorithm involves two nonlinear solves
    that don't handle negative numbers well.

    If you need to process a negative pressure or have a different sign
    convention, use external logic to change the signs of the input
    pressure and the output velocity.

    Parameters
    ----------
    dp : float
        Pressure drop in units Pa*m**3/kg (pressure divided by
        density). dp must be greater than zero, as mentioned above.
    D : float
        Pipe diameter in meters
    L : float
        Pipe length in meters
    nu : float
        Kinematic viscosity in m**2/s (often 1.544e-5 m**2/s)
    roughness : float
        Pipe roughness in meters (usually 0.0025e-3 m)

    Returns
    -------
    u_sol : float
        Average velocity, always greater than zero as mentioned
        earlier.
    """

    # Colebrook's formula - zero of fd
    def colebrook(fd, Re, sr):

        return 1/np.sqrt(fd) + 2.0*np.log10(sr/3.7 + 2.51/(Re*np.sqrt(fd)))

    # zero of u for a pipe
    def u_zero(u, dp, D, L, nu, sr):

        # Reynolds number
        Re = u*D/nu

        # usual range for fd
        a = 0.008
        b = 0.1

        # solve for fd
        sol_fd = optimize.brentq(colebrook, a, b, args=(Re,sr))

        fd = sol_fd

        return u - np.sqrt(2*dp*D/(L*fd))

    sr = roughness/D        # stiffness ratio
    a = 1                   # velocity lower bound
    b = 1000                # velocity upper bound

    # solve

    # Estimate velocity with laminar expression
    u_temp = 2*(D**2)*dp/(64*L*nu)

    Re = np.abs(u_temp*D/nu)

    # If turbulent, solve non-linear equation
    if Re >= 2300:
        u_sol = optimize.brentq(u_zero, a, b, args=(dp, D, L, nu, sr))
    # If laminar, use the temporary velocity
    else:
        u_sol = u_temp
        # fd = 64/Re
        # u_sol = np.sqrt(dp*2*D/(L*fd))

    return u_sol

def pipe_dp_of_u(u, D, L, nu, roughness):
    """Finds pressure drop in a clean, circular pipe given average
    velocity, pipe geometry, and flow/pipe characteristics.

    Switches between laminar and turbulent at Re = 2300, so there is a
    discontinuity here.

    Follows same sign convention as pipe_u_of_dp. dp and u are both
    always positive. Use external logic for different sign conventions.

    Parameters
    ----------
    u : float
        Average fluid velocity [m/s]. Always greater than zero.
    D : float
        Pipe diameter [m]
    L : float
        Pipe length [m]
    nu : float
        Kinematic viscosity [m**2/s]
    roughness : float
        Pipe wall roughness [m]

    Returns
    -------
    dp : float
        Pressure drop over density for conditions [Pa*m**3/kg]. Always
        greater than zero.

    """

    # Reynolds number
    Re = u*D/nu

    # Colebrook formula
    def colebrook(fd, Re, sr):

        return 1/np.sqrt(fd) + 2.0*np.log10(sr/3.7 + 2.51/(Re*np.sqrt(fd)))

    # Laminar case
    if Re < 2300:

        fd = 64.0/Re

    # turbulent case
    else:

        # Stiffness ratio
        sr = roughness/D

        a = 0.008   # Lower bound
        b = 0.1     # Upper bound

        fd = optimize.brentq(colebrook, a, b, args=(Re,sr))

    # Solve for dp
    dp = (L*fd/(D*2.0))*u**2

    return dp

def sudden_contraction(u_thr, d, D):

    return (u_thr**2)*0.5*0.42*(1 - (d**2)/(D**2))

def sudden_expansion(u_thr, d, D):

    return (u_thr**2)*0.5*(1 - (d**2)/(D**2))**2

# # Sudden contraction pressure loss
# def hgSC(u_avg,d,D):
#     return -(u_avg**2)*0.5*0.42*(1 - (d**2)/(D**2))
#
# # Sudden expansion pressure loss
# def hgSE(u_thr,d,D):
#     return -(u_thr**2)*0.5*(1 - (d**2)/(D**2))**2
