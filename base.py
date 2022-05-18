from doctest import REPORT_ONLY_FIRST_FAILURE
import types
from typing import Tuple
import numpy as np
import constant

def grid_search(R, S, deltaR, deltaS, E, partialE0, PSR: types.FunctionType):
    nR = 2 * R / deltaR
    nS = 2 * S / deltaS
    rs = np.linspace(-R, R, nR)
    ss = np.linspace(-S, S, nS)
    deltas = []
    for r in rs:
        for s in ss:
            partialE0_hat = PSR(E, np.zeros(partialE0.shape[0]))
            delta_rs = {'r': r, 's': s, 'delta': np.abs(partialE0 - partialE0_hat)}
            deltas.append(delta_rs)
    return min(deltas, key=lambda delta_rs: delta_rs['delta'])

def generate_x(O: Tuple, R: float):
    """From the center point in n dimensional, generate 2n point round O by adding / subtracting R value 

    Args:
        O (Tuple): the center point
        R (float): radius

    Returns:
        list: around points
    """
    n = len(O)
    xs = []
    for i in range(0, n):
        x_plus, x_minus = O.copy(), O.copy()
        x_plus[i] += R
        x_minus[i] -= R
        xs.append(x_plus)
        xs.append(x_minus)
    return xs

def spatial_search(R: float, T: float = 10e-8, tmax: int = 1000, k: int = 2, E: types.FunctionType = None, partialE0: np.ndarray = np.empty, PSR: types.FunctionType = None):
    n = partialE0.shape[0]
    O = np.array([0, 0])
    t = 0
    while(True):
        X = []
        deltas = []
        for r, s in X:
            partialE0_hat = PSR(E, np.zeros(n))
            delta_rs = {'r': r, 's': s, 'delta': np.abs(partialE0 - partialE0_hat)}
            deltas.append(delta_rs)
        min_value = min(deltas, key=lambda delta_rs: delta_rs['delta'])
        if min_value < T or t == tmax:
            return O
        O_new = min_value
        if O_new != O:
            R = 2*R
        else:
            R = R / 2
        t += 1
    return


def two_term_psr(E:types.FunctionType, thetas: np.ndarray):
    """\partial E(x) = 1/2(E(x + s) - E(x - s)). Only in case lambdas = [-1, 1]

    Args:
        E (types.FunctionType): cost function
        thetas (np.ndarray): parameters

    Returns:
        np.ndarray: gradient value
    """
    grads = np.zeros(thetas.shape[0])
    for i in range(0, thetas.shape[0]):
        thetas_plus, thetas_minus = thetas.copy(), thetas.copy()
        thetas_plus[i] += constant.two_term_psr_constant['s']
        thetas_minus[i] -= constant.two_term_psr_constant['s']
        grads[i] = constant.two_term_psr_constant['r']*(E(thetas_plus) - E(thetas_minus))
    return grads

def four_term_psr(E: types.FunctionType, thetas:np.ndarray):
    """\partial E(x) = 1/2(E(x + s) - E(x - s)). Only in case lambdas = [-1, 0, 1]
    Args:
        E (types.FunctionType): cost function
        thetas (np.ndarray): parameters

    Returns:
        np.ndarray: gradient value
    """
    grads = np.zeros(thetas.shape[0])
    for i in range(0, len(thetas)):
        thetas1, thetas2 = thetas.copy(), thetas.copy()
        thetas3, thetas4 = thetas.copy(), thetas.copy()
        thetas1[i] += constant.four_term_psr_constant['alpha']
        thetas2[i] -= constant.four_term_psr_constant['alpha']
        thetas3[i] += constant.four_term_psr_constant['beta']
        thetas4[i] -= constant.four_term_psr_constant['beta']
        grads[i] = - (constant.four_term_psr_constant['d_plus'] * (
            E(thetas1) - E(thetas2)) - constant.four_term_psr_constant['d_minus'] * (
            E(thetas3) - E(thetas4)))
    return grads
