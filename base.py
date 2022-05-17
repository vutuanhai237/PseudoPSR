import types
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
