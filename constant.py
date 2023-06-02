import numpy as np
import metrics
import qiskit

def unit_vector(i, length):
    unit_vector = np.zeros((length))
    unit_vector[i] = 1.0
    return unit_vector

num_shots = 20000
learning_rate = 0.04
noise_prob = 0.00
backend = qiskit.Aer.get_backend('aer_simulator')
# For parameter-shift rule
two_term_psr_constant = {
    'r': 1/2,
    's': np.pi / 2
}

metric = metrics.Cosine
four_term_psr = {
    'alpha': np.pi / 2,
    'beta' : 3 * np.pi / 2,
    'd_plus' : (np.sqrt(2) + 1) / (4*np.sqrt(2)),
    'd_minus': (np.sqrt(2) - 1) / (4*np.sqrt(2))
}