import numpy as np
import metrics
import qiskit


num_shots = 10000
learning_rate = 0.04
noise_prob = 0.00
backend = qiskit.Aer.get_backend('qasm_simulator')
# For parameter-shift rule
two_term_psr_constant = {
    'r': 1/2,
    's': np.pi / 2
}

metric = metrics.Cosine
four_term_psr_constant = {
    'alpha': np.pi / 2,
    'beta' : 3 * np.pi / 2,
    'd_plus' : (np.sqrt(2) + 1) / (4*np.sqrt(2)),
    'd_minus': (np.sqrt(2) - 1) / (4*np.sqrt(2))
}