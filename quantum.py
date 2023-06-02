import qiskit
import constant
import numpy as np

def measure(qc: qiskit.QuantumCircuit, qubits, cbits=[]):
    """As its function name
    Args:
        qc (qiskit.QuantumCircuit): measuremed circuit
        qubits (np.ndarray): list of measuremed qubit
        cbits (list, optional): classical bits. Defaults to [].
    Returns:
        qiskit.QuantumCircuit: added measure gates circuit
    """
    if cbits == []:
        cbits = qubits.copy()
    for i in range(0, len(qubits)):
        qc.measure(qubits[i], cbits[i])
    counts = qiskit.execute(
            qc, backend=constant.backend,
            shots=constant.num_shots).result().get_counts()
    return counts.get("0" * len(qubits), 0) / constant.num_shots

def zz_measure(qc: qiskit.QuantumCircuit):
    _00 = np.asarray([1, 0, 0, 0])
    _01 = np.asarray([0, 1, 0, 0])
    _10 = np.asarray([0, 0, 1, 0])
    _11 = np.asarray([0, 0, 0, 1])
    psi = qiskit.quantum_info.Statevector(qc)
    return np.abs(np.inner(_00, psi))**2 - np.abs(np.inner(_01, psi))**2 - np.abs(np.inner(_10, psi))**2 + np.abs(np.inner(_11, psi))**2


def create_cry_ansatz(qc: qiskit.QuantumCircuit, thetas):
    for i in range(0, qc.num_qubits):
        if i == qc.num_qubits - 1:
            qc.cry(thetas[i], qc.num_qubits - 1, 0)
        else:
            qc.cry(thetas[i], 0, i + 1)
    return qc

def create_rzxz_ansatz(qc: qiskit.QuantumCircuit, thetas):
    k = 0
    for i in range(0, qc.num_qubits):
        qc.rz(thetas[k], i)
        qc.rx(thetas[k + 1], i)
        qc.rz(thetas[k + 2], i)
        k = k + 3
    return qc