import qiskit
import constant

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

def create_cry_ansatz(qc: qiskit.QuantumCircuit, thetas):
    for i in range(0, qc.num_qubits):
        if i == qc.num_qubits - 1:
            qc.cry(thetas[i], qc.num_qubits - 1, 0)
        else:
            qc.cry(thetas[i], 0, i + 1)
    return qc

