import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN

def build_qnn(q_num):
    # Create a quantum feature map
    feature_map = ZZFeatureMap(q_num, reps=2)

    # Create a variational circuit
    var_circuit = RealAmplitudes(q_num, entanglement='linear', reps=2)

    # Combine the feature map and variational circuit
    circuit = feature_map.compose(var_circuit)

    observable = SparsePauliOp.from_list([("Z" + "I" * (q_num-1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=var_circuit.parameters,
    )

    return qnn



