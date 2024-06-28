from numpy.typing import NDArray
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import qmc


def sample_errors(error_rate: float, qubits: int, use_qmc: bool, size: int) -> csr_matrix:
    """
    Generate errors on the lattice using depolarizing noise.

    :param error_rate: The error rate.
    :param qubits: The amount of qubits on the lattice.
    :param use_qmc: If quasi monte-carlo sampling should be used.
    :param size: The batch size of the noise.
    """
    p_i = 1 - error_rate
    p_a = error_rate / 3

    if use_qmc:
        cumulative_distribution = np.array([p_a, p_a, p_a, p_i]).cumsum()
        sampler = qmc.Sobol(d=qubits).random(size)
        samples = np.searchsorted(cumulative_distribution, sampler)
        noise = np.array(['X', 'Z', 'Y', 'I'])[samples]
    else:
        random = np.random.default_rng()
        noise = random.choice(
            ['I', 'X', 'Z', 'Y'],
            size=qubits * size,
            p=[p_i, p_a, p_a, p_a]
        ).reshape(size, qubits)

    """Generate noise for both primal and dual lattice."""
    noise_bsf = np.zeros((2, size, qubits), dtype=np.uint8)

    """Check where error are applied."""
    noise_bsf[0][np.where((noise == 'X') | (noise == 'Y'))] = 1
    noise_bsf[1][np.where((noise == 'Z') | (noise == 'Y'))] = 1

    """Format errors in BSF."""
    errors = np.hstack([noise_bsf[0], noise_bsf[1]])
    errors = csr_matrix(errors, dtype=np.uint8)
    return errors


def generate_syndrome(
        stabilizer_matrix: csr_matrix,
        errors: csr_matrix,
) -> NDArray[np.uint8]:
    """
    Generate the matrix of stabilizers ( M(sigma) ).

    :param stabilizer_matrix: The stabilizer matrix of the code.
    :param errors: The errors array in bsf (b, 2n).
    :return: The syndromes (b, 2, l**d).
    """
    qubits = errors.shape[1] // 2  # Slice into primal / dual error noise.

    """Separate errors to primal and dual lattice."""
    errors_primal, errors_dual = errors[:, :qubits], errors[:, qubits:]
    errors_dp = hstack((errors_dual, errors_primal), format="csr", dtype=np.uint8)  # Re arrange errors.

    """Calculate syndrome from stabilizer matrix and errors."""
    syndromes = errors_dp @ stabilizer_matrix.T
    syndromes.data = np.mod(syndromes.data, 2)
    syndromes.eliminate_zeros()  # Format to make it csr.

    syndromes = np.array(syndromes.todense(), dtype=np.uint8)
    return syndromes


def get_logical_errors(
        logicals: csr_matrix,
        errors: csr_matrix,
) -> NDArray[np.uint8]:
    """
    Get logical content of the error.

    :param logicals: The logicals on the lattice.
    :param errors: The errors.
    :returns: The logical content (b, 2**d).
    """
    qubits = errors.shape[1] // 2  # Slice into primal / dual error noise.

    """Separate errors to primal and dual lattice."""
    errors_primal, errors_dual = errors[:, :qubits], errors[:, qubits:]
    errors_dp = hstack((errors_dual, errors_primal), format="csr", dtype=np.uint8)  # Re arrange errors.

    """Calculate logical content from stabilizer matrix and errors."""
    logical_content = errors_dp @ logicals.T
    logical_content.data = np.mod(logical_content.data, 2)
    logical_content.eliminate_zeros()

    logical_content = np.array(logical_content.todense(), dtype=np.uint8)
    return logical_content
