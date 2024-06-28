import torch
from torch import Tensor
from panqec.codes import StabilizerCode
import numpy as np
from numpy.typing import NDArray
from ._auxiliary_functions import generate_syndrome, sample_errors, get_logical_errors
from typing import Callable
from scipy.sparse import csr_matrix, vstack


class DataGenerator:
    """Data generator object."""
    """Stabilizer code specific attributes."""
    logicals: csr_matrix
    stabilizers: csr_matrix
    n: int  # number of physical qubits
    d: int

    """Generation attributes."""
    batch_size: int
    error_rate: float

    """Some private attributes."""
    _verbose_print: Callable[[str], None]
    _categorical_dict: dict[tuple[int, ...], int]
    _categorical_classification: bool

    def __init__(
            self,
            code: StabilizerCode,
            error_rate: float,
            batch_size: int,
            categorical_classification: bool = True,
            one_hot: bool = False,
            verbose: bool = True,
            for_ldpc: bool = False,
    ) -> None:
        """
        Initialize the Dataset.

        :param code: The stabilizer code associated.
        :param error_rate: The error rate.
        :param batch_size: The batch size.
        :param categorical_classification: Whether the task is to do categorical classification or multi label.
        :param one_hot: If classes should be returned one-hot encoded (Only has affect when using categorical classification).
        :param verbose: If messages should be printed.
        :param for_ldpc: If used in ldpc library -> slightly different process.
        """
        self._verbose_print: Callable[[str], None] = print if verbose else lambda x: None
        self._categorical_classification = categorical_classification
        self._one_hot = one_hot
        self._for_ldpc = for_ldpc

        self.d = len(code.size)

        self.error_rate = error_rate
        self.batch_size = batch_size

        """Get X and Z logicals from lattice and combine them."""
        x_logical, z_logical = csr_matrix(code.logicals_x), csr_matrix(code.logicals_z)
        self.logicals = vstack((x_logical, z_logical))

        """Transpose the stabilizers."""
        block_size = code.size[0] ** self.d
        x, y = code.stabilizer_matrix.shape

        original = np.array(code.stabilizer_matrix.todense())
        matrix = np.zeros_like(original)
        for i in range(x // block_size):
            for j in range(y // block_size):
                matrix[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size] = (
                    original[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size].T
                )

        self.stabilizers = code.stabilizer_matrix if for_ldpc else csr_matrix(matrix)
        self.n = code.n

    def _check_class(self, logical_error: NDArray) -> int:
        """
        Get the class corresponding to a logical error.

        :param logical_error: The logical error.
        :returns: The class as int [0: n].
        """
        power = 2 ** (np.array(range(self.d*2))[::-1])
        return np.inner(logical_error, power)

    def generate_batch(self, use_qmc: bool, device: torch.device) -> tuple[Tensor, Tensor, csr_matrix] | tuple[Tensor, Tensor]:
        """
        Generate the dataset.

        :param use_qmc: Whether quasi-monte carlo sampling is used.
        :param device: The device that uses the data.
        :returns: The syndrome and logical error. If used for ldpc library additionally return errors.
        """
        self._verbose_print("\tGenerating Errors")
        errors = sample_errors(self.error_rate, self.n, use_qmc, self.batch_size)


        self._verbose_print("\tConstructing Syndrome Matrices")
        syndrome_matrices = generate_syndrome(self.stabilizers, errors)

        self._verbose_print("\tMeasuring Logicals")
        logical_errors = get_logical_errors(self.logicals, errors)

        """ Transform to indices if we use categorical classification."""
        if self._categorical_classification:
            # CrossEntropyLoss requires indices as y_true.
            logical_errors = np.apply_along_axis(self._check_class, 1, logical_errors)

            # Transform to one-hot encoded classes if needed.
            if self._one_hot:
                encoded_arr = np.zeros((logical_errors.size, 2**(2*self.d)), dtype=int)
                encoded_arr[np.arange(logical_errors.size), logical_errors] = 1
                logical_errors = encoded_arr

        """Convert to tensors."""
        syndrome_matrices = torch.tensor(data=syndrome_matrices, dtype=torch.float, device=device)
        logical_errors = torch.tensor(data=logical_errors, dtype=torch.long if not self._one_hot else torch.float, device=device)
        if self._for_ldpc:
            return syndrome_matrices, logical_errors, errors
        return syndrome_matrices, logical_errors
