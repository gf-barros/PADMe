""" DMD modules
    Acknowledged that f-strings and logging should not be used together in some situations. 
    For instance, see: https://docs.python.org/3/howto/logging.html#optimization """
# pylint: disable=logging-fstring-interpolation

import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DMD:
    """
    Parameters
    ----------
    snapshots_matrix : NumPy 2D-array
        Snapshots matrix.
    dmd_parameters : Dict
        Contains DMD hyperparameters:
            - factorization_algorithm: str
                Defines possible type of factorization algorithm:
                    - randomized_svd
                    - incremental_svd
                    - standard_svd

            - basis_vectors : int
                Number of basis vectors in DMD modes (r)
            - randomized_svd_parameters : Dict
                If svd_randomized is chosen as factorization_algorithm,
                some hyperparameter are required. This key is not needed if
                a different factorization_algorithm is required.
                    Keys:
                    - power_iterations : int
                    - oversampling : int
            - dt_simulation : float
                Time step size used in the simulations (for input and output).
            - starting_step : int (Optional)
                Starting column for slicing the snapshot matrix (default: 0)
            - ending_step : int (Optional)
                Ending column for slicing the snapshot matrix (default: snapshots_matrix.shape[1])

        Example of dmd_parameters dict:
            dmd_parameters = {
                "factorization_algorithm": "randomized_svd",
                "basis_vectors": 50,
                "randomized_svd_parameters":
                    {
                        "power_iterations": 1,
                        "oversampling": 20
                    }
                "starting_step": 20,
                "dt_simulation": 0.05
            }

    Returns
    -------
    self.dmd_approximation : Dict
        Dictionary containing modes, eigenvalues, singular values and approximate solution.

    """

    def __init__(self, snapshots_matrix, dmd_parameters):
        self.snapshots_matrix = snapshots_matrix
        self.parameters = dmd_parameters
        self.upper_bound = None
        self.lower_bound = None
        self.snapshots_x1 = None
        self.snapshots_x2 = None
        self.dmd_approximation = {}

    def __split_dataset(self):
        self.lower_bound = self.parameters.get("starting_step", 0)
        self.upper_bound = self.parameters.get(
            "ending_step", self.snapshots_matrix.shape[1]
        )

        # X1 where X1 = [x0, x1, ... , xn-1]
        split_x1_matrix = self.snapshots_matrix[
            :, self.lower_bound : self.upper_bound - 1
        ]
        # X2 where X2 = [x1, x2, ... , xn]
        split_x2_matrix = self.snapshots_matrix[
            :, self.lower_bound + 1 : self.upper_bound
        ]

        # Check if snapshots matrix rank is >= the basis vectors
        rank_x1 = self.upper_bound - 1 - self.lower_bound
        if rank_x1 <= self.parameters["basis_vectors"]:
            self.parameters["basis_vectors"] = rank_x1
            logger.info(
                (
                    "Snapshots matrix rank is lesser than basis vectors chosen "
                    f"as input. \nChanging dmd_parameters['basis_vectors'] to {rank_x1}."
                )
            )

        # Update snapshots matrix with correct dimensions
        self.snapshots_matrix = self.snapshots_matrix[
            :, self.lower_bound : self.upper_bound
        ]

        return split_x1_matrix, split_x2_matrix

    def __randomized_svd(self, mat, basis_vectors, power_iterations, oversampling):
        matrix_rank = mat.shape[1]
        p_random_vectors = np.random.randn(matrix_rank, basis_vectors + oversampling)
        z_projected_matrix = mat @ p_random_vectors
        for _ in range(power_iterations):
            z_projected_matrix = mat @ (mat.T @ z_projected_matrix)
        q_values, _ = np.linalg.qr(z_projected_matrix, mode="reduced")
        y_reduced_matrix = q_values.T @ mat
        u_vectors_y, s_values, vt_vectors = np.linalg.svd(
            y_reduced_matrix, full_matrices=0
        )
        u_vectors = q_values @ u_vectors_y
        return u_vectors, s_values, vt_vectors

    def __standard_svd(self, mat):
        u_vectors, s_values, vt_vectors = np.linalg.svd(
            mat,
            full_matrices=False,
            compute_uv=True,
            hermitian=False,
        )
        return u_vectors, s_values, vt_vectors

    def factorization(self):
        self.snapshots_x1, self.snapshots_x2 = self.__split_dataset()
        if self.parameters["factorization_algorithm"] == "randomized_svd":
            u_vectors, s_values, vt_vectors = self.__randomized_svd(
                self.snapshots_x1,
                self.parameters["basis_vectors"],
                self.parameters["randomized_svd_parameters"]["power_iterations"],
                self.parameters["randomized_svd_parameters"]["oversampling"],
            )
            self.dmd_approximation["u"] = u_vectors[
                :, 0 : self.parameters["basis_vectors"]
            ]
            self.dmd_approximation["s"] = s_values[0 : self.parameters["basis_vectors"]]
            self.dmd_approximation["vt"] = vt_vectors[
                0 : self.parameters["basis_vectors"], :
            ]

        elif self.parameters["factorization_algorithm"] == "standard_svd":
            u_vectors, s_values, vt_vectors = self.__standard_svd(
                self.snapshots_x1,
            )
            self.dmd_approximation["u"] = u_vectors[
                :, 0 : self.parameters["basis_vectors"]
            ]
            self.dmd_approximation["s"] = s_values[0 : self.parameters["basis_vectors"]]
            self.dmd_approximation["vt"] = vt_vectors[
                0 : self.parameters["basis_vectors"], :
            ]

    def dmd_core(self):
        s_values = np.divide(1.0, self.dmd_approximation["s"])
        s_values = np.diag(s_values)
        u_vectors = np.transpose(self.dmd_approximation["u"])
        vt_vectors = np.transpose(self.dmd_approximation["vt"])
        a_tilde = np.linalg.multi_dot(
            [u_vectors, self.snapshots_x2, vt_vectors, s_values]
        )
        self.dmd_approximation["eigenvals_original"], eigenvec = np.linalg.eig(a_tilde)
        eigenval = np.log(self.dmd_approximation["eigenvals_original"]) / (
            self.parameters["dt_simulation"]
        )
        self.dmd_approximation["eigenvals_processed"] = eigenval
        phi_dmd = np.linalg.multi_dot(
            [self.snapshots_x2, vt_vectors, s_values, eigenvec]
        )
        phi_inv = np.linalg.pinv(phi_dmd)
        initial_vector = self.snapshots_x1[:, 0]
        b_vector = np.dot(phi_inv, initial_vector)
        b_vector = b_vector[:, np.newaxis]
        t_vector = (
            np.arange(start=self.lower_bound, stop=self.upper_bound)
            * self.parameters["dt_simulation"]
        )
        t_vector = t_vector[np.newaxis, :]
        self.dmd_approximation["eigenvals_processed"] = eigenval
        eigenval = eigenval[:, np.newaxis]
        temp = np.multiply(eigenval, t_vector)
        temp = np.exp(temp)
        dynamics = np.multiply(b_vector, temp)
        x_dmd = np.dot(phi_dmd, dynamics)
        self.dmd_approximation["dmd_matrix"] = x_dmd
