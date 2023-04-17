# Notebooks

## How to use PADMe on Jupyter Notebooks

### Demonstration notebooks

Basically PADMe can be invoked using two dictionaries as guides:
- The first one, `snapshot_ingestion_parameters`, is responsible for giving directions towards the data that will be ingested. In this current version, PADMe supports four types of simulation output files: HDF5 from libMesh, FEniCS and EdgeCFD, ASCII vtk from FreeFem++. This dictionary is structured according to these data types, although more types can be added as new keys and read functions can be included. Current keys are:
  - `filenames`: List[str] (common to all possible file_type_str)
      List of strings containing the files to be ingested.
      Eg. ["/home/file001.h5", "/home/file002.h5"]
  - `dataset`: str (exclusive to h5_libmesh)
      String informing the key where the data will be read.
      Eg. "pressure"
  - `starting_line`: int (exclusive to vtk_freefem)
      Int describring starting line for reading nodal values in file.
      Eg. `2256`
  - ending_line: int (exclusive to vtk_freefem)
      Int describing ending line for reading nodal values in file.
      Eg. 2456



- `dmd_parameters`: Contains DMD hyperparameters:
  - `factorization_algorithm`: str
      Defines possible type of factorization algorithm:
          - randomized_svd
          - incremental_svd
          - standard_svd

  - `basis_vectors` : int
      Number of basis vectors in DMD modes (r)
  - `randomized_svd_parameters` : Dict
      If svd_randomized is chosen as factorization_algorithm,
      some hyperparameter are required. This key is not needed if
      a different factorization_algorithm is required.
          Keys:
          - power_iterations : int
          - oversampling : int
  - `dt_simulation` : float
      Time step size used in the simulations (for input and output).
  - `starting_step` : int (Optional)
      Starting column for slicing the snapshot matrix (default: 0)
  - `ending_step` : int (Optional)
      Ending column for slicing the snapshot matrix (default: snapshots_matrix.shape[1])
