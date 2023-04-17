""" Functions for building features for DMD modeling """

from itertools import islice

import h5py
import meshio
import logging
import numpy as np

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_vtk(filename, starting_line=125939, ending_line=210239):
    """
    Function used to read nodal values from ASCII vtk files.

    Parameters
    ----------
    filename : str
        String containing the filename for reading files in vtk (freefem++).
    starting_line : int
        Integer representing the line where nodal values start.
        Default: 125939 (SIRDs snapshots)
    ending_line : int
        Integer representing the line where nodal values end.
        Default: 210239 (SIRDs snapshots)

    Returns
    -------
    array : np.array
        Numpy array containing nodal values.
    """
    with open(filename) as lines:
        array = np.genfromtxt(islice(lines, starting_line, ending_line))
    return array


def read_h5_libmesh(filename, dataset):
    """
    Function used to read nodal values from H5 files.

    Parameters
    ----------
    filename : str
        String containing the filename for reading files in h5 (libMesh and EdgeCFD).
    dataset : str
        String containing the dataset desired.

    Returns
    -------
    array : np.array
        Numpy array containing nodal values.
    """
    h5_file = h5py.File(filename, "r")
    data = h5_file[(dataset)]
    data_array = np.array(data, copy=True)
    h5_file.close()
    return data_array


def read_h5_fenics(filename, dataset="vector_0"):
    """
    Function used to read nodal values from H5 files.

    Parameters
    ----------
    filename : str
        String containing the filename for reading files in h5 (libMesh and EdgeCFD).
    dataset : str
        String containing the dataset desired.

    Returns
    -------
    array : np.array
        Numpy array containing nodal values.
    """
    h5_file = h5py.File(filename, "r")
    for key in h5_file.keys():
        group = f[key]
        for key in group.keys():
            data = group[(dataset)]
    data_array = np.array(data, copy=True)
    h5_file.close()
    return data_array


def snapshots_assembly(file_type_str, snapshot_ingestion_parameters):
    """
    Function used to assembly snapshots matrix from csv, h5 or vtk files.

    Parameters
    ----------
    file_type_str : str
        String describing the simulations output files type.
        Possible inputs:
            - h5_libmesh: libMesh/EdgeCFD HDF5 files.
            - h5_fenics: FEniCS HDF5 files.
            - vtk_freefem: freefem++ ASCII VTK files.
    snapshot_ingestion_parameters : dict
        Dictionary containing the information regarding the files.
        Keys:
            - filenames: List[str] (common to all possible file_type_str)
                List of strings containing the files to be ingested.
                Eg. ["/home/file001.h5", "/home/file002.h5"]
            - dataset: str (exclusive to h5_libmesh)
                String informing the key where the data will be read.
                Eg. "pressure"
            - starting_line: int (exclusive to vtk)
                Int describring starting line for reading nodal values in file.
                Eg. 2256
            - ending_line: int (exclusive to vtk)
                Int describing ending line for reading nodal values in file.
                Eg. 2456

    Returns
    -------
    snapshots : np.array
        Numpy 2D array containing the snapshots matrix.
    """
    logger.info("Starting choice of file type:")
    if file_type_str == "h5_libmesh":
        logger.info("libMesh/EdgeCFD HDF5 file selected.")
        filenames = snapshot_ingestion_parameters["filenames"]
        dataset = snapshot_ingestion_parameters["dataset"]
        ingestion_function = read_h5_libmesh
        ingestion_parameters = [filenames[0], dataset]
    elif file_type_str == "h5_fenics":
        logger.info("FEniCS HDF5 file selected.")
        filenames = snapshot_ingestion_parameters["filenames"]
        dataset = "vector_0"
        ingestion_function = read_h5_fenics
        ingestion_parameters = [filenames[0], dataset]
    elif file_type_str == "vtk_freefem":
        logger.info("FreeFem++ vtk file selected.")
        filenames = snapshot_ingestion_parameters["filenames"]
        starting_line = snapshot_ingestion_parameters["starting_line"]
        ending_line = snapshot_ingestion_parameters["ending_line"]
        ingestion_function = read_vtk
        ingestion_parameters = [filenames[0], starting_line, ending_line]

    first_snapshot = ingestion_function(*ingestion_parameters)
    rows = first_snapshot.shape[0]
    columns = len(filenames)
    snapshots = np.zeros((rows, columns))
    snapshots[:, 0] = first_snapshot
    for i in range(1, columns):
        ingestion_parameters[0] = filenames[i]
        data = ingestion_function(*ingestion_parameters)
        snapshots[:, i] = data
    return snapshots


def find_nearest(row_vec, matrix):
    """
    Function used to map nearest FE nodes and reorder the snapshot rows.
    Used in the map_dofs function.

    Parameters
    ----------
    row_vec : np.array
        Numpy array containing the snapshot.
    matrix : np.2d_array
        Numpy 2D array containing nodal coordinates.

    Returns
    -------
    np.array
        Numpy 2D array containing the snapshots mapped through the domain.
    """
    dist_array = np.sqrt(
        (matrix[:, 0] - row_vec[0]) ** 2
        + (matrix[:, 1] - row_vec[1]) ** 2
        + (matrix[:, 2] - row_vec[2]) ** 2
    )
    return np.where(dist_array <= np.min(dist_array))[0]


def map_dofs(prefix_dir, prefix_file, num_partitions, mesh, datasets, problem):
    """
    Function used to map nearest FE nodes and reorder the snapshot rows.
    Used in the map_dofs function.

    Parameters
    ----------
    row_vec : np.array
        Numpy array containing the snapshot.
    matrix : np.2d_array
        Numpy 2D array containing nodal coordinates.

    Returns
    -------
    np.array
        Numpy 2D array containing the snapshots mapped through the domain.
    """
    coords_serial = mesh.points
    coords_parallel = np.atleast_2d([0.0, 0.0, 0.0])
    data_parallel = np.atleast_2d([0.0])
    for partition in range(num_partitions):
        filename = (
            prefix_dir
            + "step23"
            + "/"
            + prefix_file
            + "_"
            + str(num_partitions)
            + "_"
            + str(partition).zfill(3)
            + "_00"
            + str(23).zfill(3)
            + ".h5"
        )
        h5_file = h5py.File(filename, "r")
        data = np.array(h5_file["coords"][:])
        data = np.reshape(data, (int(data.shape[0] / 3), 3))
        num_data = np.array(h5_file[datasets][:])
        coords_parallel = np.vstack((coords_parallel, data))
        data_parallel = np.vstack((data_parallel, num_data[:, np.newaxis]))
    data_parallel = data_parallel[1:, :]
    coords_parallel = coords_parallel[1:, :]
    unique_items, indices = np.unique(coords_parallel, return_index=True, axis=0)
    dofs = []
    for row in np.round(coords_serial, 5):
        row2 = find_nearest(row, unique_items)
        dofs.append(row2[0])

    np.save("E:/data/usnccm/total_data/" + problem + "/indices", indices)
    np.save("E:/data/usnccm/total_data/" + problem + "/dofs", dofs)
    np.save(
        "E:/data/usnccm/total_data/" + problem + "/dofs_with_interface",
        coords_parallel.shape[0],
    )
    snapshots = data_parallel[indices]
    snapshots = snapshots[dofs]
    mesh_out = meshio.Mesh(
        mesh.points, mesh.cells_dict, point_data={"test": (snapshots)}
    )
    mesh_out.write("check.vtk")
    print(
        f"index checking: unique_items[dofs] - coords_serial = \
        {np.linalg.norm(unique_items[dofs] - coords_serial, 'fro')/np.linalg.norm(coords_serial, 'fro'):10.3e}"  # pylint: disable=line-too-long
    )
    return indices, dofs, data_parallel.shape[0]
