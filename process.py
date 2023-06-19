import h5py
import numpy as np

from constants import (
    N_VOXELS_L0,
    N_VOXELS_L1,
    N_VOXELS_L2,
    N_VOXELS_L3,
    N_VOXELS_L12,
    N_VOXELS_L13,
    N_VOXELS_L14,
    N_VOXELS,
)


def preprocess(file_name):
    """
    Create 2 arrays: one array for voxel energies and one for the conditon energy.
    Args:
        file_name : name of the HDF5 file
    Returns:
        List of arrays.
    """
    # Read the HDF5 file
    h5_file = h5py.File(file_name)
    incident_energies = np.array(h5_file["incident_energies"])
    showers = np.array(h5_file["showers"])
    # Max incident energy
    max_energy = np.max(incident_energies)
    training_data = []
    training_condition = []
    # Loop over the events and build the ratios
    for event in range(len(incident_energies)):
        current_event = []
        energy_voxels_l0 = showers[event][:N_VOXELS_L0]
        energy_voxels_l1 = showers[event][N_VOXELS_L0 : N_VOXELS_L0 + N_VOXELS_L1]
        energy_voxels_l2 = showers[event][
            N_VOXELS_L0 + N_VOXELS_L1 : N_VOXELS_L0 + N_VOXELS_L1 + N_VOXELS_L2
        ]
        energy_voxels_l3 = showers[event][
            N_VOXELS_L0
            + N_VOXELS_L1
            + N_VOXELS_L2 : N_VOXELS_L0
            + N_VOXELS_L1
            + N_VOXELS_L2
            + N_VOXELS_L3
        ]
        energy_voxels_l12 = showers[event][
            N_VOXELS_L0
            + N_VOXELS_L1
            + N_VOXELS_L2
            + N_VOXELS_L3 : N_VOXELS_L0
            + N_VOXELS_L1
            + N_VOXELS_L2
            + N_VOXELS_L3
            + N_VOXELS_L12
        ]
        energy_voxels_l13 = showers[event][
            N_VOXELS_L0
            + N_VOXELS_L1
            + N_VOXELS_L2
            + N_VOXELS_L3
            + N_VOXELS_L12 : N_VOXELS_L0
            + N_VOXELS_L1
            + N_VOXELS_L2
            + N_VOXELS_L3
            + N_VOXELS_L12
            + N_VOXELS_L13
        ]
        energy_voxels_l14 = showers[event][
            N_VOXELS_L0
            + N_VOXELS_L1
            + N_VOXELS_L2
            + N_VOXELS_L3
            + N_VOXELS_L12
            + N_VOXELS_L13 :
        ]
        energy_sum_l0 = np.sum(energy_voxels_l0)
        energy_sum_l1 = np.sum(energy_voxels_l1)
        energy_sum_l2 = np.sum(energy_voxels_l2)
        energy_sum_l3 = np.sum(energy_voxels_l3)
        energy_sum_l12 = np.sum(energy_voxels_l12)
        energy_sum_l13 = np.sum(energy_voxels_l13)
        energy_sum_l14 = np.sum(energy_voxels_l14)

        total_energy = (
            energy_sum_l0
            + energy_sum_l1
            + energy_sum_l2
            + energy_sum_l3
            + energy_sum_l12
            + energy_sum_l13
            + energy_sum_l14
        )
        # Ratio of voxels energies
        current_event.append(energy_voxels_l0 / energy_sum_l0)
        current_event.append(energy_voxels_l1 / energy_sum_l1)
        current_event.append(energy_voxels_l2 / energy_sum_l2)
        current_event.append(energy_voxels_l3 / energy_sum_l3)
        current_event.append(energy_voxels_l12 / energy_sum_l12)
        current_event.append(energy_voxels_l13 / energy_sum_l13)
        current_event.append(energy_voxels_l14 / energy_sum_l14)
        fraction = total_energy / incident_energies[event]
        if fraction < 6:
            # Etot / Etruth
            # Here it is also / 6 to have everthing in [0,1] as for the low energetic pions this applies
            current_event.append(
                (total_energy / incident_energies[event]) / 6
            )  # MeV / MeV
            # The E layer fractions
            current_event.append(energy_sum_l0 / total_energy)  # MeV / MeV
            current_event.append(energy_sum_l1 / total_energy)  # MeV / MeV
            current_event.append(energy_sum_l2 / total_energy)  # MeV / MeV
            current_event.append(energy_sum_l3 / total_energy)  # MeV / MeV
            current_event.append(energy_sum_l12 / total_energy)  # MeV / MeV
            current_event.append(energy_sum_l13 / total_energy)  # MeV / MeV
            current_event.append(energy_sum_l14 / total_energy)  # MeV / MeV
            training_data.append(np.nan_to_num(np.hstack(current_event)))
            training_condition.append(incident_energies[event] / max_energy)
        else:
            pass
    training_data = np.array(training_data)  # for pions shape: nbShowers, 541
    training_condition = np.array(training_condition)  # shape: nbShowers, 1
    # close the file and delete uncessary arrays
    h5_file.close()
    del incident_energies
    del showers
    return training_data, training_condition


def postprocess(predicted_energies, incident_energies):
    predicted_energies_rescaled = []
    for event in range(len(predicted_energies)):
        total_energy = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 1
            ]
            * incident_energies[event]
        )
        total_energy *= 6
        energy_sum_l0 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 1 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 2
            ]
            * total_energy
        )
        energy_sum_l1 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 2 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 3
            ]
            * total_energy
        )
        energy_sum_l2 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 3 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 4
            ]
            * total_energy
        )
        energy_sum_l3 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 4 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 5
            ]
            * total_energy
        )
        energy_sum_l12 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 5 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 6
            ]
            * total_energy
        )
        energy_sum_l13 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 6 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 7
            ]
            * total_energy
        )
        energy_sum_l14 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 7 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
                + 8
            ]
            * total_energy
        )

        energy_voxels_l0 = predicted_energies[event][:N_VOXELS_L0] * energy_sum_l0
        energy_voxels_l1 = (
            predicted_energies[event][N_VOXELS_L0 : N_VOXELS_L0 + N_VOXELS_L1]
            * energy_sum_l1
        )
        energy_voxels_l2 = (
            predicted_energies[event][
                N_VOXELS_L0 + N_VOXELS_L1 : N_VOXELS_L0 + N_VOXELS_L1 + N_VOXELS_L2
            ]
            * energy_sum_l2
        )
        energy_voxels_l3 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
            ]
            * energy_sum_l3
        )
        energy_voxels_l12 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
            ]
            * energy_sum_l12
        )
        energy_voxels_l13 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
            ]
            * energy_sum_l13
        )
        energy_voxels_l14 = (
            predicted_energies[event][
                N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13 : N_VOXELS_L0
                + N_VOXELS_L1
                + N_VOXELS_L2
                + N_VOXELS_L3
                + N_VOXELS_L12
                + N_VOXELS_L13
                + N_VOXELS_L14
            ]
            * energy_sum_l14
        )
        event_i = []
        event_i.append(energy_voxels_l0)
        event_i.append(energy_voxels_l1)
        event_i.append(energy_voxels_l2)
        event_i.append(energy_voxels_l3)
        event_i.append(energy_voxels_l12)
        event_i.append(energy_voxels_l13)
        event_i.append(energy_voxels_l14)
        event_i = np.concatenate(event_i)
        predicted_energies_rescaled.append(event_i)
    predicted_energies_rescaled = np.array(predicted_energies_rescaled)
    return predicted_energies_rescaled


def load_incident_energies(file_name):
    """
    Returns
    Args:
        file_name : name of the HDF5 file
    Returns:

    """
    # Read the HDF5 file
    h5_file = h5py.File(file_name)
    incident_energies = np.array(h5_file["incident_energies"])
    max_energy = np.max(incident_energies)
    h5_file.close()
    return incident_energies, max_energy
