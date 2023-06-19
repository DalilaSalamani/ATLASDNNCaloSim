from argparse import ArgumentParser
import h5py
import numpy as np
import tensorflow as tf

from model import VAE
from process import postprocess, load_incident_energies
from constants import (
    INTERMEDIATE_DIMS,
    LATENT_DIM,
    ORIGINAL_DIM,
    KERNEL_INITIALIZER,
    BIAS_INITIALIZER,
    ACTIVATION,
    ACTIVATION_ETOT_DIV_ETRUTH,
    GLOBAL_CHECKPOINT_DIR,
    GEN_DIR,
)


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--version", type=int)
    argument_parser.add_argument("--epoch", type=int)
    argument_parser.add_argument(
        "--file-name", type=str, default="dataset/dataset_1_pions_1.hdf5"
    )
    args = argument_parser.parse_args()
    return args


def main():
    # Parse arguments.
    args = parse_args()
    version = args.version
    epoch = args.epoch
    file_name = args.file_name
    # Instantiate the VAE model
    vae = VAE(
        original_dim=ORIGINAL_DIM,
        intermediate_dim1=INTERMEDIATE_DIMS[0],
        intermediate_dim2=INTERMEDIATE_DIMS[1],
        intermediate_dim3=INTERMEDIATE_DIMS[2],
        intermediate_dim4=INTERMEDIATE_DIMS[3],
        latent_dim=LATENT_DIM,
        kernel_initializer=KERNEL_INITIALIZER,
        bias_initializer=BIAS_INITIALIZER,
        activation=ACTIVATION,
        activ_frac_etot_etruth=ACTIVATION_ETOT_DIV_ETRUTH,
        optimizer=tf.optimizers.Adam(),
        w_reco=ORIGINAL_DIM,
    )
    # Load the weights
    vae.vae.load_weights(f"{GLOBAL_CHECKPOINT_DIR}/VAE-V{version}-{epoch}.h5")
    # Load the incident energies
    incident_energies, max_energy = load_incident_energies(file_name)
    # Build the generator's input : LATENT_DIM-Gaussians, condition vector of energies
    z_n_gauss = np.random.normal(
        loc=0, scale=1, size=(len(incident_energies), LATENT_DIM)
    )
    z = np.column_stack((z_n_gauss, incident_energies / max_energy))
    # Predict energies
    predicted_energies = vae.decoder.predict(z)
    # Rescale back the energies
    predicted_energies = postprocess(predicted_energies, incident_energies)
    print(f"Generation completed of {len(predicted_energies)} showers.")
    # Save the hdf5 file (same format as the Geant4 dataset)
    dataset_file = h5py.File(f"{GEN_DIR}/VAE_dataset_1_pions_1.hdf5", "w")
    dataset_file.create_dataset(
        "incident_energies",
        data=incident_energies.reshape(len(incident_energies), -1),
        compression="gzip",
    )
    dataset_file.create_dataset(
        "showers",
        data=predicted_energies.reshape(len(predicted_energies), -1),
        compression="gzip",
    )
    dataset_file.close()
    print(f"The HDF5 file {GEN_DIR}/VAE_dataset_1_pions_1.hdf5 is saved.")


if __name__ == "__main__":
    exit(main())
