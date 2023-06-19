import os
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import EarlyStopping
import warnings

from gpu_limiter import GPULimiter
from process import preprocess
from model import VAE
from constants import (
    GPU_IDS,
    MAX_GPU_MEMORY_ALLOCATION,
    INTERMEDIATE_DIMS,
    LATENT_DIM,
    ORIGINAL_DIM,
    KERNEL_INITIALIZER,
    BIAS_INITIALIZER,
    BATCH_SIZES,
    LEARNIN_RATES,
    VALIDATION_SPLIT,
    EPOCHS,
    ACTIVATION,
    ACTIVATION_ETOT_DIV_ETRUTH,
    GLOBAL_CHECKPOINT_DIR,
    PATIENCE,
)


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--file-name", type=str, default="dataset/dataset_1_pions_1.hdf5"
    )
    argument_parser.add_argument("--gpu-ids", type=str, default=GPU_IDS)
    argument_parser.add_argument(
        "--max-gpu-memory-allocation", type=int, default=MAX_GPU_MEMORY_ALLOCATION
    )
    args = argument_parser.parse_args()
    return args


def main():
    """
    # Filter out RuntimeWarnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Filter out TensorFlow warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    """

    # Parse arguments.
    args = parse_args()
    file_name = args.file_name
    gpu_ids = args.gpu_ids
    max_gpu_memory_allocation = args.max_gpu_memory_allocation

    # Set GPU memory limits.
    GPULimiter(_gpu_ids=gpu_ids, _max_gpu_memory_allocation=max_gpu_memory_allocation)()

    # Data loading/preprocessing
    # The preprocess function reads the data and performs preprocessing and encoding for the values of energy,
    energies_train, cond_e_train = preprocess(file_name)

    # Test with different batch sizes and learning rates
    for test_version in range(8):
        print(f" ....... Currently test_version {test_version} ....... ")

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
            optimizer=tf.optimizers.Adam(LEARNIN_RATES[test_version]),
            w_reco=ORIGINAL_DIM,
        )

        # Load previous best model
        if test_version > 1:
            # Get the list of files in the folder
            files = os.listdir(GLOBAL_CHECKPOINT_DIR)
            if len(files) > 1:
                # Sort the files based on their creation time
                files.sort(
                    key=lambda x: os.path.getctime(
                        os.path.join(GLOBAL_CHECKPOINT_DIR, x)
                    )
                )
                # Get the name of the last added file
                last_added_file = files[-1]
                # Load the last added file from the previous test_version
                vae.vae.load_weights(f"{GLOBAL_CHECKPOINT_DIR}/{last_added_file}")

        # The model checkpoint callback
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            "%s/VAE-V%s-{epoch:02d}.h5" % (GLOBAL_CHECKPOINT_DIR, test_version),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1,
        )

        # The early stopping callback
        callback_early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            verbose=1,
            mode="auto",
        )

        # Train the VAE model
        noise = np.random.normal(0, 1, size=(energies_train.shape[0], LATENT_DIM))
        _ = vae.vae.fit(
            [energies_train, cond_e_train, noise],
            [energies_train],
            shuffle=True,
            verbose=0,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            batch_size=BATCH_SIZES[test_version],
            callbacks=[callback_checkpoint, callback_early_stopping],
        )


if __name__ == "__main__":
    exit(main())
