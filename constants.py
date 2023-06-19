import tensorflow as tf

"""
Directories.
"""
# Directory to load the full simulation dataset.
INIT_DIR = "./dataset"
# Directory to save VAE checkpoints
GLOBAL_CHECKPOINT_DIR = "./checkpoint"
# Directory to save model after conversion to a format that can be used in C++.
CONV_DIR = "./conversion"
# Directory to save validation plots.
VALID_DIR = "./validation"
# Directory to save VAE generated showers.
GEN_DIR = "./generation"

"""
GPU ressources
"""
# GPU identifiers separated by comma, no spaces.
GPU_IDS = "0"
# Maximum allowed memory on one of the GPUs (in GB)
MAX_GPU_MEMORY_ALLOCATION = 32

"""
Experiment constants
"""
# Number of layers considered (for eg for pions eta 0.2 7 layers)
N_LAYERS = 7
# Number of voxels for each layer (this is the case of pions)
N_VOXELS_L0 = 8
N_VOXELS_L1 = 10 * 10
N_VOXELS_L2 = 10 * 10
N_VOXELS_L3 = 5
N_VOXELS_L12 = 15 * 10
N_VOXELS_L13 = 16 * 10
N_VOXELS_L14 = 10
# The total number of voxels in all the layers
N_VOXELS = (
    N_VOXELS_L0
    + N_VOXELS_L1
    + N_VOXELS_L2
    + N_VOXELS_L3
    + N_VOXELS_L12
    + N_VOXELS_L13
    + N_VOXELS_L14
)
# The total number ratios N_VOXELS + 1 (Etot/Etruth) + N_LAYERS
ORIGINAL_DIM = N_VOXELS + 1 + N_LAYERS

"""
Model & training parameters
"""
INTERMEDIATE_DIMS = [1500, 1000, 500, 100]
LATENT_DIM = 50
KERNEL_INITIALIZER = "RandomNormal"
BIAS_INITIALIZER = "Zeros"
ACTIVATION = tf.keras.layers.LeakyReLU()
ACTIVATION_ETOT_DIV_ETRUTH = "sigmoid"
BATCH_SIZES = [1000, 500, 250, 1000, 1000, 1000, 500, 5000]
LEARNIN_RATES = [0.01, 0.001, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
VALIDATION_SPLIT = 0.15
EPOCHS = 100000
# Number of epochs with no improvement after which training will be stopped
PATIENCE = 10
