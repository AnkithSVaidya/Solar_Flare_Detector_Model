"""

This file stores all relevant constants relating to the Solar Flare Project

Instead of defining constants inside classes, they should be defined here.


"""

####
# PREPROCESSING CONSTANTS
####

# The seed to use for all random processes- ensures repeatability
SEED = 42

# The percentage of training data to use for validation
VALIDATION_SPLIT = .2

# The resolution of the image (pixels)
RESOLUTION = 128 

# The cutoff of what defines a Solar Flare (1e-9 is no flare, 1e-7 - 1e-5 are flares)
CUTOFF = 1e-6


####
# TRAINING CONSTANTS
####

# The batch size for training
BATCH_SIZE = 32                                 # 32 was found to be best at loading the GPU

# The number of epochs for training
NUM_EPOCHS = 40
NUM_EPOCHS_CONTINUUM = 30

# The learning rate for magnetogram CNN
LEARNING_RATE = 5e-5
