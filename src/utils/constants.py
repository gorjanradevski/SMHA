# Dataset specifics
PAD_ID = 0
UNK_ID = 1
min_unk_sub = 4

# Imagenet specifics
WIDTH = 224
HEIGHT = 224
NUM_CHANNELS = 3
VGG_MEAN = [123.68, 116.78, 103.94]

# Pascal sentences splits
pascal_train_size = 0.8
pascal_val_size = 0.1

# Metrics
inference_for_recall_at = [1, 5, 10]

# Model specifics
embedding_size = 300
decay_rate_epochs = 3
