# index2word constants
PAD_ID = 0
UNK_ID = 1


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


# Training specifics
# Terminate training if recall_at k is not at least
recall_at_least_pascal = {1: 0.3, 5: 0.45, 10: 0.6}
recall_at_least_flickr = {1: 0.15, 5: 0.3, 10: 0.45}
