from mrcnn.config import Config
from mrcnn import model as modellib, utils
import os

class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = 'custom'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2 # Background + num classes


     # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # len(train_batch)

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9




config = CustomConfig()
config.display()


DEFAULT_LOGS_DIR = os.path.join(os.getcwd(), "logs")

if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)


model = modellib.MaskRCNN(mode="training",
                           config=config,
                           model_dir=DEFAULT_LOGS_DIR)


weights_path = "/workspace/mask_rcnn_coco.h5"
model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
