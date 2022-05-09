import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import json
import sys
import skimage.draw

ROOT_DIR = os.path.abspath(os.getcwd())
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import model as modellib, utils



# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
#
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)

DATASET_PATH = os.path.join(ROOT_DIR, 'name')
OBJ_IDX_MAP_PATH = os.path.join(DATASET_PATH, 'obj_idx_map.json')
with open(OBJ_IDX_MAP_PATH) as f:
    obj_idx_map = json.load(f)

print(obj_idx_map)
obj_idx = list(obj_idx_map.keys())
obj_name = list(obj_idx_map.values())
print(obj_idx)
print(obj_name)

# init training folders
processed_dir = os.path.join(os.getcwd(), 'name')
train_dir = os.path.join(processed_dir, 'train')
train_images_n = len(os.listdir(train_dir))-1

val_dir = os.path.join(processed_dir, 'val')
val_images_n = len(os.listdir(val_dir))-1

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
    STEPS_PER_EPOCH = train_images_n
    # len(train_batch)

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class CustomDataset(utils.Dataset):
    def load_VIA(self, dataset_dir, subset, class_idx,class_names):
        """Load the custom dataset from VIA.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or predict
        """

        for i in range(len(class_idx)):
            self.add_class("name", class_idx[i], class_names[i])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {name:'a'},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir,
                                                  "via_region_data.json")))

        annotations = list(annotations.values())  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)

            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                names = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                names = [r['region_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "name",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a surgery dataset image, delegate to parent class.
        # image_info = self.image_info[image_id]
        # if image_info["source"] != "object":
        #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        #        print(info["names"])

        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        # In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        for i, p in enumerate(class_names):
            # "name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
            if p['name'] == 'chips_can':
                class_ids[i] = 1
            elif p['name'] == 'master_chef_can':
                class_ids[i] = 2

            # assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids




def train(model, class_idx, class_names):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_VIA(DATASET_PATH, "train", class_idx, class_names)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_VIA(DATASET_PATH, "val", class_idx, class_names)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.

    print("Training network heads")
    model.train(dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')
                # augmentation=augmentation)












config = CustomConfig()
config.display()
model = modellib.MaskRCNN(mode="training",
                           config=config,
                           model_dir=DEFAULT_LOGS_DIR)
weights_path = COCO_WEIGHTS_PATH
# # weights_path = model.find_last()
model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
print("start train model")
train(model, class_idx=obj_idx,class_names=obj_name)
