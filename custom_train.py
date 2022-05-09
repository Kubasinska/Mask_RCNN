import os
import sys
import json
#import datetime
import numpy as np
import skimage.draw
# import imgaug



desired_weights = 'coco' # coco, last, path


#%%
# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd())

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DATASET_PATH = os.path.join(ROOT_DIR, 'danno')

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

#%%


#%%

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2 # Background + num classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100                                                                                # len(train_batch)

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

#%%

            
#%%
            
class CustomDataset(utils.Dataset):
    def load_VIA(self, dataset_dir, subset, hc=False):
        """Load the surgery dataset from VIA.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or predict
        """
        # Add classes. We have only one class to add.
        self.add_class("danno", 1, "ammaccatura")
        self.add_class("danno", 2, "vernice")


        
#        if hc is True:
#            for i in range(1,14):
#                self.add_class("surgery", i, "{}".format(i))
#            self.add_class("surgery", 14, "arm")

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
        with open(os.path.join(dataset_dir, "via_region_data.json"), "r") as file:
            annotations = json.load(file).decode("utf-8")

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
                "danno",
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
        image_info = self.image_info[image_id]
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
        #"name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
            if  p['danno'] == 'ammaccatura':
                class_ids[i] = 1
            elif p['danno'] == 'vernice':
                class_ids[i] = 2

        
                
            #assert code here to extend to other labels
        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids            
    
            
#%%
        
# augmentation = imgaug.augmenters.Sometimes(0.5, [
#                     imgaug.augmenters.Fliplr(0.5),
#                     imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0)),
#                     imgaug.augmenters.Flipud(0.5),
#                     imgaug.augmenters.EdgeDetect(alpha=(0, 0.7)),
#                     imgaug.augmenters.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
#                     imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#                     imgaug.augmenters.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
#                     imgaug.augmenters.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
#                     imgaug.augmenters.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
#                     imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
#                     imgaug.augmenters.Affine(
#                                             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
#                                             translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
#                                             rotate=(-5, 5), # rotate by -45 to +45 degrees
#                                             shear=(-5, 5), # shear by -16 to +16 degrees
#                                             )
#                     ])    
    




#%%
            
            
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_VIA(DATASET_PATH, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_VIA(DATASET_PATH, "val")
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
                # augmentation = augmentation)
    
#%%

config = CustomConfig()
config.display()

#%%

model = modellib.MaskRCNN(mode="training", 
                          config=config,
                          model_dir=DEFAULT_LOGS_DIR)

#%%

# Select weights file to load
if desired_weights == "coco":
    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
elif desired_weights == "last":
    # Find last trained weights
    weights_path = model.find_last()
elif desired_weights == "imagenet":
    # Start from ImageNet trained weights
    weights_path = model.get_imagenet_weights()
else:
    weights_path = desired_weights
    
# Load weights
print("Loading weights ", weights_path)
if desired_weights == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
else:
    model.load_weights(weights_path, by_name=True)

#%%
    
train(model)

