import os
import random
import json
import io
import csv
import cv2
import numpy as np
import matplotlib

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

val_rate = 0.1
desired_resolution = (1024,1024)
random.seed(0)

# get raw data dir
data_dir = os.path.join(os.getcwd(), 'data')

# init training folders
processed_dir = os.path.join(os.getcwd(), 'name')
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

train_dir = os.path.join(processed_dir, 'train')
train_json_path = os.path.join(train_dir, "via_region_data.json")
if not os.path.isdir(train_dir):
    os.makedirs(train_dir)

val_dir = os.path.join(processed_dir, 'val')
val_json_path = os.path.join(val_dir, "via_region_data.json")
if not os.path.isdir(val_dir):
    os.makedirs(val_dir)

obj_idx_map_path = os.path.join(processed_dir, "obj_idx_map.json")

# init processed data
obj_idx_map = {}


train_mask_paths = []
val_mask_paths = []

train_label = {}
validation_label = {}

for root, dirs, files in os.walk(data_dir):
    # print(root)
    if root.split('/')[-1] == "masks":

        # mapping objects
        obj_idx = int(root.split('/')[-2][:3])
        name = root.split('/')[-2][4:]
        obj_idx_map[obj_idx] = name
        print(len(files))

        # get validation data index
        val_num = int(val_rate * len(files))
        val_idx = random.sample(list(range(len(files))), val_num)

        img_dir = os.path.join(*root.split('/')[:-1])
        print(img_dir)

        # here get mask to process
        for idx, file in enumerate(files):
            mask_path = os.path.join(root, file)

            img_path = "/" + os.path.join(img_dir, file.replace("_mask.pbm", ".jpg"))
            img = cv2.imread(img_path)
            # cv2.imshow('img', img)
            img_resized = cv2.resize(img, dsize=desired_resolution,interpolation = cv2.INTER_AREA)
            # cv2.imshow('img resized', img_resized)
            print(img_path)

            size = os.path.getsize(mask_path)
            label_name = file.replace("_mask.pbm", "") + name + ".jpg" + str(size)
            print(label_name)

            new_image_name = file.replace("_mask.pbm", "") + "_" + name + ".jpg"
            print(new_image_name)

            # get region
            regions = []
            mask = cv2.imread(mask_path).astype(np.uint8)
            # cv2.imshow('mask', mask)
            mask_resized = cv2.resize(mask, dsize=desired_resolution, interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('gray', gray)
            contours, hierarchy = cv2.findContours(image=gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area > 400 and area < 500000:
                    print('good')
                    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    # cv2.drawContours(img_resized, c, -1, (0, 255, 0), 10)

                    obj_contour = np.array(c)
                    obj_contour = obj_contour.reshape(obj_contour.shape[0],obj_contour.shape[2])
                    obj_contour_x = obj_contour[:, 0]
                    obj_contour_y = obj_contour[:, 1]

                    mask_label = {
                        "shape_attributes":{
                            "name": "polygon",
                            "all_points_x": list(obj_contour_x),
                            "all_points_y": list(obj_contour_y)
                        },
                        "region_attributes": {
                            "name": name
                        }
                    }

                    regions.append(mask_label)


                    # cv2.imshow('mask', mask)

                    # cv2.imshow('object contour',img_resized)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()



            # split train/validation paths
            if idx in val_idx:
                # val_mask_paths.append(mask_path)
                new_image_path = os.path.join(val_dir, new_image_name)
                validation_label[label_name]= {
                    "filename":new_image_name,
                    "size":size,
                    "regions":regions
                }




            else:
                # train_mask_paths.append(mask_path)
                new_image_path = os.path.join(train_dir, new_image_name)
                train_label[label_name]= {
                    "filename":new_image_name,
                    "size":size,
                    "regions":regions
                }

            cv2.imwrite(new_image_path, img_resized)






with open(val_json_path, 'w') as outfile:
    json.dump(validation_label, outfile, default=np_encoder)

with open(train_json_path, 'w') as outfile:
    json.dump(train_label, outfile, default=np_encoder)

with open(obj_idx_map_path, 'w') as outfile:
    json.dump(obj_idx_map, outfile, default=np_encoder)
# print(obj_idx_map)

