import os
import sys
import random
import math
import numpy as np
import tkinter
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
import coco

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from scipy.cluster.vq import whiten 
from scipy.cluster.vq import kmeans 
import pandas as pd 

import cv2

from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

print(tf.__version__)
tf.compat.v1.enable_eager_execution()
print(tf.executing_eagerly())

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(COCO_MODEL_PATH)
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

print(tf.executing_eagerly())

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax+1, xmin:xmax+1]

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class ImagePipeline():
    #Definition#
    def __init__(self):
        """
        Phase d'initialisation du Pipeline d'image
        """

        self.config = InferenceConfig()
        self.config.display()

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.rcnnModelClassNames = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                                    'bus', 'train', 'truck', 'boat', 'traffic light',
                                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                                    'teddy bear', 'hair drier', 'toothbrush']

        self.rgb_weights = [0.2989, 0.5870, 0.1140]

        self.mnistClassNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        self.initModels()

    def initModels(self):
        # Create model object in inference mode.
        self.rcnnModel = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.rcnnModel.load_weights(COCO_MODEL_PATH, by_name=True)

        self.bpModel = load_model("bodyPixModel", 16, 'mobilenet_v1')

        fashion_mnist = tf.keras.datasets.fashion_mnist

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        self.mnistModel = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        self.mnistModel.compile(optimizer='adam',
                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                metrics=['accuracy'])

        self.mnistModel.fit(self.train_images, self.train_labels, epochs=10)

        self.probability_model = tf.keras.Sequential([self.mnistModel, tf.keras.layers.Softmax()])
    
    def forward(self, image):
        # Run detection
        rcnnResults = self.rcnnModel.detect([image], verbose=1)

        # Visualize results
        r = rcnnResults[0]
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

        bpPartNames = [['left_upper_arm_front', 'left_upper_arm_back', 'right_upper_arm_front', 'right_upper_arm_back', 'left_lower_arm_front', 'left_lower_arm_back', 'right_lower_arm_front', 'right_lower_arm_back', 'torso_front', 'torso_back'], 
                      ['left_upper_leg_front', 'left_upper_leg_back', 'right_upper_leg_front', 'right_upper_leg_back', 'left_lower_leg_front', 'left_lower_leg_back', 'right_lower_leg_front', 'right_lower_leg_back']]

        forwardResults = []

        for partNames in bpPartNames:
            bpResult = self.bpModel.predict_single(image)

            tempResult = bpResult.get_mask(threshold=0.65)

            fullBodyMask = tempResult.numpy()

            tempResult2 = bpResult.get_part_mask(part_names=partNames, mask=fullBodyMask)

            partialBodyMask = tempResult2

            bodyMasks = []
            
            for x in range(len(r['class_ids'])):
                if r['class_ids'][x] == 1: # Person
                    mask = r['masks'][:,:,x]    

                    # full body

                    #fullBodyMask[:, :, 0] = np.where(mask == 1, bodyMask[:, :, 0], 0)
                    fullBodyMaskInstance = np.where(mask == 1, fullBodyMask[:, :, 0], 0)

                    final = np.full_like(image, 1)

                    for c in range(3):
                        final[:, :, c] = np.where(fullBodyMaskInstance == 1, image[:, :, c], 0)

                    # part only
                    partialBodyMaskInstance = np.where(fullBodyMaskInstance == 1, partialBodyMask[:, :, 0], 0)

                    is_all_zero = np.all((partialBodyMaskInstance == 0))
                    if not is_all_zero:
                        #body part extraction
                        final2 = np.full_like(image, 1)

                        for c in range(3):
                            final2[:, :, c] = np.where(partialBodyMaskInstance == 1, image[:, :, c], 0)

                        croppedImage = bbox2(final2)

                        grayscale_image = np.dot(croppedImage[...,:3], self.rgb_weights)

                        #predominant color extraction
                        red = [] 
                        green = [] 
                        blue = [] 
                        for row in croppedImage: 
                            for temp_r, temp_g, temp_b in row: 
                                red.append(temp_r) 
                                green.append(temp_g) 
                                blue.append(temp_b) 
                        
                        image_df = pd.DataFrame({'red' : red, 
                                                'green' : green, 
                                                'blue' : blue}) 
                        
                        image_df['scaled_color_red'] = whiten(image_df['red']) 
                        image_df['scaled_color_blue'] = whiten(image_df['blue']) 
                        image_df['scaled_color_green'] = whiten(image_df['green']) 
                        
                        cluster_centers, _ = kmeans(image_df[['scaled_color_red', 
                                                            'scaled_color_blue', 
                                                            'scaled_color_green']], 4) 
                        
                        dominant_colors = [] 
                        
                        red_std, green_std, blue_std = image_df[['red', 
                                                                'green', 
                                                                'blue']].std() 
                        
                        for cluster_center in cluster_centers: 
                            red_scaled, green_scaled, blue_scaled = cluster_center 
                            dominant_colors.append(( 
                                red_scaled * red_std / 255, 
                                green_scaled * green_std / 255, 
                                blue_scaled * blue_std / 255
                            ))

                        modelResults = {"image": grayscale_image, "color": [dominant_colors]}
                        bodyMasks.append(modelResults)

            for res in bodyMasks:
                res['rescaled_image'] = cv2.resize(res['image'], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

                arr = np.asarray(res['rescaled_image'][None, ...])
                res['prediction'] = self.probability_model.predict(arr)
                res['best_prediction'] = np.argmax(res['prediction'])

            forwardResults.append(bodyMasks)
        return forwardResults