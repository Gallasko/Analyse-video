import os
import sys
import random
import math
import numpy as np
import tkinter
from six import b
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time  
import scipy

from PIL import Image

import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
#import coco
from .coco import CocoConfig

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from scipy.cluster.vq import whiten 
from scipy.cluster.vq import kmeans 
import pandas as pd 

import cv2
import threading
import asyncio

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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

print(tf.executing_eagerly())

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin:ymax+1, xmin:xmax+1]

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

rgb_weights = [0.2989, 0.5870, 0.1140]

mnistClassNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def getRcnnResult(model, image, resultModel):
    t1 = time.perf_counter()
    res = model.detect(image)
    resultModel[0] = res
    t2 = time.perf_counter()
    
    print(f"RCNN Result took: {t2 - t1:0.4f} seconds")

def getBodyPixResult(model, image, partNames, resultModel):
    t1 = time.perf_counter()

    bpResult = model.predict_single(image)
    tempResult = bpResult.get_mask(threshold=0.65)
    fullBodyMask = tempResult.numpy()
    tempResult2 = bpResult.get_part_mask(part_names=partNames, mask=fullBodyMask)
    partialBodyMask = tempResult2

    resultModel[0] = [fullBodyMask, partialBodyMask]

    t2 = time.perf_counter()

    print(f"Body Pix Result took: {t2 - t1:0.4f} seconds")

def combineMasks(mask, fullBodyMask, partialBodyMask, image, box, place, resultMask):
    #t1 = time.perf_counter()
    
    # full body

    #fullBodyMask[:, :, 0] = np.where(mask == 1, bodyMask[:, :, 0], 0)
    #fullBodyMaskInstance = np.where(mask == 1, fullBodyMask[:, :, 0], 0)
    #fullBodyMaskInstance = cv2.bitwise_and(fullBodyMask[:, :, 0], mask) #Does not work
    #fullBodyMaskInstance = fullBodyMask[:, :, 0].copy()
    #fullBodyMaskInstance[mask == 0] = 0
    #fullBodyMaskInstance[mask != 0] = fullBodyMask[mask != 0]

    #mask = (obstacle == 0)
    #fullBodyMaskInstance = fullBodyMask[:, :, 0]
    #np.putmask(fullBodyMaskInstance, (mask == 1), 1)

    #final = np.full_like(image, 1)

    #for c in range(3):
    #    final[:, :, c] = np.where(fullBodyMaskInstance == 1, image[:, :, c], 0)

    # part only
    partialBodyMaskInstance = np.where(mask == 1, partialBodyMask[:, :, 0], 0)
    #partialBodyMaskInstance = cv2.bitwise_and(partialBodyMask, fullBodyMaskInstance[:, :, 0])

    #partialBodyMaskInstance = partialBodyMask[:, :, 0]
    #np.putmask(partialBodyMaskInstance, (fullBodyMaskInstance == 1), 1)

    is_all_zero = np.all((partialBodyMaskInstance == 0))
    if not is_all_zero:
        #body part extraction
        final2 = np.full_like(image, 1)

        for c in range(3):
            final2[:, :, c] = np.where(partialBodyMaskInstance == 1, image[:, :, c], 0)
        #final2 = cv2.bitwise_and(image, partialBodyMaskInstance)
        #final2 = image[:, :, :]
        #for c in range(3):
        #    np.putmask(final2[:, :, c], (partialBodyMaskInstance == 1), 1)

        #t2 = time.perf_counter()

        #print(f"Mask combination took: {t2 - t1:0.4f} seconds")
        
        croppedImage = bbox2(final2)

        #grayscale_image = np.dot(croppedImage[...,:3], rgb_weights)
        grayscale_image = croppedImage #TODO make it B&W
        #t3 = time.perf_counter()

        #print(f"Mask crop took: {t3 - t2:0.4f} seconds")

        #ar = np.asarray(final2)
        #shape = ar.shape
        #ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

        pil_img = Image.fromarray(np.uint8(final2)).convert('RGB')

        img = pil_img.copy()
        #img.thumbnail((100, 100))

        # Reduce colors (uses k-means internally)
        paletted = img.convert('P', palette=Image.ADAPTIVE, colors=4)

        # Find the color that occurs most often
        palette = paletted.getpalette()

        #print(palette)
        #color_counts = sorted(paletted.getcolors(), reverse=True)
        #palette_index = color_counts[0][1]
        #dominant_color = palette[palette_index*3:palette_index*3+3]


        #codes, dist = scipy.cluster.vq.kmeans(ar, 4)
        #print('cluster centres:\n', codes)

        """
        #predominant color extraction
        dominant_colors = [] 
        red = [] 
        green = [] 
        blue = [] 
        for row in croppedImage: 
            for temp_r, temp_g, temp_b in row: 
                red.append(temp_r) 
                green.append(temp_g) 
                blue.append(temp_b) 
        
        if (len(red) > 0 and len(green) > 0 and len(blue) > 0):
            image_df = pd.DataFrame({'red' : red, 
                                    'green' : green, 
                                    'blue' : blue}) 

            if(len(image_df['red']) > 4 and len(image_df['blue']) > 4 and len(image_df['green']) > 4):
                image_df['scaled_color_red'] = whiten(image_df['red']) 
                image_df['scaled_color_blue'] = whiten(image_df['blue']) 
                image_df['scaled_color_green'] = whiten(image_df['green']) 
                
                cluster_centers, _ = kmeans(image_df[['scaled_color_red', 
                                                    'scaled_color_blue', 
                                                    'scaled_color_green']], 4) 
                
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

       
        """
        #t4 = time.perf_counter()
        #print(f"Color find took: {t4 - t3:0.4f} seconds")

        #resultMask[0] = [place, {"image": final2, "color": dominant_colors, "box": box, "gray": grayscale_image}]
        resultMask[0] = [place, {"image": final2, "color": [np.asarray(palette[i * 3:i * 3 + 3]) / 255.0 for i in range(4)], "box": box, "gray": grayscale_image}]
    
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

        self.initModels()

    def initModels(self):
        # Create model object in inference mode.
        self.rcnnModel = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.rcnnModel.load_weights(COCO_MODEL_PATH, by_name=True)
        

        #self.bpModel = load_model("bodyPixModel", 16, 'mobilenet_v1')
        #self.bpModel = load_model("BodyPixMobileNet100", 16, 'mobilenet_v1')
        self.bpModel = load_model("BodyPixResNet50", 16, 'resnet50')

        #self.bpModel = load_model(download_model(BodyPixModelPaths.MOBILENET_RESNET50_FLOAT_STRIDE_16))
        #dl = download_model(BodyPixModelPaths.MOBILENET_FLOAT_100_STRIDE_16)
        #self.bpModel = load_model(dl)

        """
        #FashionMnist Training
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

        self.mnistModel.fit(self.train_images, self.train_labels, epochs=1000)

        self.mnistModel.save('mnistModel')
        """

        self.mnistModel = tf.keras.models.load_model('mnistModel')

        self.probability_model = tf.keras.Sequential([self.mnistModel, tf.keras.layers.Softmax()])
    
    def forward(self, image):
        bpPartNames = [['left_upper_arm_front', 'left_upper_arm_back', 'right_upper_arm_front', 'right_upper_arm_back', 'left_lower_arm_front', 'left_lower_arm_back', 'right_lower_arm_front', 'right_lower_arm_back', 'torso_front', 'torso_back'], 
                        ['left_upper_leg_front', 'left_upper_leg_back', 'right_upper_leg_front', 'right_upper_leg_back', 'left_lower_leg_front', 'left_lower_leg_back', 'right_lower_leg_front', 'right_lower_leg_back']]

        # Run detection
        #rcnnTask = asyncio.create_task(getRcnnResult(self.rcnnModel, [image]))
        #bodyPixTaskTop = asyncio.create_task(getBodyPixResult(self.bpModel, image, bpPartNames[0]))
        #bodyPixTaskBottom = asyncio.create_task(getBodyPixResult(self.bpModel, image, bpPartNames[1]))

        rcnnResults = [None]
        bpResultTop = [None]
        bpResultBottom = [None]

        threadRcnn = threading.Thread(target = getRcnnResult, args=(self.rcnnModel, [image], rcnnResults))
        threadBodyPixTop = threading.Thread(target = getBodyPixResult, args=(self.bpModel, image, bpPartNames[0], bpResultTop))
        threadBodyPixBottom = threading.Thread(target = getBodyPixResult, args=(self.bpModel, image, bpPartNames[1], bpResultBottom))

        threadRcnn.start()
        threadBodyPixTop.start()
        threadBodyPixBottom.start()

        bpResult = []

        t3 = time.perf_counter()
        #rcnnResults = await self.rcnnModel.detect(images, verbose=1)
        #rcnnResults, bpResultTop, bpResultBottom = await asyncio.gather(rcnnTask, bodyPixTaskTop, bodyPixTaskBottom)
        #rcnnResults = await rcnnTask
        #bpResultTop = await bodyPixTaskTop
        #bpResultBottom = await bodyPixTaskBottom

        threadRcnn.join()
        threadBodyPixTop.join()
        threadBodyPixBottom.join()

        bpResult.append([bpResultTop[0], "Top"])
        bpResult.append([bpResultBottom[0], "Bottom"])
        
        # Visualize results
        r = rcnnResults[0][0]

        #bboxes = utils.extract_bboxes(r['masks'])
        bboxes = r["rois"]

        t4 = time.perf_counter()

        print(f"Neural Network processing took: {t4 - t3:0.4f} seconds")

        #visualize.draw_boxes(image, r['rois'], masks=)
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        t5 = time.perf_counter()

        resultList = []
        nbThread = 0
        for bpRes in bpResult:
            
            nbBox = 0

            fullBodyMask, partialBodyMask = bpRes[0]
            
            for x in range(len(r['class_ids'])):
                if r['class_ids'][x] == 1: # Person
                    result = [None]
                    ThreadMask = threading.Thread(target = combineMasks, args=(r['masks'][:,:,x], fullBodyMask, partialBodyMask, image, bboxes[nbBox], bpRes[1], result))
                    resultList.append([result, ThreadMask])
                    resultList[nbThread][1].start()
                    nbThread += 1
                    #taskList.append(asyncio.create_task(combineMasks(r['masks'][:,:,x], fullBodyMask, partialBodyMask, image, bboxes[nbBox], bpRes[1])))
                nbBox += 1
            

            #for res in bodyMasks:
                #res['prediction']
                #res['best_prediction']
                #res['rescaled_image'] = cv2.resize(res['gray'], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

                #arr = np.asarray(res['rescaled_image'][None, ...])
                #res['prediction'] = self.probability_model.predict(arr)
                #res['best_prediction'] = np.argmax(res['prediction'])

            #forwardResults.append(bodyMasks)

        Top = []
        Bottom = []

        for res in resultList:
            res[1].join()

        for res in resultList:
            if res[0][0] is not None:
                if res[0][0][0] == "Top":
                    Top.append(res[0][0][1])
                elif res[0][0][0] == "Bottom":
                    Bottom.append(res[0][0][1])

        t6 = time.perf_counter()

        print(f"Mask Combination processing took: {t6 - t5:0.4f} seconds")

        return [Top, Bottom]