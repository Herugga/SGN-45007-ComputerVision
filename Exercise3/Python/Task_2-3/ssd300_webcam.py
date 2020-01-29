from pathlib import Path
import time
from tensorflow.keras import backend as K
import numpy as np
from models.keras_ssd300 import ssd_300
import cv2
from skimage.transform import resize as imresize

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


############################## MODEL ##########################################
# Copyright 2018 Pierluigi Ferrari (SSD keras port)

# Set the network image size.
img_height = 300
img_width = 300

# Build the Keras model
K.clear_session() # Clear previous models from memory.
model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], 
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=10,
                nms_max_output_size=400)

# Load the trained weights into the model.
weights_path = Path('weights/VGG_VOC0712_SSD_300x300_iter_120000.h5')
model.load_weights(str(weights_path), by_name=True)


classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

###############################################################################



##################################### TASK ####################################
#
# NOTE: The video player can be stopped by selecting the video player window
#       and pressing 'q'
#
# Your task:
#   1. Read a frame from camera using cam.read
#
#   2. Mirror the frame using cv2.flip
#
#   3. Make a resized copy of the frame for the network.
#      Use the imported function "imresize" and
#      size (img_height, img_width) (defined above).
#      We also want to preserve the original range (0-255),
#      therefore set the additional parameter "preserve_range"
#      to True.
#
#   4. You need an additional first dimension (batch_size, now 1)
#      to feed the image to the network. This can be achieved with:
#      image = np.array([image])
#
#   5. Use the input image to get predictions from the network.
#      Predictions are in the following form:
#      [class_id, confidence, xmin, ymin, xmax, ymax]
#      These predictions should also be thresholded according to
#      the prediction confidence. Anything above 0
#      (i.e omit 0 confidence values) should be used as a valid prediction.
#
#   6. The bounding box coordinates are for the resized image
#      (size: img_height, img_width). Transform these coordinates to the original
#      frame by multiplying x coordinates with the ratio of original and reshaped widths,
#      and y coordinates with the ratio of heights.
#      Remember to also convert these to integer-type.
#
#   7. Use cv2.rectangle to insert a bounding box to the original frame.
#
#   8. Use cv2.putText to insert the predicted class label to the bounding box.
#      Class labels are defined above, and can be indexed with the class_id
#      in the predictions.
#
###############################################################################


# Create a camera instance
cam = cv2.VideoCapture(0)

# Check if instantiation was successful
if not cam.isOpened():
    raise Exception("Could not open camera")
    
confidence_threshold = 0.1

while True:
    start = time.time()  # timer for FPS
    img = np.zeros((480, 640, 3))  # dummy image for testing

#######-your-code-starts-here-########

    # Capture frame-by-frame
    ret, frame = cam.read()

    # Flip horizontally and resize and calculate scales
    frame = cv2.flip(frame, 1)
    frameReSized = imresize(frame, (img_height, img_width),preserve_range = 'true')
    scaleW = 640 / img_height
    scaleH = 480 / img_width

    # First dimension image feed and predictions
    image = np.array([frameReSized])
    predictions = (model.predict(image))    

    # Delete row if confidence = 0
    predictionsThresholded = np.delete(predictions, np.where(predictions[0, :, 1] == 0), 1)

    # Check if detections exists
    if not (predictions[0, 0, 1] == 0):
        classIdAmount = predictionsThresholded[0, :, 0][:].astype(int)

        # Adding rectangles to all detections
        for i in range(len(classIdAmount)):
            boxes = [int(predictionsThresholded[0, i, 2]*scaleW), int(predictionsThresholded[0, i, 3]*scaleH),
                     int(predictionsThresholded[0, i, 4]*scaleW), int(predictionsThresholded[0, i, 5]*scaleH)]

            # Class id to class name
            predicted_class = classes[int(predictionsThresholded[0, :, 0][i])]
            confidence = predictionsThresholded[0, :, 1][i]
            label = "%s %f" % (predicted_class, confidence)

            # Rectangles to images
            frame = cv2.rectangle(frame,(boxes[0], boxes[1]), (boxes[2], boxes[3]), (255, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX

            frame = cv2.putText(frame, label, (boxes[0], boxes[3]-10), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    #######-your-code-ends-here-########     

   # Insert FPS/quit text and show image
    fps = "{:.0f} FPS".format(1/(time.time()-start))
    frame = cv2.putText(frame, fps, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale=0.8, color=(0,255,255))
    frame = cv2.putText(frame, 'Press q to quit', (440, 20), cv2.FONT_HERSHEY_SIMPLEX,
                      fontScale =0.8, color=(0,0,255))
    cv2.imshow('Video feed', frame)
    
    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()   
cv2.destroyAllWindows()

