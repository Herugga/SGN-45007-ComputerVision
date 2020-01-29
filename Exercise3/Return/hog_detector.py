import cv2
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']= 180

def inside(r, q):
    """
    Returns true if rectangle r = (x,y,w,h) is inside rectangle q.
    """
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

# Initalize HOG people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load a short test video (GIF)
cap = cv2.VideoCapture("run.gif")


# Your task is to find the HOG detector parameters which are able to detect both
# runners perfectly when they are fully visible (The first few frames).
# The most important parameters are 'scale' and 'winStride'. For more information:
# https://docs.opencv.org/master/d5/d33/structcv_1_1HOGDescriptor.html#a91e56a2c317392e50fbaa2f5dc78d30b
#
# The parameters are as follows:
#   winStride: A 2-tuple that is the “step size” in both the x and y 
#              location of the sliding window.
#
#   scale: controls the factor in which our image is resized at each layer of
#          the Gaussian image pyramid, ultimately influencing the number of levels
#          in the image pyramid. A smaller scale will increase the number of
#          layers in the image pyramid and the processing time.
#
#   padding:  A tuple which indicates the number of pixels in both the x and y 
#             direction in which the sliding window ROI is “padded” prior to
#             HOG feature extraction.
#
#   hitThreshold: Threshold for the distance between features and SVM 
#                 classifying plane. This can be set to a value above 0 if there
#                 is a large amount of false positives.
#
# Start by finding a value for scale (between 1.0-2.0, higher values are more 
# computationally efficient) which yields some sort of results. You have to also draw the detections,
# which are in the form x_corner, y_corner, width, height. Next, try
# decreasing winStride to achieve more detections. Finally, try increasing 
# hitThreshold to get rid of possible false positives. After this you can try to optimize
# the parameters even more by simply trying out different values. Pay also attention
# to the execution time.

# GIF SIZE = 480 x 270
##--your-code-start-here--##
scale = 1.4
winStride = (3,3)
padding = (0, 0)
hitThreshold = 0.0
##--your-code-ends-here--##


delay = 1
while cv2.waitKey(delay) != ord('q'):
    try:
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
            continue

        start = time.time()
        detections, _ = hog.detectMultiScale(img,
                                             winStride=winStride,
                                             padding=padding,
                                             scale=scale,
                                             hitThreshold=hitThreshold)
        print('Detector execution time: ~{:.3f} s | Persons found: {}'.format((time.time() - start), len(detections)))

        # Draw the detections using e.g. cv2.rectangle
      # Filter overlapping detections
        found_filtered = []
        for ri, r in enumerate(detections):
            for qi, q in enumerate(detections):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        
      
        
        # Draw the filtered detections
        for x, y, w, h in found_filtered:
            cv2.rectangle(img, (x, y),
                          (x+w, y+h),
                          (0, 255, 0), thickness=3)
        plt.imshow(img[..., ::-1])
        plt.show()
       ##--your-code-ends-here--##

        # Limit FPS to ~8 (if detector is fast enough)
        if (time.time() - start) > 0.125:
            delay = 1
        else:
            delay = max(int(125 - (time.time() - start)*1000), 1)
        cv2.imshow("Press q to exit", img)

    except KeyboardInterrupt:
        break

# Exit cleanup
cv2.destroyAllWindows()
