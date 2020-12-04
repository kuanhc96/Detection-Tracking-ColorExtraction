from scipy.spatial import distance as dist
from collections import OrderedDict
from sklearn.cluster import KMeans
import numpy as np
import cv2
class ColorLabeler:
    def __init__(self):
        # initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        colors = OrderedDict({
            "red": (220,20,60),
            "maroon":(128, 0, 0),
            "tan":(210, 180, 140),
            "orange":(255, 165, 0),
            "gold":(255, 215, 0),
            "green":(0,128,0),
            "lime": (50, 205, 50),
            "blue": (0, 0, 255),
            "navy": (0, 0, 128),
            "yellow": (254,254,34),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "pink": (255,192,203),
            "purple": (128, 0, 128),
            "indigo": (75,0,130),
            "teal": (0, 128, 128),
            "olive": (128, 128, 0),
            "gray": (128, 128, 128),
            "brown":(160, 82, 45),
            "white": (255,255,255),
            "black": (0, 0, 0)})
        # allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []
        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
        # convert the L*a*b* array from the RGB color space
        # to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def get_rgb(self, image, c):
        # construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]
        return mean
    
    def label(self, mean):
        # initialize the minimum distance found thus far
        minDist = (np.inf, None)
        # loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)
            # if the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < minDist[0]:
                minDist = (d, i)
        # return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]
