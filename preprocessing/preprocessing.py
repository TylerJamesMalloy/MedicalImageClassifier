import numpy as np
import pandas as pd
from PIL import Image
import opencv as cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
import glob
import os
import piexif
from multiprocessing import Pool, cpu_count
from functools import partial
from subprocess import check_output

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

for i in range(1,3):
    for filename in glob.iglob("../train/Type_" + str(1) + "/*.jpg"):
        # load image
        piexif.remove(filename)
        image = Image.open(filename)
        
        # process image 
        
        # resize to 256 x 256 
        
        # save image
        filename = filename.split("/")[-1]
        #image.save("../processed_images/Type_" + str(i + 1) + "/processed_" + filename)