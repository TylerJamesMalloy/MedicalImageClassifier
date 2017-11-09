from PIL import Image
import numpy as np
import glob

# Iterate through Type 1 image files 
for filename in glob.iglob('../train/Type_1/*.jpg'):
     image = Image.open(filename)
     image = np.array(image)
     print(image.shape)

# Iterate through Type 2 image files
for filename in glob.iglob('../train/Type_2/*.jpg'):
     image = Image.open(filename)
     image = np.array(image)
     print(image.shape)

# Iterate through Type 3 image files
for filename in glob.iglob('../train/Type_3/*.jpg'):
     image = PIL.Image.open(filename)
     image = np.array(image)
     print(image.shape)