from PIL import Image 
import numpy as np 
def processing(source):
    im = Image.open(source).convert("L")
    im = im.resize((8,8))
    imr = np.asarray(im)
    return imr