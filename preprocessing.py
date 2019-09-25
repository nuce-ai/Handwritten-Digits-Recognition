from PIL import Image 
import numpy as np 
def processing():
    im = Image.open("database/number1.png").convert("L")
    im = im.resize((8,8))
    imr = np.asarray(im)
    return imr