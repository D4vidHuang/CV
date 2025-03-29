import cv2
import numpy as np
from PIL import Image
from basicsr.utils import img2tensor as _img2tensor

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def imwrite(img, save_path):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)