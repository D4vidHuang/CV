import cv2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models.NAFNet import NAFNetLocal
from utils.image import imread, img2tensor, imwrite
import utils
from train.ddm import data_transform, inverse_data_transform

# python -m evaluate.eval_naf

def single_image_inference(model, img):
    # add batch dimension and transform
    img = data_transform(img.unsqueeze(0)) 
    print("Input image shape:", img.shape)
    with torch.no_grad():
        output = model(img) 
    # output_img = tensor2img(output.squeeze(0), rgb2bgr=False)
    output = inverse_data_transform(output.squeeze(0))
    return output

def main():
    # load input image
    img_path = 'images/outputs/00020.png'  
    output_path = 'images/outputs/00020_epoch_25.png'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    img = imread(img_path)
    img_tensor = img2tensor(img)

    # initialize the network
    model = NAFNetLocal(img_channel=3, 
                    width=64, 
                    middle_blk_num=1, 
                    enc_blk_nums=[1, 1, 1, 28], 
                    dec_blk_nums=[1, 1, 1, 1])

    # load the checkpoint
    path = 'Param/RainDrop/NAFNet/epoch25.pth.tar'
    checkpoint_n = torch.load(path, map_location=device, weights_only=False)
    # if 'params' in checkpoint_n:
    #     state_dict = checkpoint_n['params']
    # else:
    #     state_dict = checkpoint_n
    state_dict = checkpoint_n['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    
    # print("New state_dict keys:", new_state_dict.keys())
    model.load_state_dict(new_state_dict, strict=True)

    # inference
    model.eval()
    output_img = single_image_inference(model, img_tensor)
    utils.logging.save_image(output_img, output_path)
    print(f"Output image saved to {output_path}")

if __name__ == '__main__':
    main()