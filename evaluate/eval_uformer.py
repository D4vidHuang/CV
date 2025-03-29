import os
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from models.Uformer import Uformer
from torchvision.transforms.functional import crop
import torchvision.utils as tvu
from utils.image import imread, img2tensor

def data_transform(X):
    return (2 * X - 1.0).float()

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def single_image_inference(model, img):
    img = img.unsqueeze(0)  # add batch dimension
    print("Input image shape:", img.shape)
    # img = img.flatten(start_dim=0, end_dim=1) if img.ndim == 5 else img
    with torch.no_grad():
        output = process_image(img, image_size = 256, model=model, data_transform=data_transform, inverse_data_transform=inverse_data_transform)
    output = output.squeeze(0)
    return output

def process_image(x_cond, image_size, stride= 128, manual_batching_size=32, model=None, data_transform=None, inverse_data_transform=None):
    input_res = image_size
    print('input_res', input_res)

    # 计算裁剪的位置
    h_list = [i for i in range(0, x_cond.shape[2] - input_res + 1, stride)]
    w_list = [i for i in range(0, x_cond.shape[3] - input_res + 1, stride)]
    h_list = h_list + [x_cond.shape[2] - input_res]
    w_list = w_list + [x_cond.shape[3] - input_res]
    print('h_list', h_list)
    print('w_list', w_list)

    corners = [(i, j) for i in h_list for j in w_list]

    p_size = input_res
    x_grid_mask = torch.zeros_like(x_cond)

    # 更新x_grid_mask
    for (hi, wi) in corners:
        x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

    et_output = torch.zeros_like(x_cond)

    # 处理裁剪的小块
    x_cond_patch = torch.cat([crop(x_cond, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)

    for i in range(0, len(corners), manual_batching_size):
        print(f"Processing patch {i}/{len(corners)}")
        Output = model(data_transform(x_cond_patch[i:i+manual_batching_size]).float())

        # 累加输出结果
        for didx, (hi, wi) in enumerate(corners[i:i + manual_batching_size]):
            et_output[0, :, hi:hi + p_size, wi:wi + p_size] += Output[didx]

    x_output = torch.div(et_output, x_grid_mask)
    x_output = inverse_data_transform(x_output)
    return x_output
    # tvu.save_image(x_output, output_image_path)
    # print(f"Output image saved at: {output_image_path}")

def main():
    img_path = 'images/intermediate/00001_0.png'  
    output_path = 'images/intermediate/00001_uformer.png'
    img = imread(img_path)
    img_tensor = img2tensor(img)

    load_path = 'checkpoints/epoch100.pth.tar'
    model = Uformer(embed_dim = 16, modulator = True)
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    x_output = single_image_inference(model, img_tensor)
    tvu.save_image(x_output, output_path)
    print(f"Output image saved to {output_path}")

if __name__ == '__main__':
    main()
