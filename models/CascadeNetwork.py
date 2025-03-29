import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision

from models.Uformer import Uformer
from models.NAFNet import NAFNetLocal
from evaluate.eval_uformer import data_transform, inverse_data_transform
from torchvision.transforms.functional import crop

import utils
import utils.logging

# 定义级联网络
class CascadeNetwork(nn.Module):
    def __init__(self, uformer_kwargs, nafnet_kwargs):
        super(CascadeNetwork, self).__init__()
        # 初始化 Uformer
        self.uformer = Uformer(**uformer_kwargs)
        # 初始化 NAFNet
        self.nafnet = NAFNetLocal(**nafnet_kwargs)
    
    def data_transform(self, X):
        return 2 * X - 1.0
    
    def inverse_data_transform(self, X):
        return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
    
    def forward(self, x):
        intermediate_output = self.uformer(x)
        intermediate_output = self.inverse_data_transform(intermediate_output)
        # 再将 Uformer 的输出传递给 NAFNet
        # print(intermediate_output)
        final_output = self.nafnet(self.data_transform(intermediate_output))
        # print(final_output)
        return final_output

    def load_pretrained_weights(self, device, uformer_checkpoint_path, nafnet_checkpoint_path):
        # 加载 Uformer 的预训练权重
        uformer_checkpoint = torch.load(uformer_checkpoint_path, map_location=device, weights_only=False)
        uformer_state_dict = uformer_checkpoint['state_dict']
        new_uformer_state_dict = {k.replace('module.', ''): v for k, v in uformer_state_dict.items()}
        self.uformer.load_state_dict(new_uformer_state_dict, strict=True)

        # 加载 NAFNet 的预训练权重
        nafnet_checkpoint = torch.load(nafnet_checkpoint_path, map_location=device, weights_only=False)
        # if 'params' in nafnet_checkpoint:
        #     nafnet_state_dict = nafnet_checkpoint['params']
        # else:
        #     nafnet_state_dict = nafnet_checkpoint
        nafnet_state_dict = nafnet_checkpoint['state_dict']
        new_nafnet_state_dict = {k.replace('module.', ''): v for k, v in nafnet_state_dict.items()}
        self.nafnet.load_state_dict(new_nafnet_state_dict, strict=True)

def create_cascade_nets():
    # img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]
    # img_size=input_size, embed_dim=16, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False
    uformer_kwargs = {
        'embed_dim': 16, 
        'modulator': True,}
    nafnet_kwargs = {'img_channel': 3, 'width': 64, 'middle_blk_num': 1, 'enc_blk_nums': [1, 1, 1, 28], 'dec_blk_nums': [1, 1, 1, 1]}
    model = CascadeNetwork(uformer_kwargs, nafnet_kwargs)
    return model

def single_image_inference(model, img):
    img = img.unsqueeze(0)  # add batch dimension
    # print("Input image shape:", img.shape)
    # img = img.flatten(start_dim=0, end_dim=1) if img.ndim == 5 else img
    with torch.no_grad():
        output = process_image(img, image_size = 256, model=model, data_transform=data_transform, inverse_data_transform=inverse_data_transform)
    output = output.squeeze(0)
    return output

def process_image(x_cond, image_size, stride= 128, manual_batching_size=32, model=None, data_transform=None, inverse_data_transform=None):
    input_res = image_size
    print('input_res', input_res)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device', device)
    x_cond = x_cond.to(device)

    # 计算裁剪的位置
    h_list = [i for i in range(0, x_cond.shape[2] - input_res + 1, stride)]
    w_list = [i for i in range(0, x_cond.shape[3] - input_res + 1, stride)]
    h_list = h_list + [x_cond.shape[2] - input_res]
    w_list = w_list + [x_cond.shape[3] - input_res]
    # print('h_list', h_list)
    # print('w_list', w_list)

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
        # print('The shape of output', Output.shape)
        for didx, (hi, wi) in enumerate(corners[i:i + manual_batching_size]):
            et_output[0, :, hi:hi + p_size, wi:wi + p_size] += Output[didx]

    x_output = torch.div(et_output, x_grid_mask)
    x_output = inverse_data_transform(x_output)
    return x_output

def main():
    img_path = 'images/inputs/00001.png'  
    output_path = 'images/outputs/00001_2.png'
    uformer_path = 'checkpoints/epoch100.pth.tar'
    nafnet_path = 'Param/RainDrop/NAFNet/epoch25.pth.tar'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # input_img = Image.open(img_path)
    # input_img = input_img.resize((256, 256), Image.Resampling.LANCZOS)
    # transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # input_tensor = transforms(input_img)
    # input_tensor = input_tensor.unsqueeze(0)
    # input_tensor = data_transform(input_tensor)
    input = utils.image.imread(img_path)
    input_tensor = utils.image.img2tensor(input)

    model = CascadeNetwork(
        uformer_kwargs={'embed_dim': 16, 'modulator': True},
        nafnet_kwargs={'img_channel': 3, 'width': 64, 'middle_blk_num': 1, 'enc_blk_nums': [1, 1, 1, 28], 'dec_blk_nums': [1, 1, 1, 1]}
    )
    model.load_pretrained_weights(device, uformer_path, nafnet_path)
    model.eval()
    with torch.no_grad():
        output = single_image_inference(model, input_tensor)
    utils.logging.save_image(output, output_path)
    print(f"Output image saved to {output_path}")

if __name__ == '__main__':
    main()