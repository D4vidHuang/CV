from evaluate.eval_naf import imread, img2tensor, imwrite
import evaluate.eval_naf as eval_naf
import evaluate.eval_uformer as eval_uformer
from models.NAFNet import NAFNet
from models.Uformer import Uformer

import torch
import utils

def main():
    img_path = 'images/inputs/00005.png'  
    output_path = 'images/outputs/00005.png'

    img = imread(img_path)
    img_tensor = img2tensor(img)
    uformer_path = 'checkpoints/epoch100.pth.tar'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))

    uformer = Uformer(embed_dim = 16, modulator = True)
    checkpoint_u = torch.load(uformer_path, map_location=device, weights_only=False)
    state_dict = checkpoint_u['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    uformer.load_state_dict(new_state_dict, strict=True)
    with torch.no_grad():
        output = eval_uformer.single_image_inference(uformer, img_tensor)
    
    nefnet = NAFNet(img_channel=3,
                    width=64,
                    middle_blk_num=1,
                    enc_blk_nums=[1, 1, 1, 28],
                    dec_blk_nums=[1, 1, 1, 1])
    # path = 'checkpoints/NAFNet-REDS-width64.pth'
    checkpoint_n = torch.load('Param/RainDrop/NAFNet/epoch10.pth.tar', map_location=device, weights_only=False)
    # if 'params' in checkpoint:
    #     state_dict = checkpoint['params']
    # else:
    #     state_dict = checkpoint
    state_dict = checkpoint_n['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    nefnet.load_state_dict(new_state_dict, strict=True)
    nefnet.eval()
    with torch.no_grad():
        output = eval_naf.single_image_inference(nefnet, output)
    # tvu.save_image(output, output_path)
    utils.logging.save_image(output, output_path)

if __name__ == '__main__':
    main()