import argparse
import os
import yaml
import torch
import torch.utils.data
import numpy as np
import datasets
from train.ddm import DenoisingDiffusion

#CUDA_VISIBLE_DEVICES=1,2 python train.py --config daytime_64.yml --test_set Raindrop_DiT, RDiffusion, onego
#CUDA_VISIBLE_DEVICES=1,2 python train.py --config daytime_128.yml --test_set IDT, restormer
#CUDA_VISIBLE_DEVICES=1,2 python train.py --config daytime_256.yml --test_set ICRA256, Uformer, atgan
#CUDA_VISIBLE_DEVICES=1,2 python train.py --config daytime_64.yml --test_set Raindrop_DiT
#CUDA_VISIBLE_DEVICES=1,2 python train.py --config nighttime_256.yml --test_set atgan
# python -m train.train --config NAFNet.yml --resume checkpoints/NAFNet-REDS-width64.pth --test_set NAFNet
# python -m train.train --config NAFNet.yml --test_set NAFNet
# python -m train.train --config Cascade.yml --resume Param/RainDrop/Cascade/epoch110.pth.tar --test_set Cascade
# python -m train.train --config Cascade.yml --resume checkpoints/uformerday-epoch100.pth.tar --test_set Cascade
# python -m train.train --config Cascade.yml --resume checkpoints/uformernight-epoch100.pth.tar --test_set Cascade

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Raindrop Clarity with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, default='',
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--test_set", type=str, default='Uformer',
                        help="restoration test set results: ['Uformer', 'NAFNet']")
    parser.add_argument("--image_folder", default='images/results', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    torch.cuda.empty_cache()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)

    # create model
    print("=> creating model...")
    diffusion = DenoisingDiffusion(args, config)
    print(torch.cuda.memory_summary())
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
