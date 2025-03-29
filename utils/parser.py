import argparse
import os
import yaml
import torch.backends.cudnn as cudnn
import numpy as np

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Raindrop Clarity with Uformer')
    parser.add_argument("--config", type=str, default='NAFNet-width64.yml',
                        help="Path to the config file")
    parser.add_argument('--resume', default='/home1/yeying/release/RDiffusion/Param/NightRaindrop/Uformer/epoch100.pth.tar', type=str,
                        help='Path for the model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='Uformer',
                        help="restoration test set results: ['Raindrop_DiT', 'RDiffusion', 'IDT', 'restormer', 'Uformer', 'ICRA256', 'onego', 'atgan']")
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('--sid', type=str, default='00197')
    args = parser.parse_args()
    print(args)

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