import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import crop
from torch.utils.tensorboard import SummaryWriter

import utils
from models.Uformer import create_uformer_nets
from models.NAFNet import create_nafnet_nets
from models.CascadeNetwork import create_cascade_nets
from models.loss import Gradient_Loss, PSNRLoss
from utils.ImageMetrics import ImageMetrics
import utils.logging


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def noise_estimation_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        if self.args.test_set == 'Uformer':
            self.model = create_uformer_nets()
            self.model_name = 'Uformer'
            assert self.config.data.image_size == 256, f"Expected image_size 256, but got {self.config.data.image_size}"

        elif self.args.test_set == 'NAFNet':
            self.model = create_nafnet_nets()
            self.model_name = 'NAFNet'
        
        elif self.args.test_set ==  'Cascade':
            self.model = create_cascade_nets()
            self.model_name = 'Cascade'
        
        print(f"Using model: {self.model_name}")
        self.model.to(self.device)
        # if training the network on multiple GPUs
        # if torch.cuda.device_count() > 1:
        # self.model = torch.nn.DataParallel(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.n_epochs,
            eta_min=2e-5 
        )
        self.start_epoch, self.step = 0, 0

        # tensorboard
        log_dir = os.path.join('runs', self.config.data.dataset, self.model_name, time.strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(log_dir)

        self.psnr_loss = PSNRLoss(loss_weight=config.loss.psnr_weight, toY=True).to(self.device) if config.loss.psnr_weight else None
        self.grad_loss = Gradient_Loss(device=self.device).to(self.device) if config.loss.grad_weight else None
        self.loss_l1 = torch.nn.L1Loss(reduce=True, size_average=True) if config.loss.pixel_weight else None
        self.loss_l2 = torch.nn.MSELoss(reduce=True, size_average=True) if config.loss.l2_weight else None
    
    def load_checkpoint(self, load_path):
        if self.model_name == 'Uformer' or self.model_name == 'NAFNet':
            # load checkpoint
            checkpoint_u = torch.load(load_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint_u['state_dict']
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=True)
            
            # load optimizer
            self.optimizer.load_state_dict(checkpoint_u['optimizer'])

            # load epoch and step
            self.start_epoch = checkpoint_u['epoch']
            self.step = checkpoint_u['step']
            print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, self.start_epoch, self.step))
        elif self.model_name == 'Cascade':
            print('loading checkpoint for Cascade')
            # load checkpoint
            checkpoint_c = torch.load(load_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint_c['state_dict']
            if not self.config.training.resume:
                state_dict = {k.replace('module', 'uformer'): v for k, v in state_dict.items()}
            else:
                self.optimizer.load_state_dict(checkpoint_c['optimizer'])
                self.start_epoch = checkpoint_c['epoch']
                self.step = checkpoint_c['step']
                print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, self.start_epoch, self.step))
            result = self.model.load_state_dict(state_dict, strict=False)
            # print("Missing keys:", result.missing_keys)
            # print("Unexpected keys:", result.unexpected_keys)
            # import sys
            # sys.exit()
            
            # load optimizer
            # self.optimizer.load_state_dict(checkpoint_c['optimizer'])
            # load epoch and step
            # self.start_epoch = checkpoint_c['epoch']
            # self.step = checkpoint_c['step']
            # print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, self.start_epoch, self.step))
        # elif self.model_name == 'NAFNet':
        #     # load checkpoint
        #     checkpoint_n = torch.load(load_path, map_location=self.device)
        #     state_dict = checkpoint_n['state_dict']
        #     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        #     self.model.load_state_dict(state_dict, strict=True)
        else:
            raise NotImplementedError(self.model_name)

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_checkpoint(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            avg_loss = 0
            cnt = 0
            for i, (x, y) in enumerate(train_loader):
                # print(i,x.shape,y)
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                X_input = x[:,:3,:,:]
                X_GT    = x[:,3:,:,:]
                X_output = self.model(X_input)
                loss = 0
                
                if self.loss_l1:
                    lossl1 = self.loss_l1(X_output, X_GT)
                    loss += self.config.loss.pixel_weight * lossl1
                if self.loss_l2:
                    lossl2 = self.loss_l2(X_output, X_GT)
                    loss += self.config.loss.l2_weight * lossl2
                if self.psnr_loss:
                    X_output_inversed = inverse_data_transform(X_output)
                    X_GT_inversed = inverse_data_transform(X_GT)
                    psnr = self.psnr_loss(X_output_inversed, X_GT_inversed)
                    loss += self.config.loss.psnr_weight * psnr
                if self.grad_loss:
                    grad_loss = self.grad_loss(X_output, X_GT)
                    loss += self.config.loss.grad_weight * grad_loss
                avg_loss += loss.item()
                cnt += 1

                # record loss   
                if self.step % 100 == 0:
                    print(f"step: {self.step}, avg_loss: {avg_loss/cnt:.4f}")
                    # print('-grad_loss-',grad_loss)
                    self.writer.add_scalar('Training/Avg_Loss', avg_loss/cnt, self.step)
                    self.writer.add_scalar('Training/Data_Time', data_time / (i+1), self.step)
                    # self.writer.add_scalar('Training/L1_Loss', lossl1.item(), self.step)
                    if self.config.loss.grad_weight:
                        self.writer.add_scalar('Training/Gradient_Loss', grad_loss.item(), self.step)
                
                # update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del x, X_input, X_GT, X_output, loss
                torch.cuda.empty_cache()

                data_start = time.time()

                # validation
                # if self.step % self.config.training.validation_freq == 0:
                #     self.model.eval()
                #     self.sample_validation_patches(val_loader, self.step)

            # print average loss
            avg_loss /= cnt
            print(f"epoch: {epoch}, avg_loss: {avg_loss:.4f}")
            
            # update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            self.writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
            
            # save checkpoint
            if (epoch+1) % self.config.training.snapshot_freq  == 0:
                checkpoint_path = os.path.join('Param/', self.config.data.dataset + '/' + self.model_name +'/'+self.config.data.condition+'-epoch'+str(epoch + 1))
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'step': self.step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=checkpoint_path)
                print(f"Checkpoint saved at: {checkpoint_path}")

            torch.cuda.empty_cache()
    
    def get_image(self, x, image_size, stride= 128, manual_batching_size=32, model=None, data_transform=None, inverse_data_transform=None):
        input_res = image_size

        # calculate cropping positions
        h_list = [i for i in range(0, x.shape[2] - input_res + 1, stride)]
        w_list = [i for i in range(0, x.shape[3] - input_res + 1, stride)]
        h_list = h_list + [x.shape[2] - input_res]
        w_list = w_list + [x.shape[3] - input_res]

        corners = [(i, j) for i in h_list for j in w_list]

        p_size = input_res
        x_grid_mask = torch.zeros_like(x)

        # update x_grid_mask
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        et_output = torch.zeros_like(x)

        # process cropped small pieces
        x_cond_patch = torch.cat([crop(x, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)

        for i in range(0, len(corners), manual_batching_size):
            print(f"Processing patch {i}/{len(corners)}")
            Output = model(data_transform(x_cond_patch[i:i+manual_batching_size]).float())

            # accumulate output results
            for didx, (hi, wi) in enumerate(corners[i:i + manual_batching_size]):
                et_output[0, :, hi:hi + p_size, wi:wi + p_size] += Output[didx]

        x_output = torch.div(et_output, x_grid_mask)
        x_output = inverse_data_transform(x_output)
        return x_output
    
    # only process a single batch of validation images
    def sample_validation_patches(self, val_loader, step):

        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        psnr_list = []
        ssim_list = []
        lpips_list = []
        weighted_avg_list = []
        metrics_calculator = ImageMetrics(lpips_net='vgg')

        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                print(f'the shape of x is {x.shape}')
                break
            x_input = x[:, :3, :, :].to(self.device)
            ground_truth = x[:, 3:, :, :].to(self.device)
            self.writer.add_images('Validation/Ground_Truth', ground_truth, step)
            self.writer.add_images('Validation/Input', x_input, step)
            utils.logging.save_image(ground_truth, os.path.join(image_folder, str(step), f"{i}_loss_gt.png"))
            utils.logging.save_image(x_input, os.path.join(image_folder, str(step), f"{i}_loss_origin_input.png"))
            # utils.logging.save_image(ground_truth, os.path.join(image_folder, str(step), f"{i}_loss_psnr_gt.png"))
            if self.config.data.resize:
                x_output = self.get_image(x_input, self.config.data.image_size, stride=128, manual_batching_size=32, model=self.model, data_transform=data_transform, inverse_data_transform=inverse_data_transform)
            else:
                x_input = data_transform(x_input)
                x_output = self.model(x_input)
                x_output = inverse_data_transform(x_output)
            utils.logging.save_image(x_output, os.path.join(image_folder, str(step), f"{i}_loss_output.png"))
            self.writer.add_images('Validation/Output', x_output, step)

            print("Calculating metrics...")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_input = x[:, :3, :, :].to(self.device)
                ground_truth = x[:, 3:, :, :].to(self.device)
                if self.config.data.resize:
                    x_output = self.get_image(x_input, self.config.data.image_size, stride=128, manual_batching_size=32, model=self.model, data_transform=data_transform, inverse_data_transform=inverse_data_transform)

                else:
                    x_input = data_transform(x_input)
                    x_output = self.model(x_input)
                    x_output = inverse_data_transform(x_output)
                x_output_np = x_output.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                ground_truth_np = ground_truth.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                
                cur_psnr = metrics_calculator.calculate_psnr(x_output_np[0], ground_truth_np[0], test_y_channel=True)
                cur_ssim = metrics_calculator.calculate_ssim(x_output_np[0], ground_truth_np[0], test_y_channel=True)
                cur_lpips = metrics_calculator.calculate_lpips(x_output_np[0], ground_truth_np[0])
                weighted_avg = cur_psnr + 10 * cur_ssim - 5 * cur_lpips
                
                psnr_list.append(cur_psnr)
                ssim_list.append(cur_ssim)
                lpips_list.append(cur_lpips)
                weighted_avg_list.append(weighted_avg)
                break
            
            # Calculate average metrics
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            avg_lpips = np.mean(lpips_list)
            avg_weighted_avg = np.mean(weighted_avg_list)
            print(f'PSNR of {len(psnr_list)} pic: {avg_psnr:.2f} dB')
            print(f'SSIM: {avg_ssim:.4f}')
            print(f'LPIPS: {avg_lpips:.4f}')
            print(f'Weighted Metric: {avg_weighted_avg:.4f}')
            self.writer.add_scalar('Validation/PSNR', avg_psnr, step)
            self.writer.add_scalar('Validation/SSIM', avg_ssim, step)
            self.writer.add_scalar('Validation/LPIPS', avg_lpips, step)
            self.writer.add_scalar('Validation/Weighted_Metric', avg_weighted_avg, step)