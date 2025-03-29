import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Gradient_Loss(nn.Module):
    def __init__(self, device):
        super(Gradient_Loss, self).__init__()
        self.sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_X = torch.from_numpy(self.sobel_filter_X).float().to(device)
        self.sobel_filter_Y = torch.from_numpy(self.sobel_filter_Y).float().to(device)

    def forward(self, output, gt):
        #0,1,2,3
        b, c, h, w = output.size()

        output_X_c, output_Y_c = [], []
        gt_X_c, gt_Y_c = [], []
        for i in range(c):
            output_grad_X = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            output_grad_Y = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)
            gt_grad_X = F.conv2d(gt[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            gt_grad_Y = F.conv2d(gt[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)

            output_X_c.append(output_grad_X)
            output_Y_c.append(output_grad_Y)
            gt_X_c.append(gt_grad_X)
            gt_Y_c.append(gt_grad_Y)

        # Concatenate along the channel dimension (dim = 1)
        output_X = torch.cat(output_X_c, dim=1)
        output_Y = torch.cat(output_Y_c, dim=1)
        gt_X = torch.cat(gt_X_c, dim=1)
        gt_Y = torch.cat(gt_Y_c, dim=1)

        grad_loss = torch.mean(torch.abs(output_X - gt_X)) + torch.mean(torch.abs(output_Y - gt_Y))

        return grad_loss

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
        assert len(pred.size()) == 4
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

def main():
    from PIL import Image
    import torchvision.transforms as transforms
    
    img1_path = 'images/outputs/00020.png'
    img2_path = 'images/gt/00020.png'
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

    img1 = transform(Image.open(img1_path))
    img2 = transform(Image.open(img2_path))
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    psnr_loss = PSNRLoss() 
    loss = psnr_loss(img1, img2)
    
    print(f"Image shapes: {img1.shape}")
    print(f"Value ranges: [{img1.min():.3f}, {img1.max():.3f}]")
    print(f"PSNR Loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()