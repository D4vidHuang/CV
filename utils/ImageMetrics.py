import cv2
import numpy as np
import torch
import lpips

class ImageMetrics:
    def __init__(self, lpips_net='vgg'):
        """
        Initialize the ImageMetrics class.
        
        Args:
            lpips_net (str): The network to use for LPIPS calculation. Default is 'vgg'.
        """
        self.loss_fn_vgg = lpips.LPIPS(net=lpips_net)  # Load LPIPS model once

    def calculate_psnr(self, img1, img2, test_y_channel=False):
        """Calculate PSNR (Peak Signal-to-Noise Ratio).

        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: psnr result.
        """
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        assert img1.shape[2] == 3, "Input images must have 3 channels."
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        if test_y_channel:
            img1 = self.to_y_channel(img1)
            img2 = self.to_y_channel(img2)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20. * np.log10(255. / np.sqrt(mse))

    def calculate_ssim(self, img1, img2, test_y_channel=False):
        """Calculate SSIM (structural similarity).

        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

        Returns:
            float: ssim result.
        """
        assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')
        assert img1.shape[2] == 3, "Input images must have 3 channels."
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        if test_y_channel:
            img1 = self.to_y_channel(img1)
            img2 = self.to_y_channel(img2)

        ssims = []
        for i in range(img1.shape[2]):
            ssims.append(self._ssim(img1[..., i], img2[..., i]))
        return np.array(ssims).mean()

    def _ssim(self, img1, img2):
        """Calculate SSIM for one channel images.

        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].

        Returns:
            float: ssim result.
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def calculate_lpips(self, img1, img2):
        """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].

        Returns:
            float: lpips result.
        """
        torchres = torch.from_numpy(img1.transpose((2, 0, 1))).float().unsqueeze(0)
        torchgt = torch.from_numpy(img2.transpose((2, 0, 1))).float().unsqueeze(0)
        cur_lpips = self.loss_fn_vgg(torchres, torchgt)
        return cur_lpips.cpu().data.numpy()[0][0][0][0]

    def to_y_channel(self, img):
        """Change to Y channel of YCbCr.

        Args:
            img (ndarray): Images with range [0, 255].

        Returns:
            ndarray: Images with range [0, 255] (float type) without round.
        """
        img = img.astype(np.float32) / 255.
        if img.ndim == 3 and img.shape[2] == 3:
            img = self.bgr2ycbcr(img, y_only=True)
            img = img[..., None]
        return img * 255.

    def bgr2ycbcr(self, img, y_only=False):
        """Convert a BGR image to YCbCr image.

        Args:
            img (ndarray): The input image. It accepts:
                1. np.uint8 type with range [0, 255];
                2. np.float32 type with range [0, 1].
            y_only (bool): Whether to only return Y channel. Default: False.

        Returns:
            ndarray: The converted YCbCr image.
        """
        img_type = img.dtype
        img = self._convert_input_type_range(img)
        if y_only:
            out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
        else:
            out_img = np.matmul(
                img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
        out_img = self._convert_output_type_range(out_img, img_type)
        return out_img

    def _convert_input_type_range(self, img):
        """Convert the type and range of the input image.

        Args:
            img (ndarray): The input image.

        Returns:
            ndarray: The converted image with type of np.float32 and range of [0, 1].
        """
        img_type = img.dtype
        img = img.astype(np.float32)
        if img_type == np.float32:
            pass
        elif img_type == np.uint8:
            img /= 255.
        else:
            raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
        return img

    def _convert_output_type_range(self, img, dst_type):
        """Convert the type and range of the image according to dst_type.

        Args:
            img (ndarray): The image to be converted.
            dst_type (np.uint8 | np.float32): The desired type.

        Returns:
            ndarray: The converted image with desired type and range.
        """
        if dst_type not in (np.uint8, np.float32):
            raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
        if dst_type == np.uint8:
            img = img.round()
        else:
            img /= 255.
        return img.astype(dst_type)