import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class RainDrop:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.list = []
        # for i in range(1,197):
        #     self.list.append('%05d'%i)
        # self.testlist = ['00060','00129','00144','00147','00151','00161','00164','00168',
        # '00173','00174','00176','00181','00184','00185','00186','00187','00188','00189',
        # '00190','00191','00192','00193','00194','00195','00196','00197']
        print(config.data.data_dir)
        for folder in os.listdir(config.data.data_dir):
            full_path = os.path.join(config.data.data_dir, folder)
            print(full_path)
            for nfolder in os.listdir(full_path):
                print(nfolder)
                # if os.path.isdir(full_path):
                #     self.list.append(folder)
                self.list.append(nfolder)

        # self.testlist = random.sample(self.list, 20)
        self.list = self.list[:121]
        self.testlist = self.list[:20]

        # Example: print the selected folder names
        self.trainlist =  [x for x in self.list if x not in self.testlist]
        print('-trainlist-',self.trainlist)
        print('-testlist-',self.testlist)

    def get_loaders(self, parse_patches=True, validation='raindrop'):
        print("=> evaluating raindrop test set...")
        train_dataset = RainDropDataset(dir=os.path.join(self.config.data.data_dir),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=self.trainlist,
                                        parse_patches=parse_patches)

        val_dataset = RainDropDataset(dir=os.path.join(self.config.data.data_dir),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      filelist=self.testlist,
                                      parse_patches=parse_patches)


        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class RainDropDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()
        print('-dir-',dir)

        self.dir = dir
        train_list = []
        input_names = []
        gt_names = []
        for i in range(len(filelist)):
            inpdir = os.path.join(self.dir, 'Drop',filelist[i])
            # gtdir = inpdir.replace('/Drop/','/Clear/')
            inpdir = os.path.normpath(inpdir)  # Normalizes slashes based on OS
            gtdir = inpdir.replace(os.path.normpath('/Drop/'), os.path.normpath('/Clear/'))
            # print(inpdir,gtdir)
            listinpdir = sorted(os.listdir(inpdir))
            for j in range(len(listinpdir)): #len(listinpdir)
                input_names.append(os.path.join(inpdir, listinpdir[j]))
            listgtdir = sorted(os.listdir(gtdir))
            for j in range(len(listgtdir)): #len(listgtdir)
                gt_names.append(os.path.join(gtdir, listgtdir[j]))
        print('len(input_names),len(gt_names) = ',len(input_names),len(gt_names))
        # print(input_names)
        print(input_names[0],gt_names[0])
        print(input_names[1],gt_names[1])
        print(input_names[-2],gt_names[-2])
        print(input_names[-1],gt_names[-1])
                # train_list.append()
            # train_list = os.path.join(dir, filelist)
            # print(train_list)
            # with open(train_list) as f:
            #     contents = f.readlines()
            #     input_names = [i.strip() for i in contents]
            #     gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0], [0], h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        # print('input_name',input_name)
        # print('gt_name',gt_name)

        # datasetname = re.split('/', input_name)[-4]
        # img_vid = re.split('/', input_name)[-2]
        # img_id = re.split('/', input_name)[-1][:-4]
        # img_id = datasetname+'__'+img_vid+'__'+img_id


        normalized_path = os.path.normpath(input_name)
        parts = normalized_path.split(os.sep)

        # Check if the path has enough components
        if len(parts) < 4:
            raise ValueError(f"Invalid path structure: expected at least 4 parts, got {len(parts)} in '{input_name}'")

        datasetname = parts[-4]
        img_vid = parts[-2]
        # Use os.path.splitext to remove the file extension safely
        img_id = os.path.splitext(parts[-1])[0]

        # Combine the components
        img_id = datasetname + '__' + img_vid + '__' + img_id


        # print('-1-',input_name, gt_name)
        # input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        input_img = PIL.Image.open(input_name)
        gt_img = PIL.Image.open(gt_name)


        if self.parse_patches:
            wd_new = 512
            ht_new = 512
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS) #PIL.Image.ANTIALIAS
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            # print('-input_img.shape,gt_img.shape-',input_img.size,gt_img.size)
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            wd_new = 256
            ht_new = 256
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            # print(input_img.shape,gt_img.shape)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
