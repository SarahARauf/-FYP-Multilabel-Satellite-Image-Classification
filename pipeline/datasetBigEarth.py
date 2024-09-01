
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np

# modify for transformation for vit
# modfify wider crop-person images

# DataSet(train/test_file, args.train_aug, args.img_size, args.dataset)


class DataSet(Dataset):
    def __init__(self,
                ann_files,
                augs,
                img_size,
                dataset,
                ):
        self.dataset = dataset #dataset name
        self.ann_files = ann_files #the class label json file
        self.augment = self.augs_function(augs, img_size) #using a function to do augs and change img size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]) 
            ] 
            # In this paper, we normalize the image data to [0, 1]
            # You can also use the so called 'ImageNet' Normalization method
        )
        self.anns = []
        self.load_anns()
        print(self.augment)

        # in wider dataset we use vit models
        # so transformation has been changed
        if self.dataset == "wider":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ] 
            )        

    def augs_function(self, augs, img_size):            
        t = []
        if 'randomflip' in augs:
            t.append(transforms.RandomHorizontalFlip())
        if 'ColorJitter' in augs:
            t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0))
        if 'resizedcrop' in augs:
            t.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        if 'RandAugment' in augs:
            t.append(RandAugment())

        t.append(transforms.Resize((img_size, img_size)))

        return transforms.Compose(t)
    
    #loading all json files
    def load_anns(self):
        self.anns = []
        for ann_file in self.ann_files:
            json_data = json.load(open(ann_file, "r"))
            self.anns += json_data

    def __len__(self): #returns the number of samples in the dataset
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
        img_rgb = Image.open(ann["img_path"]+"_rgb.jpg").convert("RGB") #append the band combination
        img_veg = Image.open(ann["img_path"]+"_veg.jpg").convert("RGB") #append the band combination


        if self.dataset == "wider":
            x, y, w, h = ann['bbox']
            img_area = img.crop([x, y, x+w, y+h])
            img_area = self.augment(img_area)
            img_area = self.transform(img_area)
            message = {
                "img_path": ann['img_path']+"_sir",
                "target": torch.Tensor(ann['target']), 
                "img": img_area
            }
        else: # voc and coco, mlrsnet
            img_rgb = self.augment(img_rgb) #do augmentations on img
            img_veg = self.augment(img_veg)

            # img_fc = self.transform(img_fc) #do transformation on img
            img_rgb = self.transform(img_rgb) #do transformation on img
            img_veg = self.transform(img_veg) #do transformation on img

            message = {
                "img_path": ann["img_path"], #imae path
                # "img_path_sir": ann["img_path"]+"_sir.jpg", #imae path
                # "img_path_rgb": ann["img_path"]+"_rgb.jpg", #imae path

                "target": torch.Tensor(ann["target"]), #image target
                # "img_fc": img_fc, #image transformed
                "img_rgb": img_rgb,
                "img_veg": img_veg
            }

        #print(img.shape)
        #print("in dataset.py")

        return message
        # finally, if we use dataloader to get the data, we will get
        # {
        #     "img_path": list, # length = batch_size
        #     "target": Tensor, # shape: batch_size * num_classes
        #     "img": Tensor, # shape: batch_size * 3 * 224 * 224
        # }
