import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from pipeline.resnet_csra import ResNet_CSRA
from pipeline.vit_csra_1inp import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
# from pipeline.swin_csra import SWIN_BASE_PATCH4_224
# from pipeline.dataset import DataSet
from torchvision.transforms import transforms
from utils.evaluation.eval import voc_classes, wider_classes, coco_classes, class_dict


# Usage:
# This demo is used to predict the label of each image
# if you want to use our models to predict some labels of the VOC2007 images
# 1st: use the models pretrained on VOC2007
# 2nd: put the images in the utils/demo_images
# 3rd: run demo.py

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model default resnet101
    parser.add_argument("--model", default="vit_B16_224", type=str)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.3, type=float)
    parser.add_argument("--load_from", default="E:\\Sarah\\CSRA-master\\checkpoint\\vit_B16_224_BE\\epoch_100.pth", type=str)
    parser.add_argument("--img_dir", default="E:\\Sarah\\CSRA-master\\utils\\rgb_new", type=str)
    parser.add_argument("--img_size", default=224, type=int)

    # dataset
    parser.add_argument("--dataset", default="bigearth", type=str)
    parser.add_argument("--num_cls", default=43, type=int)

    args = parser.parse_args()
    return args
    

def demo():
    args = Args()

    # model = None

    # model 
    if args.model == "resnet101": 
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls)
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]) # Z-score =  x-mean/std
        #img_size = 448
    if args.model == "vit_B16_224":
        model = VIT_B16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #img_size = 224
    if args.model == "vit_L16_224":
        model = VIT_L16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #img_size = 224
    if args.model == "swin_base_patch4_224":
        model = SWIN_BASE_PATCH4_224(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


    model.cuda()
    print("Loading weights from {}".format(args.load_from))
    model.load_state_dict(torch.load(args.load_from), strict=False)

    # image pre-process
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # prediction of each image's label
    for img_file in os.listdir(args.img_dir):
        print(os.path.join(args.img_dir, img_file), end=" prediction: ")

        #uncomment: one img only
        img = Image.open(os.path.join(args.img_dir, img_file)).convert("RGB")
        img = transform(img)
        img = img.cuda()
        img = img.unsqueeze(0)
        model.eval()
        logit = model(img).squeeze(0)
        logit = nn.Sigmoid()(logit)

        # if img_file.split("_")[-1] == "fc":
        #     img_fc = Image.open(os.path.join(args.img_dir, img_file)).convert("RGB")
            

        #     img_fc = Image.open(ann["img_path"]+"_fc.jpg").convert("RGB") #append the band combination
        #     img_sir = Image.open(ann["img_path"]+"_sir.jpg").convert("RGB") #append the band combination
        #     img_nwar = Image.open(ann["img_path"]+"_nwar.jpg").convert("RGB") #append the band combination



        pos = torch.where(logit > 0.5)[0].cpu().numpy()
        for k in pos:
            print(class_dict[args.dataset][k], end=",")
        print()


if __name__ == "__main__":
    demo()
