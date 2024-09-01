import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
from pipeline.resnet_csra import ResNet_CSRA
from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA
#from pipeline.swin_csra import SWIN_BASE_PATCH4_224
from utils.evaluation.eval import class_dict
from pipeline.datasetBigEarth import DataSet


# Function to parse command line arguments
def Args():
    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument("--model", default="resnet101", type=str)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam", default=0.1, type=float)
    parser.add_argument("--load_from", default="E:\\Sarah\\CSRA-master\\checkpoint\\resnet101\\epoch_100.pth", type=str)
    parser.add_argument("--img_size", default=120, type=int)
    parser.add_argument("--dataset", default='bigearth', type=str)
    parser.add_argument("--num_cls", default=43, type=int)
    parser.add_argument("--rgb_dir", default="E:\\Sarah\\CSRA-master\\utils\\SelectedImages_rgb", type=str)
    parser.add_argument("--veg_dir", default="E:\\Sarah\\CSRA-master\\utils\\SelectedImages_veg", type=str)
    args = parser.parse_args()
    return args

# Function to load and preprocess images
def load_and_transform_image(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    return img

# Main function to run the demo
def demo():
    args = Args()

    # Load the model based on the specified architecture
    if args.model == "resnet101":
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls)
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    elif args.model == "vit_B16_224":
        model = VIT_B16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif args.model == "vit_L16_224":
        model = VIT_L16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise ValueError("Unsupported model type: {}".format(args.model))

    # Load model weights
    model.cuda()
    print("Loading weights from {}".format(args.load_from))
    model.load_state_dict(torch.load(args.load_from), strict=False)

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Process images and make predictions
    for img_file in os.listdir(args.rgb_dir):
        print(os.path.join(args.rgb_dir, img_file))
        print(os.path.join(args.veg_dir, img_file.replace("_rgb", "_veg")))


        # Load and transform images
        rgb = Image.open(os.path.join(args.rgb_dir, img_file))
        veg = Image.open(os.path.join(args.veg_dir, img_file.replace("_rgb", "_veg")))

        rgb = transform(rgb)
        veg = transform(veg)

        rgb = rgb.cuda()
        veg = veg.cuda()
        
        rgb = rgb.unsqueeze(0)
        veg = veg.unsqueeze(0)

        # Make predictions
        model.eval()
        logit = model(rgb, veg).squeeze(0)
        logit = nn.Sigmoid()(logit)

        # Output predictions
        pos = torch.where(logit > 0.4)[0].cpu().numpy()
        for k in pos:
            print(class_dict[args.dataset][k], end=",")
        print()

if __name__ == "__main__":
    demo()
