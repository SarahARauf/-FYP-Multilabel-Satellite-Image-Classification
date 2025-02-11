import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from torch.utils.data import DataLoader
from pipeline.resnet_csra import ResNet_CSRA
from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
#from pipeline.swin_csra import SWIN_BASE_PATCH4_224
from pipeline.datasetBigEarth import DataSet
from utils.evaluation.eval import evaluation
#from utils.evaluation.eval import WarmUpLR
from tqdm import tqdm


def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model default resnet101
    parser.add_argument("--model", default="resnet101", type=str)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument("--load_from", default="models_local/resnet101_voc07_head1_lam0.1_94.7.pth", type=str)
    # dataset
    parser.add_argument("--dataset", default="voc07", type=str)
    parser.add_argument("--num_cls", default=20, type=int)
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=16, type=int)

    args = parser.parse_args()
    return args
    

def val(args, model, test_loader, test_file, test_results, class_results):
    model.eval()
    print("Test on Pretrained Models")
    result_list = []

    print('hello')

    # calculate logit
    for index, data in enumerate(tqdm(test_loader)):

        # #uncomment: one img
        # img = data['img'].cuda()
        # target = data['target'].cuda()
        # img_path = data['img_path']

        img_rgb = data['img_rgb'].cuda()
        img_veg = data ['img_veg'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path'] #img_path only used to get name of file




        # img_fc = data['img_fc'].cuda()
        # img_nwar = data['img_nwar'].cuda()
        # img_sir = data ['img_sir'].cuda()
        # target = data['target'].cuda() #target y data
        # img_path = data['img_path']


        with torch.no_grad():
            # #uncomment
            # logit = model(img)
            logit = model(img_rgb, img_veg)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    
    # cal_mAP OP OR
    evaluation(result=result_list, types=args.dataset, ann_path=test_file[0], results_csv=test_results, epoch=1, is_testing_data=class_results, )



def main():
    args = Args()

    # model 
    if args.model == "resnet101": 
        # model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix)
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls)

    if args.model == "vit_B16_224":
        model = VIT_B16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
    if args.model == "vit_L16_224":
        model = VIT_L16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
    if args.model == "swin_base_patch4_224":
        model = SWIN_BASE_PATCH4_224(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)

    model.cuda()
    print("Loading weights from {}".format(args.load_from))
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model.module.load_state_dict(torch.load(args.load_from))
    else:
        model.load_state_dict(torch.load(args.load_from))

    # data
    if args.dataset == "voc07":
        test_file = ['data/voc07/test_voc07.json']
    if args.dataset == "coco":
        test_file = ['data/coco/val_coco2014.json']
    if args.dataset == "wider":
        test_file = ['data/wider/test_wider.json']
    if args.dataset == "mlrsnet":
        test_file = ["data/mlrsnet/test_mlrsnet.json"]

    if args.dataset == "bigearth":
        test_file = ["data/bigearth/test_bigearth.json"]

    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    os.makedirs(f"checkpoint/{args.model}", exist_ok=True)


    test_results = f"checkpoint/{args.model}/test_results.csv"
    class_results = f"checkpoint/{args.model}/class_results.csv"

    with open(test_results, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with open(class_results, mode='w', newline='') as csvfile:
        fieldnames = ['class','precision', 'recall', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    val(args, model, test_loader, test_file, test_results, class_results)


if __name__ == "__main__":
    main()
