import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import csv #added
import os #added
from torch.utils.data import DataLoader
from pipeline.resnet_csra import ResNet_CSRA
# from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
# from pipeline.swin_csra import SWIN_BASE_PATCH4_224
from pipeline.dataset import DataSet
from utils.evaluation.eval import evaluation
from utils.evaluation.warmUpLR import WarmUpLR
from tqdm import tqdm


# modify for wider dataset and vit models

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument("--cutmix", default=None, type=str) # the path to load cutmix-pretrained backbone
    # dataset
    parser.add_argument("--dataset", default="voc07", type=str)
    parser.add_argument("--num_cls", default=20, type=int)
    #parser.add_argument("--train_aug", default=["randomflip"], type=list)
    parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    # optimizer, default SGD
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--w_d", default=0.0001, type=float, help="weight_decay")
    parser.add_argument("--warmup_epoch", default=2, type=int)
    parser.add_argument("--total_epoch", default=1, type=int)#changed total epochs to 1
    parser.add_argument("--print_freq", default=100, type=int)
    args = parser.parse_args()
    return args
    

def train(i, args, model, train_loader, optimizer, warmup_scheduler, train_file, loss_csv, results_csv): #added results_csv
    print()
    model.train()
    epoch_begin = time.time() #for each epoch
    result_list = [] #added

    total_loss = 0.0 #added

    for index, data in enumerate(train_loader): #for each batch
        batch_begin = time.time() 
        # img = data['img'].cuda() #image data
        #added for bigearth:
        img_fc = data['img_rgb'].cuda()
        mg_nwar = data['img_veg'].cuda()
        target = data['target'].cuda() #target y data

        optimizer.zero_grad() 
        # logit, loss = model(img, target) #get logit and loss for each image target pair NOTE, edit this to take in more than one img from dataloader

        # #added for bigearth
        logit, loss = model(img_rgb, img_veg, target) #get logit and loss for each image target pair NOTE, edit this to take in more than one img from dataloader

        loss = loss.mean() #computes mean of the individual losses if the loss is a tensor with multiple values (for multiple images in a batch)
        total_loss += loss.item() #added the loss here

        # Assuming your model produces probabilities using a sigmoid activation
        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist() #added
        #ground_truth = target.cpu().numpy().tolist() #added

        #added
        for k in range(len(data['img_path'])):
            result_list.append(
                {
                    "file_name": data['img_path'][k].split("/")[-1].split(".")[0],
                    "scores": result[k],
                    #"ground_truth": ground_truth[k]
                }
            )


        loss.backward() #calculated the graidents of the loss with respect to the model parameters
        optimizer.step() #performs parameter update based on computed gradients
        t = time.time() - batch_begin #calculates time taken for current batch


        if index % args.print_freq == 0: #print batchwise
            print("Epoch {}[{}/{}]: loss:{:.5f}, lr:{:.5f}, time:{:.4f}".format(
                i, 
                args.batch_size * (index + 1),
                len(train_loader.dataset),
                loss,
                optimizer.param_groups[0]["lr"],
                float(t)
            ))

        if warmup_scheduler and i <= args.warmup_epoch: #checks if a learning rate warm up scheduler provided, and if current epoch is within warm up phase
            warmup_scheduler.step()
        
    average_loss = total_loss / len(train_loader) #added (computes avg loss for current epoch)
    t = time.time() - epoch_begin


    #added
    with open(loss_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'training_loss'])
        writer.writerow({'epoch': i, 'training_loss': average_loss})

    
    #added
    evaluation(result=result_list, types=args.dataset, ann_path=train_file[0], results_csv=results_csv, epoch=i)

    print("Epoch {} training ends, total {:.2f}s".format(i, t))


def val(i, args, model, test_loader, test_file, results_csv, loss_csv):
    model.eval()
    print("Test on Epoch {}".format(i))
    result_list = []

    total_loss = 0.0 #added for validation loss


    # calculate logit
    for index, data in enumerate(tqdm(test_loader)): #for each batch 
        #img = data['img'].cuda()

        # #added for big earth
        img_fc = data['img_rgb'].cuda()
        img = data ['img_veg'].cuda()

        target = data['target'].cuda()
        img_path = data['img_path'] #img_path only used to get name of file

        with torch.no_grad():
            #logit, loss = model(img, target) #changed from logit = model(img)

            # #added for big earth
            logit, loss = model(img_rgb, img_veg target) #changed from logit = model(img)
            loss = loss.mean() #added
            total_loss+= loss.item() #added


        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    #added
    average_loss = total_loss / len(test_loader)
    with open(loss_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'validation_loss'])
        writer.writerow({'epoch': i, 'validation_loss': average_loss})



    # cal_mAP OP OR
    evaluation(result=result_list, types=args.dataset, ann_path=test_file[0], results_csv=results_csv, epoch=i)



def main():
    args = Args()

    # model
    if args.model == "resnet101": 
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix)
    if args.model == "vit_B16_224":
        model = VIT_B16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
    if args.model == "vit_L16_224":
        model = VIT_L16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
    if args.model == "swin_base_patch4_224":
        model = SWIN_BASE_PATCH4_224(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)

        
    model.cuda()
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # data
    if args.dataset == "voc07":
        train_file = ["data/voc07/trainval_voc07.json"]
        test_file = ['data/voc07/test_voc07.json']
        step_size = 4
    if args.dataset == "coco":
        train_file = ['data/coco/train_coco2014.json']
        test_file = ['data/coco/val_coco2014.json']
        step_size = 5
    if args.dataset == "wider":
        train_file = ['data/wider/trainval_wider.json']
        test_file = ["data/wider/test_wider.json"]
        step_size = 5
        args.train_aug = ["randomflip"]
    if args.dataset == "mlrsnet":
        train_file = ["data/mlrsnet/train_mlrsnet.json"]
        test_file = ['data/mlrsnet/val_mlrsnet.json']
        step_size = 4
    if args.dataset == "bigearth":
        train_file = ["data/bigearth/train_bigearth.json"]
        test_file = ['data/bigearth/val_bigearth.json']
        step_size = 4


    train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    #two data trainers
    

    # optimizer and warmup
    backbone, classifier = [], []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier.append(param)
        else:
            backbone.append(param)
    optimizer = optim.SGD(
        [
            {'params': backbone, 'lr': args.lr},
            {'params': classifier, 'lr': args.lr * 10}
        ],
        momentum=args.momentum, weight_decay=args.w_d)    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    iter_per_epoch = len(train_loader)
    if args.warmup_epoch > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warmup_epoch)
    else:
        warmup_scheduler = None

    os.makedirs(f"checkpoint/{args.model}", exist_ok=True)

    train_loss = f"checkpoint/{args.model}/train_loss.csv"
    val_loss = f"checkpoint/{args.model}/val_loss.csv"

    train_results = f"checkpoint/{args.model}/train_results.csv"#unused for now
    val_results = f"checkpoint/{args.model}/val_results.csv"

    with open(train_loss, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'training_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with open(val_loss, mode='w', newline='') as csvfile:
        fieldnames = ['epoch', 'validation_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with open(train_results, mode='w', newline='') as csvfile:
        fieldnames = ['epoch','mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


    with open(val_results, mode='w', newline='') as csvfile:
        fieldnames = ['epoch','mAP', 'CP', 'CR', 'CF1', 'OP', 'OR', 'OF1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()





    # training and validation
    for i in range(1, args.total_epoch + 1):
        train(i, args, model, train_loader, optimizer, warmup_scheduler, train_file, train_loss, train_results)
        torch.save(model.state_dict(), "checkpoint/{}/epoch_{}.pth".format(args.model, i))
        val(i, args, model, test_loader, test_file, val_results, val_loss)
        scheduler.step()



if __name__ == "__main__":
    main()
