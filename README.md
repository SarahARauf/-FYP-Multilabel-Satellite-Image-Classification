# CSRA 
This is the official code of ICCV 2021 paper:<br>
[Residual Attention: A Simple But Effective Method for Multi-Label Recoginition](https://arxiv.org/abs/2108.02456)<br>

![attention](https://github.com/Kevinz-code/CSRA/blob/master/utils/pipeline.PNG)

### Demo, Train and Validation code have been released! (including VIT on Wider-Attribute)
This package is developed by Mr. Ke Zhu (http://www.lamda.nju.edu.cn/zhuk/) and we have just finished the implementation code of ViT models. If you have any question about the code, please feel free to contact Mr. Ke Zhu (zhuk@lamda.nju.edu.cn). The package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Jianxin Wu (mail to 
wujx2001@gmail.com).

## Requirements
- Python 3.7
- pytorch 1.6
- torchvision 0.7.0
- pycocotools 2.0
- tqdm 4.49.0, pillow 7.2.0

## Dataset
We expect VOC2007, COCO2014 and Wider-Attribute dataset to have the following structure:
```
Dataset/
|-- VOCdevkit/
|---- VOC2007/
|------ JPEGImages/
|------ Annotations/
|------ ImageSets/
......
|-- COCO2014/
|---- annotations/
|---- images/
|------ train2014/
|------ val2014/
......
|-- WIDER/
|---- Annotations/
|------ wider_attribute_test.json
|------ wider_attribute_trainval.json
|---- Image/
|------ train/
|------ val/
|------ test/
...
```
Then directly run the following command to generate json file (for implementation) of these datasets.
```shell
python utils/prepare/prepare_voc.py  --data_path  Dataset/VOCdevkit
python utils/prepare/prepare_coco.py --data_path  Dataset/COCO2014
python utils/prepare/prepare_wider.py --data_path Dataset/WIDER
```
which will automatically result in annotation json files in *./data/voc07*, *./data/coco* and *./data/wider*

## Demo
We provide prediction demos of our models. The demo images (picked from VCO2007) have already been put into *./utils/demo_images/*, you can simply run demo.py by using our CSRA models pretrained on VOC2007:
```shell
CUDA_VISIBLE_DEVICES=0 python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset voc07 --load_from OUR_VOC_PRETRAINED.pth --img_dir utils/demo_images
```
which will output like this:
```shell
utils/demo_images/000001.jpg prediction: dog,person,
utils/demo_images/000004.jpg prediction: car,
utils/demo_images/000002.jpg prediction: train,
...
```


## Validation
We provide pretrained models on [Google Drive](https://www.google.com/drive/) for validation. ResNet101 trained on ImageNet with **CutMix** augmentation can be downloaded 
[here](https://drive.google.com/u/0/uc?export=download&confirm=kYfp&id=1T4AxsAO2tszvhn62KFN5kaknBtBZIpDV).
|Dataset      | Backbone  |   Head nums   |   mAP(%)  |  Resolution     | Download   |
|  ---------- | -------   |  :--------:   | ------ |  :---:          | --------   |
| VOC2007     |ResNet-101 |     1         |  94.7  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=bXcv&id=1cQSRI_DWyKpLa0tvxltoH9rM4IZMIEWJ)   |
| VOC2007     |ResNet-cut |     1         |  95.2  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=otx_&id=1bzSsWhGG-zUNQRMB7rQCuPMqLZjnrzFh)  |
| VOC2007 (extra)    |ResNet-cut |     1         |  96.8  |  448x448 |[download](https://drive.google.com/u/0/uc?id=1XgVE3Q3vmE8hjdDjqow_2GyjPx_5bDjU&export=download)  |
| COCO        |ResNet-101 |     4         |  83.3  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=EWtH&id=1e_WzdVgF_sQc--ubN-DRnGVbbJGSJEZa)   |
| COCO        |ResNet-cut |     6         |  85.6  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=uEcu&id=17FgLUe_vr5sJX6_TT-MPdP5TYYAcVEPF)   |
| COCO        |VIT_L16_224 |     8      |  86.5  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=1Rmm&id=1TTzCpRadhYDwZSEow3OVdrh1TKezWHF_)|
| COCO        |VIT_L16_224* |     8     |  86.9  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=xpbJ&id=1zYE88pmWcZfcrdQsP8-9JMo4n_g5pO4l)|
| Wider       |VIT_B16_224|     1         |  89.0  |  224x224 |[download](https://drive.google.com/u/0/uc?id=1qkJgWQ2EOYri8ITLth_wgnR4kEsv0bfj&export=download)   |
| Wider       |VIT_L16_224|     1         |  90.2  |  224x224 |[download](https://drive.google.com/u/0/uc?id=1da8D7UP9cMCgKO0bb1gyRvVqYoZ3Wh7O&export=download)   |

For voc2007, run the following validation example:
```shell
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from MODEL.pth
```
For coco2014, run the following validation example:
```shell
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 4 --lam 0.5 --dataset coco --num_cls 80  --load_from MODEL.pth
```
For wider attribute with ViT models, run the following
```shell
CUDA_VISIBLE_DEVICES=0 python val.py --model vit_B16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14  --load_from ViT_B16_MODEL.pth
CUDA_VISIBLE_DEVICES=0 python val.py --model vit_L16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14  --load_from ViT_L16_MODEL.pth
```
To provide pretrained VIT models on Wider-Attribute dataset, we retrain them recently, which has a slightly different performance (~0.1%mAP) from what has been presented in our paper. The structure of the VIT models is the initial VIT version (**An image is worth 16x16 words: Transformers for image recognition at scale**, [link](https://arxiv.org/pdf/2010.11929.pdf)) and the implementation code of the VIT models is derived from [http://github.com/rwightman/pytorch-image-models/](http://github.com/rwightman/pytorch-image-models/). 
## Training
#### VOC2007
You can run either of these two lines below 
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20 --cutmix CutMix_ResNet101.pth
```
Note that the first command uses the Official ResNet-101 backbone while the second command uses the ResNet-101 pretrained on ImageNet with CutMix augmentation
[link](https://drive.google.com/u/0/uc?export=download&confirm=kYfp&id=1T4AxsAO2tszvhn62KFN5kaknBtBZIpDV) (which is supposed to gain better performance).

#### MS-COCO
run the ResNet-101 with 4 heads
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 6 --lam 0.5 --dataset coco --num_cls 80
```
run the ResNet-101 (pretrained with CutMix) with 6 heads
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 6 --lam 0.4 --dataset coco --num_cls 80 --cutmix CutMix_ResNet101.pth
```
You can feel free to adjust the hyper-parameters such as number of attention heads (--num_heads), or the Lambda (--lam). Still, the default values of them in the above command are supposed to be the best.

#### Wider-Attribute
run the VIT_B16_224 with 1 heads
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model vit_B16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```
run the VIT_L16_224 with 1 heads
```shell
CUDA_VISIBLE_DEVICES=0,1 python main.py --model vit_L16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```
Note that the VIT_L16_224 model consume larger GPU space, so we use 2 GPUs to train them.
## Notice
To avoid confusion, please note the **4 lines of code** in Figure 1 (in paper) is only used in **test** stage (without training), which is our motivation. When our model is end-to-end training and testing, **multi-head-attention** (H=1, H=2, H=4, etc.) is used with different T values. Also, when H=1 and T=infty, the implementation code of **multi-head-attention** is exactly the same with Figure 1.

We didn't use any new augmentation such as **Autoaugment, RandAugment** in our ResNet series models.

## Acknowledgement

We thank Lin Sui (http://www.lamda.nju.edu.cn/suil/) for his initial contribution to this project.

###swin###

python main.py --model swin_base_patch4_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset mlrsnet --num_cls 60 --total_epoch 100

python main.py --model swin_base_patch4_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset bigearth --num_cls 43 --total_epoch 100


python val.py --num_heads 1 --lam 0.3 --dataset mlrsnet --num_cls 60  --load_from E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\epoch_100.pth --img_size 224 --model swin_base_patch4_224



python val.py --num_heads 1 --lam 0.3 --dataset mlrsnet --num_cls 60  --load_from E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\epoch_100.pth --img_size 224 --model swin_base_patch4_224

python val.py --num_heads 1 --lam 0.3 --dataset mlrsnet --num_cls 43  --load_from E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\epoch_100.pth --img_size 224 --model swin_base_patch4_224


python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224 --title "Precision using SwinT" --y_label "precision"

python demo.py --model swin_base_patch4_224 --num_heads 1 --lam 0.3 --dataset mlrsnet --load_from E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224_saved\epoch_100.pth --img_dir utils/demo_test_mlrsnet --img_size 224 --num_cls 60



###vit###

python main.py --model vit_B16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset mlrsnet --num_cls 60 --total_epoch 1

python main.py --model vit_B16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset bigearth --num_cls 43 --total_epoch 100

python val.py --num_heads 1 --lam 0.3 --dataset mlrsnet --num_cls 60  --load_from E:\Sarah\CSRA-master\checkpoint\vit_B16_224_BCE\epoch_100.pth --img_size 224 --model vit_B16_224

python val.py --num_heads 1 --lam 0.3 --dataset bigearth --num_cls 43  --load_from E:\Sarah\CSRA-master\checkpoint\vit_B16_224\epoch_100.pth --img_size 224 --model vit_B16_224



python demo.py --model vit_B16_224 --num_heads 1 --lam 0.3 --dataset mlrsnet --load_from E:\Sarah\CSRA-master\checkpoint\vit_B16_224_BCE\epoch_100.pth --img_dir utils/demo_test_mlrsnet --img_size 224 --num_cls 60

python demo.py --model vit_B16_224 --num_heads 1 --lam 0.3 --dataset bigearth --load_from E:\Sarah\CSRA-master\checkpoint\vit_B16_224\epoch_100.pth --img_dir utils/demo_test_rgb --img_size 224 --num_cls 43

python demo.py --model vit_B16_224 --num_heads 1 --lam 0.3 --dataset bigearth --load_from E:\Sarah\CSRA-master\checkpoint\vit_B16_224\epoch_100.pth --img_dir utils/rgb_new --img_size 224 --num_cls 43



python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\vit_B16_224_BCE\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\vit_B16_224_BCE\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\vit_B16_224_BCE --title "Precision using VIT" --y_label "precision"

######
python main.py --num_heads 1 --lam 0.1 --dataset mlrsnet --num_cls 60 --img_size 200 --total_epoch 100

python main.py --num_heads 1 --lam 0.1 --dataset bigearth --num_cls 43 --img_size 120 --total_epoch 100

python main.py --num_heads 1 --lam 0.1 --dataset bigearth --train_aug ["randomflip", "resizedcrop"] --num_cls 43 --img_size 120 --total_epoch 100


python val.py --num_heads 1 --lam 0.1 --dataset mlrsnet --num_cls 60  --load_from E:\Sarah\CSRA-master\checkpoint\resnet101\epoch_100.pth --img_size 200


python val.py --num_heads 1 --lam 0.1 --dataset bigearth --num_cls 43  --load_from E:\Sarah\CSRA-master\checkpoint\resnet101\epoch_100.pth --img_size 120

python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset mlrsnet --load_from E:\Sarah\CSRA-master\checkpoint\resnet101_binary\epoch_100.pth --img_dir utils/demo_test --img_size 200 --num_cls 60

python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset bigearth --load_from E:\Sarah\CSRA-master\checkpoint\resnet101_bigearthRGB\epoch_100.pth --img_dir utils/demo_test --img_size 120 --num_cls 43

python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset bigearth --load_from E:\Sarah\CSRA-master\checkpoint\resnet101_BigearthVEG\epoch_100.pth --img_dir utils/veg_new --img_size 120 --num_cls 43

python graph.py --csv E:\Sarah\CSRA-master\checkpoint\resnet101\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Validation Performance Metric"
python graph.py --csv E:\Sarah\CSRA-master\checkpoint\resnet101\train_loss.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Training Loss"


#for training vs validation Precision (macro)
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101_macro\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101_macro\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_macro --title "Precision using CSRA_MSf1" --y_label "precision"

#for training vs validation Recall (macro)
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101_macro\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101_macro\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_macro --title "Recall using CSRA_MSf1" --y_label "recall"

#for training vs validation f1 (macro)
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101_macro\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101_macro\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_macro --title "f1 using CSRA_MSf1" --y_label "f1"

#for training vs validation loss (macro)
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101_macro\train_loss.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101_macro\val_loss.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_macro --title "Loss using CSRA_MSf1" --y_label "loss"

#for test (macro)
python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\resnet101_macro\class_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_macro --title "Sample Size vs Metrics CSRA_MSf1" --x_label "Class Label Sample Size"


#----
#for training vs validation Precision (binary)
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101_binary\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101_binary\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_binary --title "Precision using CSRA_BCE" --y_label "precision"

#for training vs validation Recall (binary)
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101_binary\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101_binary\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_binary --title "Recall using CSRA_BCE" --y_label "recall"

#for training vs validation f1 (binary)
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101_binary\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101_binary\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_binary --title "f1 using CSRA_BCE" --y_label "f1"

#for training vs validation loss (binary)
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101_binary\train_loss.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101_binary\val_loss.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_binary --title "Loss using CSRA_BCE" --y_label "loss"

#for test (binary)
python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\resnet101_binary\class_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101_binary --title "Sample Size vs Metrics CSRA_BCE" --x_label "Class Label Sample Size"

python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\resnet101\class_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Sample Size vs Metrics ResNet-101" --x_label "Class Label Sample Size"

#----
#for training vs validation Precision 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Precision using CSRA_BCE for BEN-VEG" --y_label "precision"

#for training vs validation Recall 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Recall using CSRA_BCE for BEN-VEG" --y_label "recall"

#for training vs validation f1 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "f1 using CSRA_BCE for BEN-VEG" --y_label "f1"

#for training vs validation loss
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101\train_loss.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101\val_loss.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Loss using CSRA_BCE for BEN-VEG" --y_label "loss"

#for test 
python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\resnet101\class_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Sample Size vs Metrics CSRA_BCE for BEN-AGRI" --x_label "Class Label Sample Size"

#----
#for training vs validation Precision 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Precision using CSRA_BCE for BEN-FC-NWAR-SIR" --y_label "precision"

#for training vs validation Recall 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Recall using CSRA_BCE for BEN-FC-NWAR-SIR" --y_label "recall"

#for training vs validation f1 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "f1 using CSRA_BCE for BEN-FC-NWAR-SIR" --y_label "f1"

#for training vs validation loss
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\resnet101\train_loss.csv --csv_val E:\Sarah\CSRA-master\checkpoint\resnet101\val_loss.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Loss using CSRA_BCE for BEN-FC-NWAR-SIR" --y_label "loss"

#for test 
python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\resnet101\class_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet101 --title "Sample Size vs Metrics CSRA_BCE for BEN-" --x_label "Class Label Sample Size"




#---

python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\resnet50_binary\test_results_BCE.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet50_binary --title "Sample Size vs Metrics Resnet50_BCE" --x_label "Class Label Sample Size"

python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\resnet50_macro\test_results_MSf1.csv --output_dir E:\Sarah\CSRA-master\checkpoint\resnet50_macro --title "Sample Size vs Metrics Resnet50_MSf1" --x_label "Class Label Sample Size"


#-----------------

###### SWIN graphs ##############
#for training vs validation Precision 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224 --title "Precision using SwinT" --y_label "precision"

#for training vs validation Recall 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224 --title "Recall using SwinT" --y_label "recall"

#for training vs validation f1 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224 --title "f1 using SwinT" --y_label "f1"

#for training vs validation loss
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\train_loss.csv --csv_val E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\val_loss.csv --output_dir E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224 --title "Loss using SwinT" --y_label "loss"

#for test  (Class imbalance)
python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224\class_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\swin_base_patch4_224 --title "Sample Size vs Metrics SwinT" --x_label "Class Label Sample Size"

###VIT GRAPHS#####
#for training vs validation Precision 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\vit_B16_224\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\vit_B16_224\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\vit_B16_224 --title "Precision using VIT" --y_label "precision"

#for training vs validation Recall 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\vit_B16_224\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\vit_B16_224\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\vit_B16_224 --title "Recall using VIT" --y_label "recall"

#for training vs validation f1 
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\vit_B16_224\train_results.csv --csv_val E:\Sarah\CSRA-master\checkpoint\vit_B16_224\val_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\vit_B16_224 --title "f1 using VIT" --y_label "f1"

#for training vs validation loss
python graph.py --csv_train E:\Sarah\CSRA-master\checkpoint\vit_B16_224\train_loss.csv --csv_val E:\Sarah\CSRA-master\checkpoint\vit_B16_224\val_loss.csv --output_dir E:\Sarah\CSRA-master\checkpoint\vit_B16_224 --title "Loss using VIT" --y_label "loss"

#for test  (Class imbalance)
python graph.py --csv_test E:\Sarah\CSRA-master\checkpoint\vit_B16_224\class_results.csv --output_dir E:\Sarah\CSRA-master\checkpoint\vit_B16_224 --title "Sample Size vs Metrics VIT" --x_label "Class Label Sample Size"

#-----------------

img saved to: E:/Downloads/BigEarthProcessed/S2A_MSIL2A_20180225T114351_54_7_ir.jpg
img saved to: E:/Downloads/BigEarthProcessed/S2A_MSIL2A_20180225T114351_54_7_agri.jpg
img saved to: E:/Downloads/BigEarthProcessed/S2A_MSIL2A_20180225T114351_54_7_lw.jpg
Traceback (most recent call last):
  File "rasterio\\_base.pyx", line 310, in rasterio._base.DatasetBase.__init__
  File "rasterio\\_base.pyx", line 221, in rasterio._base.open_dataset
  File "rasterio\\_err.pyx", line 221, in rasterio._err.exc_wrap_pointer
rasterio._err.CPLE_OpenFailedError: 'BigEarthNet-v1.0\S2A_MSIL2A_20180225T114351_54_7\S2A_MSIL2A_20180225T114351_54_7_B12.tif' not recognized as a supported file format.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:\Downloads\ProcessImages.py", line 157, in <module>
    main()
  File "E:\Downloads\ProcessImages.py", line 133, in main
    mergeImg3band(paths, new_folder, key)
  File "E:\Downloads\ProcessImages.py", line 67, in mergeImg3band
    with rio.open(i, 'r') as f:
         ^^^^^^^^^^^^^^^^
  File "C:\Users\Admin\.conda\envs\sarah\Lib\site-packages\rasterio\env.py", line 451, in wrapper
    return f(*args, **kwds)
           ^^^^^^^^^^^^^^^^
  File "C:\Users\Admin\.conda\envs\sarah\Lib\site-packages\rasterio\__init__.py", line 317, in open
    dataset = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "rasterio\\_base.pyx", line 312, in rasterio._base.DatasetBase.__init__
rasterio.errors.RasterioIOError: 'BigEarthNet-v1.0\S2A_MSIL2A_20180225T114351_54_7\S2A_MSIL2A_20180225T114351_54_7_B12.tif' not recognized as a supported file format.


#---------------------
Complex cultivation patterns: 21470
Land principally occupied by agriculture, with significant areas of natural vegetation: 29427
Broad-leaved forest: 30236
Coniferous forest: 42337
Mixed forest: 43381
Discontinuous urban fabric: 14058
Non-irrigated arable land: 39260
Industrial or commercial units: 2610
Vineyards: 1903
Transitional woodland/shrub: 34568
Peatbogs: 4682
Water bodies: 16853
Sparsely vegetated areas: 303
Sea and ocean: 16309
Permanently irrigated land: 2680
Annual crops associated with permanent crops: 1377
Inland marshes: 1200
Agro-forestry areas: 6062
Sclerophyllous vegetation: 2179
Pastures: 20666
Water courses: 2126
Continuous urban fabric: 2194
Sport and leisure facilities: 1114
Olive groves: 2434
Rice fields: 775
Moors and heathland: 1210
Beaches, dunes, sands: 317
Fruit trees and berry plantations: 943
Green urban areas: 338
Salt marshes: 333
Intertidal flats: 197
Salines: 88
Construction sites: 255
Natural grassland: 2557
Bare rock: 662
Mineral extraction sites: 887
Airports: 183
Coastal lagoons: 301
Estuaries: 220
Road and rail networks and associated land: 666
Port areas: 108
Burnt areas: 66
Dump sites: 189


####new train, test val spli

train_df:
Class Label Counts after Downsampling:
Land principally occupied by agriculture, with significant areas of natural vegetation: 20642
Broad-leaved forest: 21154
Natural grassland: 1793
Transitional woodland/shrub: 24217
Mixed forest: 30335
Water bodies: 11735
Discontinuous urban fabric: 9853
Non-irrigated arable land: 27475
Complex cultivation patterns: 15110
Permanently irrigated land: 1882
Agro-forestry areas: 4238
Pastures: 14533
Sea and ocean: 11364
Coniferous forest: 29611
Inland marshes: 815
Peatbogs: 3286
Water courses: 1494
Dump sites: 128
Salt marshes: 231
Salines: 53
Moors and heathland: 869
Sport and leisure facilities: 801
Estuaries: 155
Vineyards: 1302
Beaches, dunes, sands: 215
Coastal lagoons: 200
Olive groves: 1703
Sclerophyllous vegetation: 1522
Rice fields: 544
Bare rock: 478
Annual crops associated with permanent crops: 965
Continuous urban fabric: 1540
Industrial or commercial units: 1802
Road and rail networks and associated land: 467
Green urban areas: 235
Mineral extraction sites: 637
Fruit trees and berry plantations: 641
Sparsely vegetated areas: 216
Construction sites: 190
Port areas: 78
Airports: 118
Intertidal flats: 141
Burnt areas: 48


val_df:
Class Label Counts after Downsampling:
Sea and ocean: 2483
Non-irrigated arable land: 5855
Complex cultivation patterns: 3205
Broad-leaved forest: 4575
Transitional woodland/shrub: 5260
Discontinuous urban fabric: 2065
Industrial or commercial units: 389
Mixed forest: 6544
Water courses: 330
Agro-forestry areas: 929
Pastures: 3008
Permanently irrigated land: 401
Coniferous forest: 6373
Continuous urban fabric: 334
Peatbogs: 713
Land principally occupied by agriculture, with significant areas of natural vegetation: 4378
Sclerophyllous vegetation: 353
Water bodies: 2538
Moors and heathland: 177
Natural grassland: 392
Inland marshes: 211
Rice fields: 131
Vineyards: 318
Olive groves: 358
Mineral extraction sites: 125
Beaches, dunes, sands: 57
Road and rail networks and associated land: 98
Sport and leisure facilities: 160
Annual crops associated with permanent crops: 206
Bare rock: 86
Green urban areas: 42
Fruit trees and berry plantations: 147
Intertidal flats: 23
Estuaries: 25
Airports: 33
Salt marshes: 50
Coastal lagoons: 53
Construction sites: 37
Salines: 17
Sparsely vegetated areas: 42
Dump sites: 32
Burnt areas: 8
Port areas: 15


test_Df:
Class Label Counts after Downsampling:
Sea and ocean: 2462
Broad-leaved forest: 4507
Coniferous forest: 6353
Mixed forest: 6502
Transitional woodland/shrub: 5091
Water bodies: 2580
Complex cultivation patterns: 3155
Land principally occupied by agriculture, with significant areas of natural vegetation: 4407
Natural grassland: 372
Non-irrigated arable land: 5930
Pastures: 3125
Discontinuous urban fabric: 2140
Bare rock: 98
Industrial or commercial units: 419
Continuous urban fabric: 320
Vineyards: 283
Olive groves: 373
Sclerophyllous vegetation: 304
Sport and leisure facilities: 153
Construction sites: 28
Agro-forestry areas: 895
Peatbogs: 683
Fruit trees and berry plantations: 155
Inland marshes: 174
Water courses: 302
Annual crops associated with permanent crops: 206
Permanently irrigated land: 397
Rice fields: 100
Moors and heathland: 164
Dump sites: 29
Estuaries: 40
Mineral extraction sites: 125
Sparsely vegetated areas: 45
Port areas: 15
Road and rail networks and associated land: 101
Green urban areas: 61
Beaches, dunes, sands: 45
Coastal lagoons: 48
Airports: 32
Salines: 18
Intertidal flats: 33
Salt marshes: 52
Burnt areas: 10

Test_df len:  17710
Train_df len:  82645
val len:  17710



#
                         C:\ProgramData\Anaconda3
csra                     C:\Users\Admin\.conda\envs\csra
sarah                    C:\Users\Admin\.conda\envs\sarah
swin                     C:\Users\Admin\.conda\envs\swin
tensorflow_nelson        C:\Users\Admin\.conda\envs\tensorflow_nelson
base                  *  C:\Users\Admin\anaconda3


(base) E:\Downloads>conda activate sarah

(sarah) E:\Downloads>python Onehotencoded.py
train_df:
Class Label Counts after Downsampling:
Land principally occupied by agriculture, with significant areas of natural vegetation: 20642
Broad-leaved forest: 21154
Natural grassland: 1793
Transitional woodland/shrub: 24217
Mixed forest: 30335
Water bodies: 11735
Discontinuous urban fabric: 9853
Non-irrigated arable land: 27475
Complex cultivation patterns: 15110
Permanently irrigated land: 1882
Agro-forestry areas: 4238
Pastures: 14533
Sea and ocean: 11364
Coniferous forest: 29611
Inland marshes: 815
Peatbogs: 3286
Water courses: 1494
Dump sites: 128
Salt marshes: 231
Salines: 53
Moors and heathland: 869
Sport and leisure facilities: 801
Estuaries: 155
Vineyards: 1302
Beaches, dunes, sands: 215
Coastal lagoons: 200
Olive groves: 1703
Sclerophyllous vegetation: 1522
Rice fields: 544
Bare rock: 478
Annual crops associated with permanent crops: 965
Continuous urban fabric: 1540
Industrial or commercial units: 1802
Road and rail networks and associated land: 467
Green urban areas: 235
Mineral extraction sites: 637
Fruit trees and berry plantations: 641
Sparsely vegetated areas: 216
Construction sites: 190
Port areas: 78
Airports: 118
Intertidal flats: 141
Burnt areas: 48


trainval_df:
Class Label Counts after Downsampling:
Sea and ocean: 2483
Non-irrigated arable land: 5855
Complex cultivation patterns: 3205
Broad-leaved forest: 4575
Transitional woodland/shrub: 5260
Discontinuous urban fabric: 2065
Industrial or commercial units: 389
Mixed forest: 6544
Water courses: 330
Agro-forestry areas: 929
Pastures: 3008
Permanently irrigated land: 401
Coniferous forest: 6373
Continuous urban fabric: 334
Peatbogs: 713
Land principally occupied by agriculture, with significant areas of natural vegetation: 4378
Sclerophyllous vegetation: 353
Water bodies: 2538
Moors and heathland: 177
Natural grassland: 392
Inland marshes: 211
Rice fields: 131
Vineyards: 318
Olive groves: 358
Mineral extraction sites: 125
Beaches, dunes, sands: 57
Road and rail networks and associated land: 98
Sport and leisure facilities: 160
Annual crops associated with permanent crops: 206
Bare rock: 86
Green urban areas: 42
Fruit trees and berry plantations: 147
Intertidal flats: 23
Estuaries: 25
Airports: 33
Salt marshes: 50
Coastal lagoons: 53
Construction sites: 37
Salines: 17
Sparsely vegetated areas: 42
Dump sites: 32
Burnt areas: 8
Port areas: 15

test_Df:
Class Label Counts after Downsampling:
Sea and ocean: 2462
Broad-leaved forest: 4507
Coniferous forest: 6353
Mixed forest: 6502
Transitional woodland/shrub: 5091
Water bodies: 2580
Complex cultivation patterns: 3155
Land principally occupied by agriculture, with significant areas of natural vegetation: 4407
Natural grassland: 372
Non-irrigated arable land: 5930
Pastures: 3125
Discontinuous urban fabric: 2140
Bare rock: 98
Industrial or commercial units: 419
Continuous urban fabric: 320
Vineyards: 283
Olive groves: 373
Sclerophyllous vegetation: 304
Sport and leisure facilities: 153
Construction sites: 28
Agro-forestry areas: 895
Peatbogs: 683
Fruit trees and berry plantations: 155
Inland marshes: 174
Water courses: 302
Annual crops associated with permanent crops: 206
Permanently irrigated land: 397
Rice fields: 100
Moors and heathland: 164
Dump sites: 29
Estuaries: 40
Mineral extraction sites: 125
Sparsely vegetated areas: 45
Port areas: 15
Road and rail networks and associated land: 101
Green urban areas: 61
Beaches, dunes, sands: 45
Coastal lagoons: 48
Airports: 32
Salines: 18
Intertidal flats: 33
Salt marshes: 52
Burnt areas: 10


Test_df len:  17710
Train_df len:  82645
val len:  17710
