from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from .csra import CSRA, MHA
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class MacroSoftF1Loss(nn.Module):

    def __init__(self):
        super(MacroSoftF1Loss, self).__init__()


    def forward(self, y, logit):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """

        y = y.float()
        #y_hat = y_hat.float()
        y_hat = torch.sigmoid(logit)
        tp = (y_hat * y).sum(dim=0) # soft
        fp = (y_hat * (1-y)).sum(dim=0) # soft
        fn = ((1-y_hat) * y).sum(dim=0) # soft
        soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16) #f1 scpre with beta=1
        cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = cost.mean()
        return macro_cost

class MacroSoftF2Loss(nn.Module):

    def __init__(self):
        super(MacroSoftF2Loss, self).__init__()


    def forward(self, y, logit):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """

        y = y.float()
        #y_hat = y_hat.float()
        y_hat = torch.sigmoid(logit)
        tp = (y_hat * y).sum(dim=0) # soft
        fp = (y_hat * (1-y)).sum(dim=0) # soft
        fn = ((1-y_hat) * y).sum(dim=0) # soft
        soft_f2 = 5 * tp / (4*2 * tp + fn + fp + 1e-16) #f2 scpre with beta=2
        cost = 1 - soft_f2  # reduce 1 - soft-f2 in order to increase soft-f2
        macro_cost = cost.mean()
        return macro_cost

class MacroSoftF0_5Loss(nn.Module):

    def __init__(self):
        super(MacroSoftF0_5Loss, self).__init__()


    def forward(self, y, logit):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """

        y = y.float()
        #y_hat = y_hat.float()
        y_hat = torch.sigmoid(logit)
        tp = (y_hat * y).sum(dim=0) # soft
        fp = (y_hat * (1-y)).sum(dim=0) # soft
        fn = ((1-y_hat) * y).sum(dim=0) # soft
        soft_f0_5 = 1.25 * tp / (0.25*2 * tp + fn + fp + 1e-16) #f2 scpre with beta=2
        cost = 1 - soft_f0_5  # reduce 1 - soft-f2 in order to increase soft-f2
        macro_cost = cost.mean()
        return macro_cost


#the main model architecture with ResNet backbone

class ResNet_CSRA(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_heads, lam, num_classes, depth=101, input_dim=6144, cutmix=None): #change from 2048 to 6144 (3072)
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet_CSRA, self).__init__(self.block, self.layers)
        self.init_weights(pretrained=True, cutmix=cutmix) #uses pretrained resnet weights

        self.classifier = MHA(num_heads, lam, input_dim, num_classes) # uses multiheaded attention as a classifier
        self.loss_func = F.binary_cross_entropy_with_logits #for binary cross entropy
        #self.loss_func = MacroSoftF1Loss() #for macro f1
        #self.loss_func = MacroSoftF2Loss()
        self.dropout = nn.Dropout(p=0.5)



    def backbone(self, x): #this is the feature extraction with input tensor x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)
        
        return x


    def forward_train(self, x1, x2, x3, target): #for training and validation set
        x1 = self.backbone(x1) #get features from input tensor
        x2 = self.backbone(x2)
        x3 = self.backbone(x3)



        x = torch.cat((x1, x2, x3), dim=1)


        x = self.dropout(x)




        logit = self.classifier(x) #gets logits from classifier

        loss = self.loss_func(logit, target, reduction="mean") #for binary cross entropy loss
        #loss = self.loss_func(target, logit) #added for macro f1
        return logit, loss

    def forward_test(self, x1, x2, x3):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x3 = self.backbone(x3)

        x = torch.cat((x1, x2, x3), dim=1)



        logit = self.classifier(x) #only gets the logit (for testing set)
        return logit

    def forward(self, x1, x2, x3, target=None): #modify this to accept more than one img
        if target is not None:
            return self.forward_train(x1, x2, x3, target)
        else:
            return self.forward_test(x1, x2, x3)



    def init_weights(self, pretrained=True, cutmix=None): #initialise the weights
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)

        model_dict = self.state_dict()
        try:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.load_state_dict(pretrained_dict)
        except:
            logger = logging.getLogger()
            logger.info(
                "the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
            state_dict = self._keysFix(model_dict, state_dict)
            self.load_state_dict(state_dict)

        # remove the original 1000-class fc
        self.fc = nn.Sequential() 
