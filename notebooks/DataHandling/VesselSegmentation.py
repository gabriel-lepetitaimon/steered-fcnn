import torch
from model import HeavyResUNet

MODEL_PATH = "VesSeg_heavy.Tnet"
STRIDE = 1024
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def segment_vessels(raw_fundus_dataset):
    d_reshaped = raw_fundus_dataset.reshape('x', shape=2048)
    d_concat = d_reshaped.apply_cv({'pre':'x'}, fundus_preprocessing, keep_parent=True, format='same')\
                         .concat(x=("x", "pre"))
    
    net = HeavyResUNet(6)
    net.load_state_dict(torch.load(MODEL_PATH))
    net = net.to(DEVICE)
    
    d_pred = d_concat.patches('x', STRIDE+127, stride=STRIDE).apply_torch('prediction', net, device=DEVICE)\
                     .subgen().unpatch('x,prediction')
    
    return d_pred.apply('vessels', lambda prediction: prediction>=0)


#############################################
#         DEPENDENCIES DATASET              #
#############################################
import cv2
import numpy as np

def fundus_preprocessing(img):
    mask = cv2.medianBlur(img[:,:,2], 21)>5
    mean_b = np.median(img[:,:,0][mask])
    mean_g = np.median(img[:,:,1][mask])
    mean_r = np.median(img[:,:,2][mask])
    mean_channels = [mean_b, mean_g, mean_r]
    img = np.clip(img.astype(np.float32) - cv2.medianBlur(img, 151)*np.expand_dims(mask, 2) + np.asarray(mean_channels).astype(np.uint8), 0, 255)
    img = cv2.GaussianBlur(img, (3,3), 0)    
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    l = lab_planes[0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return bgr


#############################################
#           DEPENDENCIES MODEL              #
#############################################
from torch import nn
import torch.nn.functional as F


def clip_center(tensor, shape):
    s = tensor.shape[-2:]
    y0 = (s[0]-shape[-2])//2
    x0 = (s[1]-shape[-1])//2
    return tensor[..., y0:y0+shape[-2], x0:x0+shape[-1]]


class ConvBN(nn.Module):
    def __init__(self, kernel, n_in, n_out=None, stride=1, relu=False, padding=0):
        super(ConvBN, self).__init__()
        
        if n_out is None:
            n_out = n_in
        
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(n_out)
        
        self.model = [self.conv, self.bn]
        if relu:
            self.model.append(nn.ReLU())
        self.seq = nn.Sequential(*self.model)
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, x):
        return self.seq(x)


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, depth=2):
        super(ResConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if in_channels != out_channels:
            self.conv0 = ConvBN(1, in_channels, out_channels, relu=False)
        else:
            self.conv0 = lambda x: x

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        
        self.m = []
        for i in range(depth):
            if i < depth-1:
                conv = ConvBN(3, out_channels, relu=True)
            else:
                conv = ConvBN(3, out_channels, relu=False)

                # Normed initialized to 0: ResConvBlock act as an identity when the training start
                # [https://arxiv.org/abs/1706.02677]
                nn.init.uniform_(conv.bn.weight, 0, 0.3)

            self.m += conv.model
            setattr(self, 'conv%s' % (i+1), conv)
            
        self.convSeq = nn.Sequential(*self.m)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv0(x)
        r = self.convSeq(x)
        r = r + clip_center(x, r.shape)
        return self.relu(r)
    
    
class ResUNet(nn.Module):
    def __init__(self, n_in, n_out=1, p_dropout=0):
        super(ResUNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        
        # --- MODEL ---
        self.conv_bn0 = ConvBN(5, n_in, 16, relu=True)
        
        # Down
        self.res1a = ResConvBlock(16, 32)
        self.pool1v2 = ConvBN(3, 32, 64, stride=2, relu=True)
        self.res2b = ResConvBlock(64)
        self.pool2v3 = ConvBN(3, 64, 128, stride=2, relu=True)
        self.res3c = ResConvBlock(128)
        self.pool3v4 = ConvBN(3, 128, 128, stride=2, relu=True)
        self.res4d = ResConvBlock(128)

        # Up
        self.decode3 = ResConvBlock(2*128, 64)
        self.decode2 = ResConvBlock(2*64, 32)
        self.decode1 = ResConvBlock(2*32, 16)

        # End
        self.final_conv = nn.Conv2d(16, n_out, kernel_size=1)
        
        self.dropout = torch.nn.Dropout(p_dropout) if p_dropout else lambda x: x
        
    def forward(self, x):
        x = self.conv_bn0(x)
        
        # Down
        x1 = self.res1a(x)
        x2 = self.res2b( self.pool1v2(x1) )
        x3 = self.res3c( self.pool2v3(x2) )
        x4 = self.res4d( self.pool3v4(x3) )
        
        b = self.dropout(x4)
        
        # Up
        x4up = F.interpolate(b, scale_factor=2)
        cat3 = torch.cat((self.dropout(clip_center(x3, x4up.shape)), x4up), 1)
        y3 = self.decode3(cat3)
        
        x3up = F.interpolate(y3, scale_factor=2)
        cat2 = torch.cat((self.dropout(clip_center(x2, x3up.shape)), x3up), 1)
        y2 = self.decode2(cat2)
        
        
        x2up = F.interpolate(y2, scale_factor=2)
        cat1 = torch.cat((self.dropout(clip_center(x1, x2up.shape)), x2up), 1)
        y1 = self.decode1(cat1)
        
        # End
        return self.final_conv(y1)
