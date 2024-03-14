import lpips
import torch
import torch.nn as nn

import net

import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import math
import torch.nn.functional as F

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(1, -1, 1, 1)
    x = (x - mean) / std
    return x


class Metric(nn.Module):

    def __init__(self, metric_type='vgg'):
        super(Metric, self).__init__()
        self.metric_type = metric_type
        if metric_type == 'vgg':
            self.model = lpips.pn.vgg16()
        elif metric_type == 'alexnet':
            self.model = lpips.pn.alexnet()
        elif metric_type == 'ssim':
            ssim_module = SSIM(data_range=1, size_average=False, channel=3) # channel=1 for grayscale images
            self.model = ssim_module
        elif metric_type == 'ms-ssim':
            ms_ssim_module = MS_SSIM(data_range=1, size_average=False, channel=3)
            self.model = ms_ssim_module
        else:
            raise ValueError(f'Invalid metric type: {metric_type}')

    def forward(self, x, y):
        if self.metric_type == 'ssim' or self.metric_type == 'ms-ssim':
            dist = self.model(x, y)
            return dist
        
        else:
            features_x = self.model(normalize(x))._asdict()
            features_y = self.model(normalize(y))._asdict()
            
            dist = 0.0
            for layer in features_x.keys():
                dist += torch.mean(torch.square(features_x[layer] - features_y[layer]), dim=(1, 2, 3))
            return dist / len(features_x)

class LPIPS(nn.Module):

    def __init__(self):
        super(LPIPS, self).__init__()
        self.dist = lpips.LPIPS(net='alex')

    def forward(self, x, y):
        # images must be in range [-1, 1]
        dist = self.dist(2 * x - 1, 2 * y - 1)
        return dist

class LPIPS_vgg(nn.Module):

    def __init__(self):
        super(LPIPS_vgg, self).__init__()
        self.dist = lpips.LPIPS(net='vgg')

    def forward(self, x, y):
        # images must be in range [-1, 1]
        dist = self.dist(2 * x - 1, 2 * y - 1)
        return dist

class PatchSimi(nn.Module):

    def __init__(self, device=None):
        super(PatchSimi, self).__init__()
        self.model = models.vgg19(pretrained=True).features.to(device).eval()
        self.layers = {"11": "conv3"}
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)
        self.kld = torch.nn.KLDivLoss(reduction='batchmean')

        self.device = device

    def get_feats(self, img):
        features = []
        for name, layer in self.model._modules.items():
            img = layer(img)
            if name in self.layers:
                features.append(img)
        return features
    
    def normalize(self, input):
        return transforms.functional.normalize(input, self.norm_mean, self.norm_std)

    def patch_simi_cnt(self, input):
        b, c, h, w = input.size()
        input = torch.transpose(input, 1, 3)
        features = input.reshape(b, h*w, c).div(c)  # resize F_XL into \hat F_XL))
        feature_t = torch.transpose(features, 1, 2)
        patch_simi = F.log_softmax(torch.bmm(features, feature_t), dim=-1)
        return patch_simi.reshape(b, -1)

    def patch_simi_out(self, input):
        b, c, h, w = input.size()
        input = torch.transpose(input, 1, 3)
        features = input.reshape(b, h*w, c).div(c)
        feature_t = torch.transpose(features, 1, 2)
        patch_simi = F.softmax(torch.bmm(features, feature_t), dim=-1)
        return patch_simi.reshape(b, -1)

    def forward(self, input, target):
        src_feats = self.get_feats(self.normalize(input))
        target_feats = self.get_feats(self.normalize(target))
        init_loss = 0.
        for idx in range(len(src_feats)):
            init_loss += F.kl_div(self.patch_simi_cnt(src_feats[idx]), self.patch_simi_out(target_feats[idx]), reduction='batchmean')
        return init_loss

class GramLoss(nn.Module):

    def __init__(self, device=None):
        super(GramLoss, self).__init__()
        self.model = models.vgg19(pretrained=True).features.to(device).eval()
        self.layers = {
            "1":"conv1",
            "6":"conv2",
            "11":"conv3",
            "20":"conv4",
            "29":"conv5"
        }
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)

        self.device = device

    def get_feats(self, img):
        features = []
        for name, layer in self.model._modules.items():
            img = layer(img)
            if name in self.layers:
                features.append(img)
        return features
    
    def normalize(self, input):
        return transforms.functional.normalize(input, self.norm_mean, self.norm_std)

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a, 1, b * c * d).div(math.sqrt(b*c*d))  # resize F_XL into \hat F_XL
        feature_t = torch.transpose(features, 1, 2)
        G = torch.bmm(features, feature_t).sum()  # compute the gram product
        return G

    def loss_from_feat(self, input_feats, target_feats):
        init_loss = 0.
        for idx in range(len(input_feats)):
            loss = F.mse_loss(self.gram_matrix(input_feats[idx]), self.gram_matrix(target_feats[idx]))
            init_loss += loss
        return init_loss

    def forward(self, input, target):
        src_feats = self.get_feats(self.normalize(input))
        target_feats = self.get_feats(self.normalize(target))
        init_loss = 0.
        for idx in range(len(src_feats)):
            loss = F.mse_loss(self.gram_matrix(src_feats[idx]), self.gram_matrix(target_feats[idx]))
            init_loss += loss
        init_loss /= 5
        return init_loss

    def content_forward(self, input):
        src_feats = self.get_feats(self.normalize(input))
        init_loss = 0
        for idx in range(len(src_feats)):
            loss = F.mse_loss(src_feats[idx], self.target_feats[idx])
            init_loss += loss
        return init_loss / len(self.layers)
