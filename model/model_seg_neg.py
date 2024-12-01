import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder
"""
Borrow from https://github.com/facebookresearch/dino
"""


class network(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.init_momentum = init_momentum
        
        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4 

        self.pooling = F.adaptive_max_pool2d
        
        self.attn_proj = nn.Conv2d(in_channels=24, out_channels=1, kernel_size=1, bias=True)
        
        self.decoder = decoder.LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes,)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)


    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)


        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward_proj(self, crops, n_iter=None):

        global_view = crops[:2]
        local_view = crops[2:]

        local_inputs = torch.cat(local_view, dim=0)

        global_output_t = self.encoder.forward_features(torch.cat(global_view, dim=0))[0].detach()
        output_t = self.proj_head_t(global_output_t)

        global_output_s = self.encoder.forward_features(torch.cat(global_view, dim=0))[0]
        local_output_s = self.encoder.forward_features(local_inputs)[0]
        output_s = torch.cat((global_output_s, local_output_s), dim=0)
        output_s = self.proj_head(output_s)
        
        return output_t, output_s
    
    def spatial_pyramid_hybrid_pool(self, x, levels=[1,2,8]):
        n,c,h,w = x.shape
        gamma = 2
        x_p = gamma * F.adaptive_max_pool2d(x, (1,1))
        for i in levels:
            pool = F.avg_pool2d(x, kernel_size=(h//i, w//i), padding=0)
            x_p = x_p + F.adaptive_avg_pool2d(pool, (1,1))
        return x_p/(gamma+len(levels))

    def forward(self, x, cam_only=False,n_iter=None,hp=True):

        cls_token, _x, x_aux,attn_weights = self.encoder.forward_features(x)
        
        
        
        attn_cat = torch.cat(attn_weights[-2:], dim=1)
        
        attn_pred = self.attn_proj(attn_cat)
        
        attn_weights=attn_pred
      
        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x4 = self.to_2D(_x, h, w)
        # B*768*14*14
        _x_aux = self.to_2D(x_aux, h, w)

        seg = self.decoder(_x4)

        if cam_only:

            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()

            return cam_aux, cam
            
        cls_aux = self.pooling(_x_aux, (1,1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1,1))
        
        if hp:
            cls_x4=self.spatial_pyramid_hybrid_pool(_x4)
        else:
            cls_x4 = self.pooling(_x4, (1,1))
            
            
            
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        cls_aux = cls_aux.view(-1, self.num_classes-1)
        
       
        return cls_token,cls_x4, seg, _x4, cls_aux,attn_weights

            