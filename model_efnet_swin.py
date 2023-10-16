import math
from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models
from swinv2 import PatchEmbed, SwinTransformerBlock


class ImageCNN(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.efficientnet_b0(pretrained =True)
        self.features.fc = nn.Sequential()
        # for param in self.features.parameters():
        #     param.requires_grad = False

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0]/255.0 - 0.485) / 0.229
    x[:, 1] = (x[:, 1]/255.0 - 0.456) / 0.224
    x[:, 2] = (x[:, 2]/255.0 - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=2):
        super().__init__()

        self._model = models.resnet18(pretrained =True)
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels, 
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)
        # for param in self._model.parameters():
        #     param.requires_grad = False

    def forward(self, inputs):
        features = 0
        for lidar_data in inputs:
            lidar_feature = self._model(lidar_data)
            features += lidar_feature
        return features

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer, 
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, (self.config.n_views) * seq_len * vert_anchors * horz_anchors+ 5*self.config.n_gps, n_embd))

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, gps):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            gps (tensor): ego-gps
        """
        
        bz = image_tensor.shape[0] // (self.config.n_views * self.seq_len)

        h, w = image_tensor.shape[2:4]
#         print('transfo',self.config.n_views , self.seq_len)
        
        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)


        # pad token embeddings along number of tokens dimension
        token_embeddings = (image_tensor).permute(0,1,3,4,2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd) # (B, an * T, C)

        token_embeddings = torch.cat([token_embeddings,gps], dim=1)
        # add (learnable) positional embedding and gps embedding for all tokens
        #print(token_embeddings.shape, self.pos_emb.shape)
        x = self.drop(self.pos_emb + token_embeddings )  # (B, an * T, C)
        x = self.blocks(x) # (B, an * T, C)
        x = self.ln_f(x) # (B, an * T, C)
        pos_tensor_out = x[:, (self.config.n_views) * self.seq_len * self.vert_anchors * self.horz_anchors:, :]
        x = x[:,:(self.config.n_views) * self.seq_len * self.vert_anchors * self.horz_anchors,:]


        x = x.view(bz, (self.config.n_views) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings



        image_tensor_out = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)

        return image_tensor_out, pos_tensor_out


class SwinBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, seq_len,window_size, config, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.config = config
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.seq_len = seq_len
        self.ape = nn.Parameter(torch.zeros(1, (self.config.n_views) * seq_len * self.config.vert_anchors * self.config.horz_anchors+ self.seq_len*self.config.n_gps, self.dim))

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])


    def forward(self, image, gps):

        
            bz = image.shape[0] // (self.config.n_views * self.seq_len)
            h, w = image.shape[2:4]
            image = image.view(bz, self.config.n_views * self.seq_len, -1, h, w)
            x = image.permute(0,1,3,4,2).contiguous()
            x = x.view(bz, -1, self.dim) # (B, an * T, C)
            x = torch.cat([x,gps], dim=1)
            x = x + self.ape

            for blk in self.blocks:
                x = blk(x)

            gps = x[:, (self.config.n_views) * self.seq_len * self.config.vert_anchors * self.config.horz_anchors:, :]
            x =  x[:,:(self.config.n_views) * self.seq_len * self.config.vert_anchors * self.config.horz_anchors,:]

            x  = x.view(bz, (self.config.n_views) * self.seq_len, self.config.vert_anchors, self.config.horz_anchors, self.dim)
            x  = x.permute(0,1,4,2,3).contiguous() # same as token_embeddings
            x  = x[:, :self.config.n_views*self.seq_len, :, :, :].contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)

            return x, gps

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        self.image_encoder_stem = models.efficientnet_b0(pretrained =True).features[0]

        self.image_encoder_layer1 = models.efficientnet_b0(pretrained =True).features[1]
        self.image_encoder_layer2 = models.efficientnet_b0(pretrained =True).features[2]

        self.image_encoder_layer3 = models.efficientnet_b0(pretrained =True).features[3]

        self.image_encoder_layer4 = models.efficientnet_b0(pretrained =True).features[4]

        self.image_encoder_layer5 = models.efficientnet_b0(pretrained =True).features[5]
        self.image_encoder_layer6 = models.efficientnet_b0(pretrained =True).features[6]
        
        self.image_encoder_avgpool = models.efficientnet_b0(pretrained =True).avgpool
        
        self.vel_emb1 = nn.Linear(2, 24)
        self.vel_emb2 = nn.Linear(24, 40)
        self.vel_emb3 = nn.Linear(40, 80)
        self.vel_emb4 = nn.Linear(80, 192)


        self.transformer1 = SwinBasicLayer(dim = 24, 
                                           input_resolution = (self.config.input_resolution),
                                           depth = config.n_layer, num_heads = config.n_head,
                                           window_size = 5, seq_len = self.config.seq_len, config = self.config)

        self.transformer2 = SwinBasicLayer(dim = 40, 
                                           input_resolution =  (self.config.input_resolution),
                                           depth = config.n_layer, num_heads = config.n_head,
                                           window_size = 5, seq_len = self.config.seq_len, config = self.config)
                                           
        self.transformer3 = SwinBasicLayer(dim = 80, 
                                           input_resolution =  (self.config.input_resolution),
                                           depth = config.n_layer, num_heads = config.n_head,
                                           window_size = 5, seq_len = self.config.seq_len, config = self.config)
                                           
        self.transformer4 = SwinBasicLayer(dim = 192, 
                                           input_resolution =  (self.config.input_resolution),
                                           depth = config.n_layer, num_heads = config.n_head,
                                           window_size = 5, seq_len = self.config.seq_len, config = self.config)
        
        
    def forward(self, image_list, gps_list):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            gps (tensor): input gps
        '''
        #if self.image_encoder.normalize:
        #    image_list = [normalize_imagenet(image_input) for image_input in image_list]


        bz, _, h, w = image_list[0].shape
        img_channel = image_list[0].shape[1]
        
        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        print('line410',image_tensor.get_device())
        print('line411',gps_list[0].get_device())
        gps = torch.stack(gps_list, dim=1).view(bz,-1,2)
        print('line412',gps.is_cuda)
        #print(gps.shape)
                
        
        image_features = self.image_encoder_stem(image_tensor)
        image_features = self.image_encoder_layer1(image_features)
        image_features = self.image_encoder_layer2(image_features)
        #print(image_features.shape)
        # fusion at (B, 24, 64, 64)
        image_embd_layer1 = self.avgpool(image_features)
        gps_embd_layer1 = self.vel_emb1(gps)
        
        image_features_layer1, gps_features_layer1 = self.transformer1(image_embd_layer1, gps_embd_layer1)

        #image_features_layer1, gps_features_layer1 = self.transformer1(image_embd_layer1, gps_embd_layer1)
        image_features_layer1 = F.interpolate(image_features_layer1, scale_factor=8, mode='bilinear')
        image_features = image_features + image_features_layer1
        image_features = self.image_encoder_layer3(image_features)


        #print(image_features.shape)
        # fusion at (B, 40, 32, 32)
        image_embd_layer2 = self.avgpool(image_features)

        #print(image_embd_layer2.shape)
        gps_embd_layer2 = self.vel_emb2(gps_features_layer1)


        image_features_layer2, gps_features_layer2 = self.transformer2(image_embd_layer2, gps_embd_layer2)

        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear')
        image_features = image_features + image_features_layer2
        image_features = self.image_encoder_layer4(image_features)

        #print(image_features.shape)
        # fusion at (B, 80, 16, 16)
        image_embd_layer3 = self.avgpool(image_features)
        gps_embd_layer3 = self.vel_emb3(gps_features_layer2)

        image_features_layer3, gps_features_layer3 = self.transformer3(image_embd_layer3, gps_embd_layer3)
        
        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear')
        image_features = image_features + image_features_layer3

        image_features = self.image_encoder_layer5(image_features)
        image_features = self.image_encoder_layer6(image_features)
        #print(image_features.shape)
        # fusion at (B, 192, 8, 8)
        image_embd_layer4 = self.avgpool(image_features)
        gps_embd_layer4 = self.vel_emb4(gps_features_layer3)

        image_features_layer4, gps_features_layer4 = self.transformer4(image_embd_layer4, gps_embd_layer4)
        image_features = image_features + image_features_layer4

        image_features = self.image_encoder_avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)

        gps_features = gps_features_layer4

        fused_features = torch.cat([image_features, gps_features], dim=1)
        # fused_features = torch.cat([image_features, lidar_features, radar_features], dim=1)

        fused_features = torch.sum(fused_features, dim=1)

        return fused_features


class SwinFuser1(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.encoder = Encoder(config).to(self.device)

        self.join = nn.Sequential(
                            nn.Linear(192, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 256),
                        ).to(self.device)
        # self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        # self.output = nn.Linear(64, 2).to(self.device)
        
    def forward(self, image_list, gps_list):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            gps (tensor): input gps
        '''
        fused_features = self.encoder(image_list, gps_list)
        z = self.join(fused_features)

       

        return z