import torch 
import numpy as np
# from model2_seq import TransFuser
from config_seq import GlobalConfig
from prettytable import PrettyTable
# from only_image_gps_transfuser import TransFuser2
# from two_images_gps_transfuser import TransFuser3
from model_efnet_gpt import TransFuser4
from model_efnet_swin import SwinFuser1
from torchvision import models

device = "cuda"
config = GlobalConfig()

add_velocity =  1
add_mask = 0
enhanced = 1
angle_norm = 1 
custom_FoV_lidar = 1 
filtered = 0
add_seg = 0

config.add_velocity = add_velocity
config.add_mask = add_mask
config.enhanced = enhanced
config.angle_norm = angle_norm
config.custom_FoV_lidar = custom_FoV_lidar
config.filtered = filtered
config.add_seg = add_seg


# model = TransFuser4(config,device)
model = SwinFuser1(config,device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

#for p in model.parameters():
#    print(p.numel())

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

count_parameters(model)

img_list = [torch.rand(2,3,config.crop,config.crop).to(device=device) for i in range(10)]
gps_list = [torch.rand(2,5,2).to(device=device),torch.rand(2,5,2).to(device=device)]

out = model(img_list, gps_list)
print(out.shape)

#model2 = models.efficientnet_b0(pretrained =True)


#block1 = models.efficientnet_b0(pretrained =True)
#print(block1)
