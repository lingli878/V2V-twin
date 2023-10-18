import os
import json
import pickle
from PIL import Image
import pandas as pd

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
# import open3d as o3d
import torchvision.transforms as transforms
from scipy import stats
from sklearn.preprocessing import normalize
import utm
import cv2
import re

class MATTIA_Data(Dataset):
    def __init__(self, root_csv, config, test=False, augment={'camera':0, 'gps':0}): # note: REMOVED LIDAR/RADAR, ADDED GPS (by Mattia)

        self.dataframe = pd.read_csv(root_csv)
        self.seq_len = config.seq_len
        # self.gps_data = [] # NO NEED...
        # self.pos_input_normalized = Normalize_loc(self.dataframe,self.seq_len)
        self.test = test
        # self.add_velocity = config.add_velocity # COULD BE HELPFUL FOR GPS FEATURES ENGINEERING
        # self.add_mask = config.add_mask
        # self.enhanced = config.enhanced
        self.augment = augment
        # self.flip = flip
        # self.add_seg = config.add_seg
        self.n_gps = config.n_gps

    def __len__(self):
        """Returns the length of the dataset. """
        return self.dataframe.index.stop # or len(self.dataframe)
        # return np.sum(self.dataframe['scenario'] == self.num_scenario)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        # data['gps'] = self.pos_input_normalized[index]
        data['front_images'] = []
        data['back_images'] = []
        # if self.flip:
        #     data['gps'][:,1] = -data['gps'][:,1]
        data['scenario'] = []
        data['loss_weight'] = [] 

        gps = np.zeros((self.seq_len,self.n_gps,2)) 
        for time_idx in range(self.seq_len):
            gps[time_idx,0,:] = np.loadtxt('./Dataset/' + self.dataframe['x'+str(time_idx+1)+'_unit1_gps1'][index])
            gps[time_idx,1,:] = np.loadtxt('./Dataset/' + self.dataframe['x'+str(time_idx+1)+'_unit2_gps1'][index])
        data['gps'] = gps
        add_front_images = []
        add_back_images = []
        add_gps = []
        instanceidx=['1','2', '3', '4', '5']#5 time instances
        
        ## data augmentation
        for stri in instanceidx:
            # camera data
            camera_dir_front ='./Dataset/' +  self.dataframe['x'+stri+'_unit1_rgb5'][index]
            camera_dir_back ='./Dataset/' +  self.dataframe['x'+stri+'_unit1_rgb6'][index]
            if self.augment['camera'] > 0: # and 'scenario31' in camera_dir:
                camera_dir = re.sub('camera_data/', 'camera_data_aug/', camera_dir)
                camera_dir = camera_dir[:-4] + '_' + str(self.augment['camera']) + '.jpg'
                add_front_images.append(camera_dir)
            else:
                add_front_images.append('./Dataset/' + self.dataframe['x'+stri+'_unit1_rgb5'][index])
                add_back_images.append('./Dataset/' +self.dataframe['x'+stri+'_unit1_rgb6'][index])
                add_gps.append(self.dataframe['x'+stri+'_unit1_gps1'][index])

        self.seq_len = len(instanceidx)

        # check which scenario is the data sample associated 
        scenarios = ['scenario36', 'scenario37', 'scenario38', 'scenario39']
        loss_weights = [1.0, 1.0, 1.0, 1.0]

        for i in range(len(scenarios)): 
            s = scenarios[i]
            if s in self.dataframe['x1_unit1_rgb5'][index]:
                data['scenario'] = s
                data['loss_weight'] = loss_weights[i]
                break

        for i in range(self.seq_len):
            if self.augment['camera'] == 0:
                # if 'scenario'+str(self.num_scenario) in add_front_images[i]:
                # if 'scenario36' in add_front_images[i] or 'scenario37' in add_front_images[i]:
                # if 'scenario31' in add_front_images[i] or 'scenario32' in add_front_images[i]:
                    # if self.augment['camera'] == 0:  # segmentation added to non augmented data
                    #     if self.add_mask:
                    #         imgs = np.array(
                    #             Image.open(add_front_images[i][:30] + '_mask' + add_front_images[i][30:]).resize(
                    #                 (256, 256)))
                    #     else:
                    #         front_imgs = np.array(Image.open(add_front_images[i]).resize((256, 256)))
                    #         back_imgs = np.array(Image.open(add_back_images[i]).resize((256,256)))
                    #         if self.add_seg:
                    #             seg = np.array(
                    #                 Image.open(add_front_images[i][:30] + '_seg' + add_front_images[i][30:]).resize(
                    #                     (256, 256)))
                    #             a = seg[..., 2]
                    #             a = a[:, :, np.newaxis]
                    #             a = np.concatenate([a, a, a], axis=2)
                    #             seg_car = cv2.bitwise_and(front_imgs, a)
                    #             front_imgs = cv2.addWeighted(front_imgs, 0.8, seg_car, 0.5, 0)
                # else:
                #     continue
                #     if self.add_mask & self.enhanced:
                #         raise Exception("mask or enhance, both are not possible")
                #     if self.add_mask:
                #         imgs = np.array(
                #             Image.open(add_front_images[i][:30] + '_mask' + add_front_images[i][30:]).resize((256, 256)))
                #     elif self.enhanced:
                #         imgs = np.array(
                #             Image.open(add_front_images[i]).resize((256, 256)))
                #     else:
                #         imgs = np.array(Image.open(add_front_images[i][:30]+'_raw'+add_front_images[i][30:]).resize((256, 256)))
            # else:
                front_imgs = np.array(Image.open(add_front_images[i]).resize((256,256)))
                back_imgs = np.array(Image.open(add_back_images[i]).resize((256,256)))
                
            # flip data augmentation
            # if self.flip:
            #     imgs = np.ascontiguousarray(np.flip(imgs,1))
            data['front_images'].append(torch.from_numpy(np.transpose(front_imgs, (2, 0, 1))))
            data['back_images'].append(torch.from_numpy(np.transpose(back_imgs, (2, 0, 1))))

        # TRUNCATED GAUSSIAN DISTRIBUTION FOR BEAM LABELS INSTEAD OF ONE-HOT
        if not self.test:
            data['beam'] = []
            data['beamidx'] = []
            data['beam_pwr'] = []
            #Gaussian distributed target instead of one-hot
            beamidx = self.dataframe['y1_unit1_overall-beam'][index] - 1 # -1 ENSURES beamidx IN 0 - 255
            _start = np.mod(beamidx - 5,256)
            _end = np.mod(beamidx + 5,256)
            x_data = list(range(_start, 256)) + list(range(0, _end)) if _end < _start else list(range(_start,_end))
            y_data = stats.norm.pdf(x_data, beamidx, 0.5)
            data_beam = np.zeros((256))
            data_beam[np.mod(x_data,256)] = y_data * 0.9858202 # ENSURES sum() = 1 OF THE TRUNCATED GAUSSIAN PDF
            # if self.flip:
            #     beamidx = 256-beamidx # BEAM IDX IS BETWEEN 1 - 256 RIGHT??
            #     data_beam = np.ascontiguousarray(np.flip(data_beam,0))
            data['beam'].append(data_beam)
            data['beamidx'].append(beamidx)
            
            # load beam power
            y_pwrs = np.zeros((4,64))
            for arr_idx in range(4): # 4 antenna arrays
                y_pwrs[arr_idx,:] = np.loadtxt('./Dataset/' + self.dataframe[f'y1_unit1_pwr{arr_idx+1}'][index])
            y_pwrs = y_pwrs.reshape((256)) # N_ARR*N_BEAMS
            data['beam_pwr'].append(torch.from_numpy(y_pwrs))
        return data

# def xy_from_latlong(lat_long):
#     """
#     lat_long shape: (n_points,2)
#     Requires lat and long, in decimal degrees, in the 1st and 2nd columns.
#     Returns same row vec/matrix on cartesian (XY) coords.
#     """
#     # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
#     x, y, *_ = utm.from_latlon(lat_long[:,0], lat_long[:,1])
#     return np.stack((x,y), axis=1)

# # IN V2V, BOTH VEHICLES ARE MOVING, WE NEED TO FIND A RELATIVE REFERENCE TO NORMALIZE LOCATION, HOW???
# def Normalize_loc(dataframe,seq_len):
#     n_samples = dataframe.index.stop
#     gps = np.zeros((n_samples,5,2,2))
#     for sample in range(3):
#         for time_idx in range(seq_len):
#             gps[sample,time_idx,0,:] = np.loadtxt(dataframe['x'+str(time_idx+1)+'_unit1_gps1'][sample])
#             gps[sample,time_idx,1,:] = np.loadtxt(dataframe['x'+str(time_idx+1)+'_unit2_gps1'][sample])
#     return gps
