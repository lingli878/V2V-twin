# import os
# import json
# import pickle
from PIL import Image
import random
import pandas as pd

import numpy as np
import torch 
from torch.utils.data import Dataset
# from tqdm import tqdm
# import sys
# import matplotlib.pyplot as plt
# import open3d as o3d
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from scipy import stats
# from sklearn.preprocessing import normalize
import utm
import cv2
import re

class MATTIA_Data(Dataset):
    def __init__(self, root_csv, config, test=False, augment=False):

        self.dataframe = pd.read_csv(root_csv)
        self.seq_len = config.seq_len
        self.test = test
        # self.add_mask = config.add_mask
        # self.enhanced = config.enhanced
        self.augment = augment
        # self.add_seg = config.add_seg
        self.gps_features = config.gps_features
        self.n_gps_features_max = config.n_gps_features_max
        self.n_gps = config.n_gps
        self.crop = config.crop

    def __len__(self):
        """Returns the length of the dataset. """
        return self.dataframe.index.stop # or len(self.dataframe)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['front_images'] = []
        data['back_images'] = []
        data['scenario'] = []
        data['loss_weight'] = []
        data['gps'] = []

        add_front_images = []
        add_back_images = []
        instanceidx = ['1','2', '3','4','5']
        
        gps = np.zeros((self.seq_len,self.n_gps,2)) 
        for time_idx in range(self.seq_len):
            gps[time_idx,0,:] = np.loadtxt(self.dataframe['x'+str(time_idx+1)+'_unit1_gps1'][index])
            gps[time_idx,1,:] = np.loadtxt(self.dataframe['x'+str(time_idx+1)+'_unit2_gps1'][index])
        
        if self.gps_features:
            data['gps'] = extract_gps_features(gps=gps,seq_len=self.seq_len,n_features_max=self.n_gps_features_max)
        else:
            for i in range(self.n_gps):
                data['gps'].append(torch.from_numpy(gps[:,i,:]))
        
        for stri in instanceidx:
            add_front_images.append(self.dataframe['x'+stri+'_unit1_rgb5'][index])
            add_back_images.append(self.dataframe['x'+stri+'_unit1_rgb6'][index])

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
            front_imgs = np.array(Image.open(add_front_images[i]).resize((self.crop,self.crop)))
            back_imgs = np.array(Image.open(add_back_images[i]).resize((self.crop,self.crop)))
            
            # apply random transformation
            if self.augment:
                front_imgs = Image.open(add_front_images[i]).resize((self.crop,self.crop))
                back_imgs = Image.open(add_back_images[i]).resize((self.crop,self.crop))
                front_imgs = np.array(apply_random_transform(front_imgs))
                back_imgs = np.array(apply_random_transform(back_imgs))
                
            data['front_images'].append(torch.from_numpy(np.transpose(front_imgs, (2, 0, 1))))
            data['back_images'].append(torch.from_numpy(np.transpose(back_imgs, (2, 0, 1))))

        # training labels
        if not self.test:
            data['beam'] = []
            data['beamidx'] = []
            data['beam_pwr'] = []
            # gaussian distributed target instead of one-hot
            beamidx = self.dataframe['y1_unit1_overall-beam'][index] - 1 # -1 ENSURES beamidx IN 0 - 255
            _start = np.mod(beamidx - 5,256)
            _end = np.mod(beamidx + 5,256)
            x_data = list(range(_start, 256)) + list(range(0, _end)) if _end < _start else list(range(_start,_end))
            y_data = stats.norm.pdf(x_data, beamidx, 0.5)
            data_beam = np.zeros((256))
            data_beam[np.mod(x_data,256)] = y_data * 0.9858202 # after truncation this ensures unitary sum of the new distribution
            # if self.flip:
            #     beamidx = 256-beamidx
            #     data_beam = np.ascontiguousarray(np.flip(data_beam,0))
            data['beam'].append(data_beam)
            data['beamidx'].append(beamidx)
            
            # load beam power
            y_pwrs = np.zeros((4,64))
            for arr_idx in range(4): # 4 antenna arrays
                y_pwrs[arr_idx,:] = np.loadtxt(self.dataframe[f'y1_unit1_pwr{arr_idx+1}'][index])
            y_pwrs = y_pwrs.reshape((256)) # N_ARR*N_BEAMS
            data['beam_pwr'].append(torch.from_numpy(y_pwrs))
        return data

def apply_random_transform(image_sample):
    """
    iamge_sample shape: (H,W,C)
    Takes as input an image and returns the same image with one of seven
    transformations applied with equal probability.
    """
    transform_choice = random.randint(1, 7)

    if transform_choice == 1:
        brightness_factor = random.uniform(0.5, 3)
        img_aug = F.adjust_brightness(image_sample, brightness_factor)
    elif transform_choice == 2:
        contrast_factor = random.uniform(0.5, 4)
        img_aug = F.adjust_contrast(image_sample, contrast_factor)
    elif transform_choice == 3:
        gamma_factor = random.uniform(0.5, 3)
        img_aug = F.adjust_gamma(image_sample, gamma_factor)
    elif transform_choice == 4:
        hue_factor = random.uniform(-0.5, 0.5)
        img_aug = F.adjust_hue(image_sample, hue_factor)
    elif transform_choice == 5:
        saturation_factor = random.uniform(0, 4)
        img_aug = F.adjust_saturation(image_sample, saturation_factor)
    elif transform_choice == 6:
        sharpness_factor = random.uniform(0, 10)
        img_aug = F.adjust_sharpness(image_sample, sharpness_factor)
    else:
        kernel_size_factor = (9, 7)
        sigma_factor = (3, 5)
        img_aug = F.gaussian_blur(image_sample, kernel_size=kernel_size_factor, sigma=sigma_factor)

    return img_aug

def xy_from_latlong(lat_long):
    """
    lat_long shape: (time_samples,n_gps,2)
    Requires lat and long, in decimal degrees, in the 1st and 2nd columns.
    Returns same row vec/matrix on cartesian (XY) coords.
    """
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:,:,0], lat_long[:,:,1])
    return np.stack((x,y), axis=2)

def extract_gps_features(gps,seq_len,n_features_max,acc_threshold=8):
    """
    SOMETIMES CURVATURE IS nan, NOT INCLUDED AS FEATURE
    
    gps shape: (seq_len,n_gps,2)
    Returns features associated with gps. The model accepts n_features_max features as input. If less
    features are considered, the function uses zero padding to reach n_features_max.
    """
    vel_unit1_mean, vel_unit1_std = np.array([-41.6961134, -24.37544635]), np.array([28.41361448, 26.90043431])
    vel_unit2_mean, vel_unit2_std = np.array([34.29699547, -15.26099414]), np.array([27.73044074, 15.43753996])
    relative_vel_mean, relative_vel_std = np.array([-75.99310888, -9.11445221]), np.array([52.79357176, 30.84193528])

    acc_unit1_mean, acc_unit1_std = np.array([-0.00614051, -0.00014148]), np.array([1.13771294, 1.70639424])
    acc_unit2_mean, acc_unit2_std = np.array([0.00486797, -0.00011349]), np.array([1.55493432, 1.8687202])
    relative_acc_mean, relative_acc_std = np.array([-1.10084791e-02, -2.79874864e-05]), np.array([1.91844784, 2.53037317])

    jerk_unit1_mean, jerk_unit1_std = np.array([3.17746803e-05, -9.16195392e-05]), np.array([16.08515896, 25.97480694])
    jerk_unit2_mean, jerk_unit2_std = np.array([-2.39528025e-05, 2.79293595e-04]), np.array([25.61524045, 28.80074501])
    relative_jerk_mean, relative_jerk_std = np.array([5.57274828e-05, -3.70913135e-04]), np.array([30.30148092, 38.88495818])

    # curvature_unit1_mean, curvature_unit1_std = np.nan, np.nan
    # curvature_unit2_mean, curvature_unit2_std = np.nan, np.nan

    angular_velocity_mean, angular_velocity_std = np.array([-0.00012271, 0.00029294]), np.array([7.47440836, 10.3856978])
    relative_angular_vel_mean, relative_angular_vel_std = -0.0004156572133953136, 12.800462401476324
    relative_distance_mean, relative_distance_std = 38.58902930566432, 43.60465162129897
    relative_coords_mean, relative_coords_std = np.array([0.36117908, -0.60185364]), np.array([39.50417896, 42.77155588])

    ## feature engineering
    dt = 0.2  # time spacing between two adjacent GPS samples
    gps_xy = xy_from_latlong(gps)
    diff_gps = np.diff(gps_xy,axis=0)

    vel_xy = diff_gps / dt * 3.6 # (4,2,2) xy coordinates of velocity for unit1 and unit2
    vel_unit1 = vel_xy[:,0]
    vel_unit2 = vel_xy[:,1]

    # correct velocity and acceleration outliers (try index = 930)        
    for i in range(len(vel_unit1)-1):
        acc_unit1 = np.diff(vel_unit1,axis=0) / dt
        acc_unit2 = np.diff(vel_unit2,axis=0) / dt
        for j in range(2):  # Loop through x and y dimensions
            if np.abs(acc_unit1[i, j]) > acc_threshold:
                acc_unit1[i, j] = 0.
                vel_unit1[i + 1, j] = vel_unit1[i, j]
            if np.abs(acc_unit2[i, j]) > acc_threshold:
                acc_unit2[i, j] = 0.
                vel_unit2[i + 1, j] = vel_unit2[i, j]

    # velocity_unit1 = np.linalg.norm(vel_unit1, axis=1) # absolute velocity in km/h
    # velocity_unit2 = np.linalg.norm(vel_unit2, axis=1) # absolute velocity in km/h

    acc_unit1 = np.diff(vel_unit1,axis=0) / dt # features (3,2)
    acc_unit2 = np.diff(vel_unit2,axis=0) / dt # features (3,2)

    relative_vel = vel_unit1 - vel_unit2
    relative_acc = acc_unit1 - acc_unit2 # features (3,2)

    # # jerk features
    jerk_unit1 = np.diff(acc_unit1,axis=0) / dt # features (2,2)
    jerk_unit2 = np.diff(acc_unit2,axis=0) / dt # features (2,2)
    relative_jerk = jerk_unit1 - jerk_unit2 # features (2,2)

    # curvature
    # acc_xy = np.diff(vel_xy, axis=0) / dt
    # curvature = np.abs( vel_xy[:-1,0]*acc_xy[:,1] - vel_xy[:-1,1]*acc_xy[:,0]  ) / (vel_xy[:-1,0]**2 + vel_xy[:-1,1]**2)**1.5
    # curvature_unit1 = np.abs( vel_unit1[:-1,0]*acc_unit1[:,1] - vel_unit1[:-1,1]*acc_unit1[:,0]  ) / (vel_unit1[:-1,0]**2 + vel_unit1[:-1,1]**2)**1.5
    # curvature_unit2 = np.abs( vel_unit2[:-1,0]*acc_unit2[:,1] - vel_unit2[:-1,1]*acc_unit2[:,0]  ) / (vel_unit2[:-1,0]**2 + vel_unit2[:-1,1]**2)**1.5
    # rate_curvature_unit1 = np.diff(curvature_unit1,axis=0)
    # rate_curvature_unit2 = np.diff(curvature_unit2,axis=0)

    orientation = np.arctan2(diff_gps[:, :, 1], diff_gps[:, :, 0]) # (4,2)
    relative_orientation = (orientation[:,0] - orientation[:,1])/(2*np.pi) # (4,1)
    angular_velocity = np.diff(orientation,axis=0) / dt # (4,1)
    relative_angular_velocity = angular_velocity[:,0] - angular_velocity[:,1]

    # GPS normalization (preserve relative positions, and values constraint between (-1,1))
    lat, lon = gps[:,:,0], gps[:,:,1]
    x = np.cos(lat) * np.cos(lon) # features (5,2)
    y = np.cos(lat) * np.sin(lon) # features (5,2)
    z = np.sin(lat)
    rel_coords = np.diff(gps_xy,axis=1).reshape((seq_len,2)) # features (5,2)
    rel_dist = np.linalg.norm(rel_coords,axis=1) # features (5,1)
    unit1_gps_normalized = np.vstack((x[:,0],y[:,0])).T # (5,2)
    unit2_gps_normalized = np.vstack((x[:,1],y[:,1])).T # (5,2)

    # =================== feature normalization
    vel_unit1_normalized = (vel_unit1 - vel_unit1_mean) / vel_unit1_std
    vel_unit2_normalized = (vel_unit2 - vel_unit2_mean) / vel_unit2_std
    relative_vel_normalized = (relative_vel - relative_vel_mean) / relative_vel_std

    acc_unit1_normalized = (acc_unit1 - acc_unit1_mean) / acc_unit1_std
    acc_unit2_normalized = (acc_unit2 - acc_unit2_mean) / acc_unit2_std
    relative_acc_normalized = (relative_acc - relative_acc_mean) / relative_acc_std

    jerk_unit1_normalized = (jerk_unit1 - jerk_unit1_mean) / jerk_unit1_std
    jerk_unit2_normalized = (jerk_unit2 - jerk_unit2_mean) / jerk_unit2_std
    relative_jerk_normalized = (relative_jerk - relative_jerk_mean) / relative_jerk_std

    # curvature_normalized = ...

    angular_velocity_normalized = (angular_velocity - angular_velocity_mean) / angular_velocity_std
    relative_angular_vel_normalized = (relative_angular_velocity - relative_angular_vel_mean) / relative_angular_vel_std

    relative_distance_normalized = (rel_dist - relative_distance_mean) / relative_distance_std
    relative_coords_normalized = (rel_coords - relative_coords_mean) / relative_coords_std

    gps_list = []

    ## UNIT1 GPS COORDS
    unit1_gps_normalized = np.vstack((x[:,0],y[:,0])).T # (5,2)
    gps_list.append(torch.from_numpy(unit1_gps_normalized))

    ## UNIT2 GPS COORDS
    unit2_gps_normalized = np.vstack((x[:,1],y[:,1])).T # (5,2)
    gps_list.append(torch.from_numpy(unit2_gps_normalized))

    ## GPS FEATURES                                                                                               zero padding
    rel_dist_pad = np.pad(relative_distance_normalized.reshape((5,1)),((0,0),(0,1)),mode='constant')        # shape (5,1) -> (5,2)
    vel_unit1_pad = np.pad(vel_unit1_normalized.reshape((4,2)),((0,1),(0,0)),mode='constant')               # shape (4,1) -> (5,2)
    vel_unit2_pad = np.pad(vel_unit2_normalized.reshape((4,2)),((0,1),(0,0)),mode='constant')               # shape (4,1) -> (5,2)
    relative_vel_pad = np.pad(relative_vel_normalized,((0,1),(0,0)),mode='constant')                        # shape (4,2) -> (5,2)
    acc_unit1_pad = np.pad(acc_unit1_normalized,((0,2),(0,0)),mode='constant')                              # shape (3,2) -> (5,2)
    acc_unit2_pad = np.pad(acc_unit2_normalized,((0,2),(0,0)),mode='constant')                              # shape (3,2) -> (5,2)
    relative_acc_pad = np.pad(relative_acc_normalized,((0,2),(0,0)),mode='constant')                        # shape (3,2) -> (5,2)
    jerk_unit1_pad = np.pad(jerk_unit1_normalized,((0,3),(0,0)),mode='constant')                            # shape (2,2) -> (5,2)
    jerk_unit2_pad = np.pad(jerk_unit2_normalized,((0,3),(0,0)),mode='constant')                            # shape (2,2) -> (5,2)
    relative_jerk_pad = np.pad(relative_jerk_normalized,((0,3),(0,0)),mode='constant')                      # shape (2,2) -> (5,2)
    # curvature_pad = np.pad(curvature,((0,2),(0,0)),mode='constant')                                       # shape (3,2) -> (5,2)
    # diff_curvature_pad = np.pad(diff_curvature,((0,3),(0,0)),mode='constant')                             # shape (3,2) -> (5,2)
    relative_orient_pad = np.pad(relative_orientation.reshape((seq_len-1,1)),((0,1),(0,1)),mode='constant') # shape (4,1) -> (5,2)
    angular_vel_pad = np.pad(angular_velocity_normalized,((0,2),(0,0)),mode='constant')                     # shape (4,2) -> (5,2)
    relative_angular_vel_pad = np.pad(relative_angular_vel_normalized.reshape((3,1)),((0,2),(0,1)),mode='constant') # shape (3,1) -> (5,2)

    gps_list.append(torch.from_numpy(relative_coords_normalized))
    gps_list.append(torch.from_numpy(rel_dist_pad))
    gps_list.append(torch.from_numpy(vel_unit1_pad))
    gps_list.append(torch.from_numpy(vel_unit2_pad))
    gps_list.append(torch.from_numpy(relative_vel_pad))
    gps_list.append(torch.from_numpy(acc_unit1_pad))
    gps_list.append(torch.from_numpy(acc_unit2_pad))
    gps_list.append(torch.from_numpy(relative_acc_pad))
    gps_list.append(torch.from_numpy(jerk_unit1_pad))
    gps_list.append(torch.from_numpy(jerk_unit2_pad))
    gps_list.append(torch.from_numpy(relative_jerk_pad))
    # gps_list.append(torch.from_numpy(curvature_pad))
    # gps_list.append(torch.from_numpy(diff_curvature_pad))
    gps_list.append(torch.from_numpy(relative_orient_pad))
    gps_list.append(torch.from_numpy(angular_vel_pad))
    gps_list.append(torch.from_numpy(relative_angular_vel_pad))

    if len(gps_list)*seq_len < n_features_max:
        [gps_list.append(torch.zeros((seq_len,2))) for _ in range(n_features_max//seq_len - len(gps_list))]
        
    # gps_stacked = np.stack(gps_list,axis=0).reshape((1,-1,2))
    # print(f'GPS feature vector shape: {gps_stacked.shape}')
    return gps_list