import argparse
import json
import os, sys
import csv

from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
torch.backends.cudnn.benchmark = True
from scheduler import CyclicCosineDecayLR

from config_seq import GlobalConfig
# from model_efnet_gpt import TransFuser4
from model_efnet_swin import SwinFuser1

import torch.multiprocessing as mp
from datetime import timedelta
import torchvision

kw='best_'# keyword for the pretrained model in finetune
# data_root = './MultiModeBeamforming/'#path to the dataset

torch.cuda.empty_cache()
n_gpus = torch.cuda.device_count()
num_workers = torch.get_num_threads()

def setup_process(rank, world_size, backend='gloo'): # nccl or gloo
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()

setup_process(rank=n_gpus,world_size=n_gpus)

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='train_SWIN_GPS_FEATURES', help='Unique experiment identifier')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=70, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=26, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to')
parser.add_argument('--gps_features', type = int, default=1, help='use more normalized GPS features')
parser.add_argument('--add_mask', type=int, default=0, help='add mask to the camera data')
parser.add_argument('--enhanced', type=int, default=0, help='use enhanced camera data')
parser.add_argument('--loss', type=str, default='focal', help='crossentropy or focal loss')
parser.add_argument('--scheduler', type=int, default=1, help='use scheduler to control the learning rate')
parser.add_argument('--load_previous_best', type=int, default=0, help='load previous best pretrained model ')
parser.add_argument('--temp_coef', type=int, default=1, help='apply temperature coefficience on the target')
parser.add_argument('--train_adapt_together', type=int, default=1, help='combine train and adaptation dataset together')
parser.add_argument('--finetune', type=int, default=0, help='first train on development set and finetune on 31-34 set')
parser.add_argument('--Test', type=int, default=0, help='Test')
parser.add_argument('--augmentation', type=int, default=1, help='data augmentation of camera')
parser.add_argument('--add_seg', type=int, default=0, help='add segmentation on 31&32 images')
parser.add_argument('--ema', type=int, default=1, help='exponential moving average')
parser.add_argument("--local_rank", default=0, type=int)
# parser.add_argument('--flip', type=int, default=0, help='flip all the data to augmentation')
args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()

writer = SummaryWriter(log_dir=args.logdir)

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""
	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.APL = [-100] # CHANGE TO APL
		self.bestval = -100
		# if args.finetune:
		self.APLft = [-100] # APLft
		if args.loss == 'ce':#crossentropy loss
			self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
		elif args.loss == 'focal':#focal loss
			self.criterion = FocalLoss()

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()
		running_acc = 0.0
		gt_beam_all = []
		pred_beam_all = []
		gt_pwr_all = []
		pred_pwr_all = []
		# Train loop
		pbar=tqdm(dataloader_train, desc='training') # HERE PUT THE DATALOADER BATCH
		for data in pbar:
			# efficiently zero gradients
			optimizer.zero_grad(set_to_none=True)
			# create batch and move to GPU
			image_front_list = []
			image_back_list = []
			gps_list = []
			for i in range(config.seq_len):
				image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32)) # IF .float() IS NOT INSERTED IT GIVES AN ERROR DUE TO uint8 IMAGE CASTING
				image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
			# [gps_list.append(data['gps'][i].to(args.device, dtype=torch.float32) for i in range(config.n_gps_features_max))]
			for i in range(config.n_gps_features_max//config.seq_len):
				gps_list.append(data['gps'][i].to(args.device, dtype=torch.float32))

			pred_beams = model(image_front_list + image_back_list, gps_list)
			pred_beam = torch.argmax(pred_beams, dim=1)
			gt_beamidx = data['beamidx'][0].to(args.device, dtype=torch.long)
			gt_beams = data['beam'][0].to(args.device, dtype=torch.float32)
			# running_acc = (pred_beam == gt_beamidx).sum().item()
			if args.temp_coef:#temperature coefficiece
				loss = self.criterion(pred_beams, gt_beams)
			else:
				loss = self.criterion(pred_beams, gt_beamidx)
			
			gt_beam_all.append(data['beamidx'][0])
			pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
			true_pwr_batch = data['beam_pwr'][0].to(args.device, dtype=torch.float32)
			gt_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx]).cpu().numpy())
			pred_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy()) # HOW TO FIND BATCH SIZE? args.batch_size or pred_beam.shape[0]?
			loss.backward()
			loss_epoch += float(loss.item())
			pbar.set_description(str(loss.item()))
			num_batches += 1
			optimizer.step()
			#cleanup()
   
			# #============== DEBUG =========================
			# if self.cur_epoch > 3:
			# 	estimated_power = (true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy()
			# 	true_power = (true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx]).cpu().numpy()
			# 	APL_score = APL(true_power,estimated_power)
			# 	true_beam = data['beamidx'][0].cpu().numpy()
			# 	# batch_acc = compute_acc(pred_beam,gt_beamidx,top_k=[1])
			# 	batch_acc = running_acc / float(pred_beam.shape[0])
			# 	print(f'\nAPL score = {APL_score:.4f} dB | acc: {batch_acc} | pred vs true beam [{pred_beam.cpu().numpy(), true_beam}]')
			# 	print(f'power: pred = {estimated_power}, true = {true_power}')
			# break
			#==============================================
			if args.ema:# Exponential Moving Averages
				ema.update()	# during training, after update parameters, update shadow weights

			self.cur_iter += 1
		pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
		gt_beam_all = np.squeeze(np.concatenate(gt_beam_all, 0))
		pred_pwr_all = np.squeeze(np.concatenate(pred_pwr_all, 0))
		gt_pwr_all = np.squeeze(np.concatenate(gt_pwr_all, 0))
		curr_acc = compute_acc(pred_beam_all, gt_beam_all, top_k=[1, 3, 5])
		# print(f'APL score PREDICTED pwr {pred_pwr_all}')
		# print(f'APL score TRUE pwr {gt_pwr_all}')
		APL_score = APL(gt_pwr_all, pred_pwr_all) # CALCULATE APL SCORE
		print('Train top beam acc: ',curr_acc, ' APL score: ',APL_score)
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		# self.cur_epoch += 1
		writer.add_scalar('APL_score_train', APL_score, self.cur_epoch)
		for i in range(len(curr_acc)):
			writer.add_scalars('curr_acc_train', {'beam' + str(i):curr_acc[i]}, self.cur_epoch)
		writer.add_scalar('curr_loss_train', loss_epoch, self.cur_epoch)
		# if args.finetune:
		if APL_score > self.APLft[-1]:
			self.APLft.append(APL_score)
			print(APL_score, self.APLft[-2], 'save new model')
			torch.save(model.state_dict(), os.path.join(args.logdir, kw + 'model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, kw + 'optim.pth'))
		else:
			print('best',self.APLft[-1])

	def validate(self):
		if args.ema:#Exponential Moving Averages
			ema.apply_shadow()    # before eval\uff0capply shadow weights
		model.eval()
		running_acc = 0.0
		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.
			gt_beam_all=[]
			pred_beam_all=[]
			scenario_all = []
			gt_pwr_all = [] # added co compute APL score
			pred_pwr_all = []
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0):
				# create batch and move to GPU
				gps = data['gps'].to(args.device, dtype=torch.float32)
				image_front_list = []
				image_back_list = []
				for i in range(config.seq_len):
					image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32)) # IF .float() IS NOT INSERTED IT GIVES AN ERROR DUE TO uint8 IMAGE CASTING
					image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
				
				pred_beams = model(image_front_list + image_back_list, [gps[:,:,0], gps[:,:,1]])
				pred_beam = torch.argmax(pred_beams, dim=1)
				gt_beam_all.append(data['beamidx'][0])
				gt_beams = data['beam'][0].to(args.device, dtype=torch.float32)
				gt_beamidx = data['beamidx'][0].to(args.device, dtype=torch.long)
				pred_beam_all.append(torch.argsort(pred_beams, dim=1, descending=True).cpu().numpy())
				running_acc += (pred_beam == gt_beamidx).sum().item()
				if args.temp_coef:
					loss = self.criterion(pred_beams, gt_beams)
				else:
					loss = self.criterion(pred_beams, gt_beamidx)
				wp_epoch += float(loss.item())
				num_batches += 1
				true_pwr_batch = data['beam_pwr'][0].to(args.device, dtype=torch.float32)
				gt_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx]).cpu().numpy())
				pred_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy())
				scenario_all.append(data['scenario'])
				#============== DEBUG ========================= 
				# estimated_power = (true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy()
				# true_power = (true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx]).cpu().numpy()
				# APL_score = APL(estimated_power,true_power)
				# true_beam = data['beamidx'][0].cpu().numpy()
				# print(f'\nAPL score = {APL_score:.4f} dB | acc: {running_acc} | pred vs true beam [{pred_beam.cpu().numpy(), true_beam}]')
				# print(f'power: pred = {estimated_power}, true = {true_power}')
				# if batch_num == 2:
				# break
				#============================================== 
    
			pred_beam_all=np.squeeze(np.concatenate(pred_beam_all,0))
			gt_beam_all=np.squeeze(np.concatenate(gt_beam_all,0))
			scenario_all = np.squeeze(np.concatenate(scenario_all,0))
			pred_pwr_all = np.squeeze(np.concatenate(pred_pwr_all, 0)) # (n_samples,1)
			gt_pwr_all = np.squeeze(np.concatenate(gt_pwr_all, 0)) # (n_samples,1)
			#calculate accuracy and APL score according to different scenarios
			# scenarios = ['scenario36', 'scenario37', 'scenario38', 'scenario39']
			scenarios = ['scenario36']
			for s in scenarios:
				beam_scenario_index = np.array(scenario_all) == s
				pred_pwr_s = pred_pwr_all[beam_scenario_index]
				gt_pwr_s = gt_pwr_all[beam_scenario_index]
				if np.sum(beam_scenario_index) > 0:
					curr_acc_s = compute_acc(pred_beam_all[beam_scenario_index], gt_beam_all[beam_scenario_index], top_k=[1,3,5])
					APL_score_s = APL(gt_pwr_s,pred_pwr_s)
					
					print(s, ' curr_acc: ', curr_acc_s, ' APL_score: ', APL_score_s)
					for i in range(len(curr_acc_s)):
						writer.add_scalars('curr_acc_val', {s + 'beam' + str(i):curr_acc_s[i]}, self.cur_epoch)
					writer.add_scalars('APL_score_val', {s:APL_score_s}, self.cur_epoch)

			curr_acc = compute_acc(pred_beam_all, gt_beam_all, top_k=[1,3,5])
   
			#============== DEBUG ========================= 
			# print(f'APL score PREDICTED pwr {pred_pwr_all}')
			# print(f'APL score TRUE pwr {gt_pwr_all}')
			# ==============================================
			APL_score_val = APL(gt_pwr_all, pred_pwr_all)
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:d}, Batch {batch_num:d}:' + f' Wp: {wp_loss:3.3f}')
			print('Val top beam acc: ',curr_acc, 'APL score: ', APL_score_val)
			writer.add_scalars('APL_score_val', {'scenario_all':APL_score_val}, self.cur_epoch)
			writer.add_scalar('curr_loss_val', wp_loss, self.cur_epoch)

			self.val_loss.append(wp_loss)
			self.APL.append(float(APL_score_val))
			self.cur_epoch += 1

		if args.ema:#Exponential Moving Averages
			ema.restore()	# after eval, restore model parameter


	def test(self):
		model.eval()
		with torch.no_grad():
			pred_beam_all=[]
			# pred_beam_confidence = []
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0): # CHANGE TO dataloader_test
				# create batch and move to GPU
				gps = data['gps'].to(args.device, dtype=torch.float32)
				image_front_list = []
				image_back_list = []
				for i in range(config.seq_len):
					image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32))
					image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
				
				pred_beams = model(image_front_list + image_back_list, [gps[:,:,0], gps[:,:,1]])
				pred_beam = torch.argmax(pred_beams, dim=1)
				pred_beam_all.append(pred_beam.cpu().numpy())
				# sm=torch.nn.Softmax(dim=1)
				# beam_confidence=torch.max(sm(pred_beams), dim=1)
				# pred_beam_confidence.append(beam_confidence[0].cpu().numpy())

			pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
			pred_beam_confidence = np.squeeze(np.concatenate(pred_beam_confidence, 0))
			# df_out = pd.DataFrame()
			# df_out['prediction'] = best_beam_pred
			# df_out.to_csv('beamwise_prediction.csv', index=False)
			save_pred_to_csv(pred_beam_all, top_k=[1, 2, 3], target_csv='beam_pred.csv')
			df = pd.DataFrame(data=pred_beam_confidence)
			df.to_csv('beam_pred_confidence_seq.csv')

	def save(self):
		save_best = False
		#print('best', self.bestval, self.bestval_epoch)
		#print(f'best APL = {self.bestval:.2f} dB @ epoch = {self.bestval_epoch}')

		if self.APL[-1] >= self.bestval:
			self.bestval = self.APL[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True
		print(f'best APL = {self.bestval:.2f} dB @ epoch = {self.bestval_epoch}')

		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
			'APL': self.APL,
		}

		# Save ckpt for every epoch
		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'final_model.pth'))
		# # Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))


		if save_best:# save the bestpretrained model
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model_val.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim_val.pth'))
			tqdm.write('====== Overwrote best model ======>')
		elif args.load_previous_best:
			model.load_state_dict(torch.load(os.path.join(args.logdir, 'best_model.pth')))
			optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'best_optim.pth')))
			tqdm.write('====== Load the previous best model ======>')

class FocalLoss(nn.Module):
	def __init__(self, gamma=2, alpha=0.25):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
	def __call__(self, input, target):
		if len(target.shape) == 1:
			target = torch.nn.functional.one_hot(target, num_classes=64)
		loss = torchvision.ops.sigmoid_focal_loss(input, target.float(), alpha=self.alpha, gamma=self.gamma,
												  reduction='mean')
		return loss

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



def save_pred_to_csv(y_pred, top_k=[1, 2, 3], target_csv='beam_pred.csv'):
	"""
    Saves the predicted beam results to a csv file.
    Expects y_pred: n_samples x N_BEAMS, and saves the top_k columns only.
    """
	cols = [f'top-{i} beam' for i in top_k]
	df = pd.DataFrame(data=y_pred[:, np.array(top_k) - 1]+1, columns=cols)
	df.index.name = 'index'
	df.to_csv(target_csv)
def compute_acc(y_pred, y_true, top_k=[1,2,3]):
    """ Computes top-k accuracy given prediction and ground truth labels."""
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)
    n_test_samples = len(y_true)
    if len(y_pred) != n_test_samples:
        raise Exception('Number of predicted beams does not match number of labels.')
    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(y_true)):
        for k_idx in range(n_top_k):
            hit = np.any(y_pred[samp_idx,:top_k[k_idx]] == y_true[samp_idx])
            total_hits[k_idx] += 1 if hit else 0
    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(y_true)*100, 4)


def APL(true_best_pwr, est_best_pwr):
    """
    Average Power Loss: average of the power wasted by using the predicted beam
    instead of the ground truth optimum beam.
    """
    
    return np.mean(10 * np.log10(est_best_pwr / true_best_pwr))


# Config
config = GlobalConfig()
config.gps_features = args.gps_features
config.add_mask = args.add_mask
config.enhanced = args.enhanced
config.add_seg = args.add_seg
data_root = config.data_root
n_gps_features_max = config.n_gps*config.seq_len
if config.gps_features:
    config.n_gps_features_max = 160 # maximum number of total gps features

import random
import numpy as np
seed = 100
random.seed(seed)
np.random.seed(seed) # numpy
torch.manual_seed(seed) # torch+CPU
# torch.cuda.manual_seed(seed) # torch+GPU
torch.use_deterministic_algorithms(False)
g = torch.Generator()
g.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============ DATASET
from data2_seq import MATTIA_Data

root_csv_train = './deepsense_challenge2023_trainset.csv'
dataset_scen = MATTIA_Data(root_csv=root_csv_train,config=config, test=False,augment=args.augmentation)

train_val_split = 0.9
train_dim = int(train_val_split * len(dataset_scen))
train_set, val_set = torch.utils.data.random_split(dataset_scen,[train_dim,len(dataset_scen) - train_dim])


train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
dataloader_train = DataLoader(train_set,batch_size=args.batch_size,shuffle=False, num_workers=8, pin_memory=True,
								  worker_init_fn=seed_worker, generator=g)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
dataloader_val = DataLoader(val_set,batch_size=args.batch_size,shuffle=True, num_workers=8, pin_memory=False,
								  worker_init_fn=seed_worker, generator=g)

# Model
# model = TransFuser4(config, args.device)
model = SwinFuser1(config,args.device)
model = DDP(model)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(tolerance=5, min_delta=0.0003)
# optimizer = nn.parallel.DistributedOptimizer(optimizer)
if args.scheduler:#Cyclic Cosine Decay Learning Rate
	scheduler = CyclicCosineDecayLR(optimizer,
	                                init_decay_epochs=15, # 15
	                                min_decay_lr=2.5e-6, # 2.5e-6
	                                restart_interval = 10, # 10
	                                restart_lr= 12.5e-5, # 12.5e-5
	                                warmup_epochs=10, # 10
	                                warmup_start_lr=2.5e-6) # 2.5e-6
 
trainer = Engine()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('======Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('======Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('======Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']
	trainer.APL = log_table['APL']


	# # FOR TESTING ONLY

	# Load checkpoint
	if args.finetune:# finetune the pretrained model

		if os.path.exists(os.path.join(args.logdir, 'all_finetune_on_'+ kw + 'model.pth')):
			print('======loading last'+'all_finetune_on_'+ kw + 'model.pth')
			model.load_state_dict(torch.load(os.path.join(args.logdir, 'all_finetune_on_'+ kw + 'model.pth')))
			optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'all_finetune_on_' + kw + 'optim.pth')))
		else:
			print('======loading '+kw+' model')
			model.load_state_dict(torch.load(os.path.join(args.logdir, kw+'model.pth')))
	else:
		print('======loading best_model')
		model.load_state_dict(torch.load(os.path.join(args.logdir, kw + 'model.pth')))
		optimizer.load_state_dict(torch.load(os.path.join(args.logdir,kw + 'optim.pth')))


ema = EMA(model, 0.999)

if args.ema:
	ema.register()

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)
if args.Test:
	trainer.test()
	print('Test finish')
else:
	dist.barrier()
	for epoch in range(trainer.cur_epoch, args.epochs):
		train_sampler.set_epoch(epoch)
		print('epoch:',epoch)
		trainer.train()
		dist.barrier()

		if not args.finetune:
			trainer.validate()
			# dist.barrier() # NECESSARY HERE OR NOT?
			trainer.save()
		if args.scheduler:
			print('lr', scheduler.get_lr())
			scheduler.step()
