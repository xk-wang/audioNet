
from torch.utils.data import Dataset
import numpy as np
import torch
import pickle
import os
from data.process_data import multi_process_data
from data.process_data import generate_data as data_generate_data
from data.configs import cqt_config
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class OMAPSDataset(Dataset):
    def __init__(self, data_config, is_trainset=True, 
                 transform_config=cqt_config ,device=DEFAULT_DEVICE):
        # read data from disk
        self.transform_config = transform_config
        self.data_config = data_config
        if is_trainset:
          if not os.path.exists(data_config.train_save_dir) or \
            data_config.train_audios != len(os.listdir(data_config.train_save_dir)):

            os.makedirs(data_config.train_save_dir, exist_ok=True)
            self.generate_data(data_config.train_img_dir, data_config.train_dir,
                          data_config.label_dir, data_config.train_save_dir)

          self.data = self.load(data_config.train_save_dir)

        else:
          if not os.path.exists(data_config.valid_save_dir) or \
            data_config.valid_audios != len(os.listdir(data_config.valid_save_dir)):

            os.makedirs(data_config.valid_save_dir, exist_ok=True)
            self.generate_data(data_config.valid_img_dir, data_config.valid_dir,
                          data_config.label_dir, data_config.valid_save_dir)
            
          self.data = self.load(data_config.valid_save_dir)
        
        self.data_num = len(self.data)

        if is_trainset:
          print("\n****** data num in OMAPS trainset: %d ******\n"%self.data_num)
        else:
          print("\n****** data num in OMAPS valset:   %d  ******\n"%self.data_num)

        # padding begin and end frames
        self.T_padding = transform_config.T//2
        paddings = [[np.zeros((transform_config.spec_len, transform_config.n_bins), dtype=np.float32),
                     np.zeros((88,), dtype=np.int8)] for i in range(self.T_padding)]
        
        self.data.extend(paddings)
        self.data[0:0] = paddings

        self.device = device          

    def __getitem__(self, index):
        # get five frames
        specs = []
        for i in range(index, index+self.transform_config.T):
          specs.append(self.data[i][0])
          if type(self.data[i][0]) == list:
            raise ValueError(self.data[i])
        specs = np.stack(specs, axis=0)
        return torch.from_numpy(specs).to(self.device), \
               torch.from_numpy(self.data[index+self.T_padding][1]).to(self.device)

    def __len__(self):
        return self.data_num

    def generate_data(self, img_dir, audio_dir, label_dir, save_dir):
      imgdirs = []
      audiopaths = []
      labelpaths = []
      savepaths = []
      # names = ['yty22_1x5_1'] #['yty22_1x5_1', 'yty20_5x7_6', ] #, 

      for audioname in sorted(os.listdir(audio_dir)):
        
        name = audioname.split('.')[0]
        # if name not in names: continue

        # print(audioname)
        sub_imgdir = os.path.join(img_dir, name)
        audiopath = os.path.join(audio_dir, audioname)
        labelpath = os.path.join(label_dir, name+'.txt')
        savepath = os.path.join(save_dir, name+'.pickle')
        imgdirs.append(sub_imgdir)
        audiopaths.append(audiopath)
        labelpaths.append(labelpath)
        savepaths.append(savepath)
      
      multi_process_data(imgdirs, audiopaths, labelpaths, savepaths)
      # data_generate_data([imgdirs[0], audiopaths[0], labelpaths[0], savepaths[0]])
      
    def load(self, data_dir):
      datas = []
      zeros = 0
      for filename in sorted(os.listdir(data_dir)):
        datapath = os.path.join(data_dir, filename)
        # print(filename)
        with open(datapath, 'rb') as f:   
          data = pickle.load(f)
          datas.extend(data)
          # for i, e in enumerate(data):
          #   print(i, np.where(e[1]==1)[0])
      return datas


class EvaluateDataset(Dataset):
    def __init__(self, data_config, is_training=False, transform_config=cqt_config ,device=DEFAULT_DEVICE):
        self.data_config = data_config
        if not is_training:
            self.data_dir = data_config.valid_save_dir
            self.audio_num = data_config.valid_audios
        else:
            self.data_dir = data_config.train_save_dir
            self.audio_num = data_config.valid_audios
    
        if len(os.listdir(self.data_dir)) != self.audio_num:
            raise ValueError("the datadir don't contain enough files!")

        self.datas = self.load(self.data_dir)
        self.data_num = len(self.datas)

        self.T_padding = transform_config.T//2
        paddings = [[np.zeros((transform_config.spec_len, transform_config.n_bins), dtype=np.float32),
                     np.zeros((88,), dtype=np.int8)] for i in range(self.T_padding)]
        
        self.datas.extend(paddings)
        self.datas[0:0] = paddings
        self.device = device

        self.transform_config = transform_config
          
    def __getitem__(self, index):
        # get five frames
        specs = []
        for i in range(index, index+self.transform_config.T):
          specs.append(self.datas[i][0])
        specs = np.stack(specs, axis=0)
        return torch.from_numpy(specs).to(self.device), \
               torch.from_numpy(self.datas[index+self.T_padding][1]).to(self.device)

    def __len__(self):
        return self.data_num

    def load(self, data_dir):
        datas = []
        for filename in sorted(os.listdir(data_dir)):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'rb') as f:
              datas.extend(pickle.load(f))
        return datas

class TestDataset(Dataset):
    def __init__(self, datapath, transform_config=cqt_config, device=DEFAULT_DEVICE):
        # read data from disk
        with open(datapath, 'rb') as f:
          self.data = pickle.load(f)
        self.data_num = len(self.data)
        self.transform_config = transform_config

        print("\n****** data num in TEST DATASET: %d ******\n"%self.data_num)

        # padding begin and end frames
        self.T_padding = transform_config.T//2
        paddings = [np.zeros((transform_config.spec_len, transform_config.n_bins), dtype=np.float32) \
                     for i in range(self.T_padding)]
        
        self.data.extend(paddings)
        self.data[0:0] = paddings
        self.device = device
          
    def __getitem__(self, index):
        # get five frames
        specs = []
        for i in range(index, index+self.transform_config.T):
          specs.append(self.data[i])
          # print(self.data[i].shape)
        specs = np.stack(specs, axis=0)
        return torch.from_numpy(specs).to(self.device)

    def __len__(self):
        return self.data_num