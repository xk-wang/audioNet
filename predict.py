from multiprocessing import Pool
from torch.utils.data import DataLoader
from data import TestDataset
from time import time
import librosa
import pickle
import torch
import numpy as np
import os
from functools import partial
from data import cqt_config, test_config, omaps_config
from model import AudioNet

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Prediction:
  def __init__(self, model, test_config, device=DEFAULT_DEVICE) -> None:

    self.model = model
    self.model.eval()
    self.audiodir = test_config.data_dir
    self.imgdir = test_config.img_dir
    self.savedir = test_config.save_dir
    self.FPS = test_config.FPS
    self.save_res_dir = test_config.save_res_dir
    self.audios = test_config.audios
    self.process = test_config.process
    self.device = device
    
    save_nums = 0
    if os.path.exists(self.savedir):
      savenames = [filename for filename in os.listdir(self.savedir) if filename.endswith("pickle")]
      save_nums = len(savenames)

    if not os.path.exists(self.savedir) or save_nums!=self.audios:

      os.makedirs(self.savedir, exist_ok=True)

      paths = []
      for filename in sorted(os.listdir(self.audiodir)):
        audiopath = os.path.join(self.audiodir, filename)
        filename = filename.split('.')[0]
        imgdir = os.path.join(self.imgdir, filename)
        savepath = os.path.join(self.savedir, filename+".pickle")

        paths.append([imgdir, audiopath, savepath])
      Prediction.multi_process_data(paths, self.FPS, self.process)

    os.makedirs(self.save_res_dir, exist_ok=True)

  def predict(self, threshold=0.5):

    with torch.no_grad():

      filenames = sorted([filename.replace(".frame", "") for filename in os.listdir(self.savedir) \
                  if filename.endswith(".frame")])
      t = time()
      for filename in filenames:
        savepath = os.path.join(self.save_res_dir, filename + ".txt")
        saveraw_path = os.path.join(self.save_res_dir, filename + ".raw")
      
        filepath = os.path.join(self.savedir, filename+".pickle")
        dataset = TestDataset(filepath, transform_config=cqt_config, device=self.device)
        loader = DataLoader(dataset, 64, shuffle=False, drop_last=False)
        audio_times = np.loadtxt(os.path.join(self.savedir, filename + ".frame")) * \
                      cqt_config.hop_len / cqt_config.sr

        total_logits = []
        for batch in loader:
          logits = self.model(batch)
          total_logits.append(logits)

        total_logits = torch.cat(total_logits, axis=0)
        total_probs = self.model.sigmoid(total_logits)
        duration = time() - t
        print("====== testing %20s: %.2fs ======"%(filename, duration))

        if audio_times.shape[0]!=total_logits.shape[0]:
          raise ValueError("the audio frames doesn't equal to the logits shape")

        notes = self.onset_search(total_probs, audio_times, threshold)

        with open(saveraw_path, 'wt') as f:
          for i, probs in enumerate(total_probs):
            info = [audio_times[i]] + probs.tolist()
            info = tuple(info)
            format_string = " ".join([ "%.4f" for i in range(89) ]) + "\n"
            f.write(format_string%info)
        
        with open(savepath, "wt") as f:
          for onset, pitch in notes:
            f.write("%-7.4f  %-7.4f  %-3d\n"%(onset, onset+0.1, pitch))

  def onset_search(self, onset_probs, audio_times, threshold=0.5):
    onset_probs = onset_probs.cpu().numpy().T
    # raise ValueError(onset_probs.shape, onset_probs.max(), onset_probs.min())
    onset_masks = onset_probs > threshold # 88xT
    notes = []

    for i in range(88):
      onset_mask = onset_masks[i]
      valid_indexs = np.argwhere(onset_mask)[:, 0]
      if len(valid_indexs)==0:
        continue

      start = last = valid_indexs[0]
      for idx in valid_indexs[1:]:
          if idx - last > 1:
              onset_max_idx = np.argmax(onset_probs[i, start:last+1], axis=0) + start
              notes.append([audio_times[onset_max_idx], i+21])
              start=idx
          last=idx

    notes.sort(key=lambda x: x[0])
    return notes
    
  # multi process
  @staticmethod
  def multi_process_data(paths, FPS, process=8):
    pool = Pool(process)
    worker = partial(Prediction.generate_data, FPS=FPS)
    res = pool.map_async(worker, paths)
    print(res.get())
    pool.close()
    pool.join()

  @staticmethod
  def generate_data(path, FPS=30):
    imgdir, audiopath, savepath = path
    img_nos = []
    for filename in os.listdir(imgdir):
      imgno = int(filename.split('.jpg')[0])
      img_nos.append(imgno)
    img_nos.sort()
    if img_nos[-1] != len(img_nos)-1:
      raise ValueError("iscontinuity of imgdir!")

    audio, sr = librosa.load(audiopath, sr=cqt_config.sr, mono=cqt_config.mono)
    spec = librosa.cqt(audio, sr=cqt_config.sr, hop_length=cqt_config.hop_len,
                        fmin=cqt_config.fmin, bins_per_octave=cqt_config.bins_per_octave,
                        n_bins=cqt_config.n_bins)
    spec = np.abs(spec).T.astype(np.float32) # Tx352
    T, _ = spec.shape
    padding = int(cqt_config.spec_len/2)
    spec = np.pad(spec, ((padding, padding), (0, 0)))

    f = open(savepath,'wb')
    datas = []
    audio_frames = []
    for imgno in img_nos:
      audio_frame = int(np.round(imgno*sr/FPS/cqt_config.hop_len))
      audio_frames.append(audio_frame)
      spec_mid = audio_frame+padding
      subspec = spec[spec_mid-2:spec_mid+3]
      datas.append(subspec)
    pickle.dump(datas, f)
    f.close()

    save_frame_path = savepath.replace(".pickle", ".frame")
    with open(save_frame_path, 'wt') as f:
      for frame in audio_frames:
        f.write("%d\n"%frame)

if __name__ == '__main__':
  device = "cuda:0"
  # frame results
  # '/home/wxk/py/audioNet/runs/transcriber-220412-16/model-37500.pt'
  # final loss: 2.93   F1: 74.94% P: 75.24% R: 74.65%
  # mean op F1: 85.51% P: 88.82% R:  82.58%


  model_path = '/home/wxk/py/audioNet/runs/transcriber-220412-16/model-52500.pt'
  model = torch.load(model_path, map_location='cpu')
  model.to(device)
  
  predictor = Prediction(model, test_config, device=device)
  predictor.predict()