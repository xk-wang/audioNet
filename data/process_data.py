from multiprocessing import Pool
import librosa
import pickle
import numpy as np
import os

from data.configs import CQT_CONFIG, OMAPS_CONFIG

cqt_config = CQT_CONFIG()
omaps_config = OMAPS_CONFIG()

# multi process
def multi_process_data(imgdirs, audiopaths, labelpaths, savepaths):
  pool = Pool(omaps_config.process)
  paths = []
  for i in range(len(audiopaths)):
    paths.append([imgdirs[i], audiopaths[i], labelpaths[i], savepaths[i]])
  res = pool.map_async(generate_data, paths)
  print(res.get())
  pool.close()
  pool.join()


def generate_data(paths):
  imgdir, audiopath, labelpath, savepath = paths
  img_nos = []
  for filename in os.listdir(imgdir):
    imgno = int(filename.split('.jpg')[0])
    img_nos.append(imgno)
  img_nos.sort()
  if img_nos[-1] != len(img_nos)-1:
    raise ValueError("iscontinuity of imgdir!")
  #
  audio, sr = librosa.load(audiopath, sr=cqt_config.sr, mono=cqt_config.mono)
  spec = librosa.cqt(audio, sr=cqt_config.sr, hop_length=cqt_config.hop_len,
                      fmin=cqt_config.fmin, bins_per_octave=cqt_config.bins_per_octave,
                      n_bins=cqt_config.n_bins)

  # raise ValueError(audiopath)
  spec = np.abs(spec).T.astype(np.float32) # Tx352
  T, _ = spec.shape
  padding = int(cqt_config.spec_len/2)
  spec = np.pad(spec, ((padding, padding), (0, 0)))
  #
  notes = np.loadtxt(labelpath, np.float32)
  labels = np.zeros(shape=(T, 88), dtype=np.int8)
  for onset, offset, pitch in notes:
    onset_frame = int(np.round(onset*cqt_config.sr/cqt_config.hop_len))
    # labels[onset_frame, int(pitch-21)] = 1
    # print(onset_frame, pitch)
    start_frame = max(onset_frame-1, 0)
    end_frame = min(onset_frame+2, T)
    labels[start_frame: end_frame, int(pitch-21)] = 1
  #
  # raise ValueError(img_nos, len(img_nos))
  f = open(savepath,'wb') 
  datas = []
  for imgno in img_nos:
    audio_frame = int(np.round(imgno*sr/omaps_config.FPS/cqt_config.hop_len))
    spec_mid = audio_frame+padding
    # feature_name = filename+"_"+str(spec_mid)
    subspec = spec[spec_mid-2:spec_mid+3]
    sublabel = labels[audio_frame]
    datas.append([subspec, sublabel])
  pickle.dump(datas, f)
  f.close()


if __name__ == '__main__':
  # imgdir = '/home/data/wxk/lab_dataset/OMAPS/train/cut_images/playing3'
  # audiopath = '/home/data/wxk/lab_dataset/OMAPS/train/mp3/playing3.mp3'
  # labelpath = '/home/data/wxk/lab_dataset/OMAPS/train/label/playing3.txt'
  # savedir = '/home/data/wxk/lab_dataset/OMAPS'
  # filename = os.path.basename(imgdir)
  # savepath = os.path.join(savedir, filename+'.pickle')
  # generate_data(imgdir, audiopath, labelpath, savepath)
  
  # pickkle_file1 = open(savepath,'rb')
  # data = pickle.load(pickkle_file1)
  # print(len(data), len(data[0]))

  imgdirs = ['/home/data/wxk/lab_dataset/OMAPS/train/cut_images/playing3',
             '/home/data/wxk/lab_dataset/OMAPS/train/cut_images/playing4']
  audiopaths = ['/home/data/wxk/lab_dataset/OMAPS/train/mp3/playing3.mp3',
                '/home/data/wxk/lab_dataset/OMAPS/train/mp3/playing4.mp3']
  labelpaths = ['/home/data/wxk/lab_dataset/OMAPS/train/label/playing3.txt',
                '/home/data/wxk/lab_dataset/OMAPS/train/label/playing4.txt']
  savepaths = ['/home/data/wxk/lab_dataset/OMAPS/playing3.pickle',
               '/home/data/wxk/lab_dataset/OMAPS/playing4.pickle']

  multi_process_data(imgdirs, audiopaths, labelpaths, savepaths)