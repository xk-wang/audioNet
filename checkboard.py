# check if the key board can be divide into three parts

from cProfile import label
import os
import numpy as np

labeldir = '/home/data/wxk/lab_dataset/OMAPS/label'
areas = [[ i for i in range(1, 28)], 
         [ i for i in range(28, 64)],
         [ i for i in range(64, 89)]
        ]
sr = 16000
hop_len = 512

for filename in os.listdir(labeldir):
  print(filename)
  filepath = os.path.join(labeldir, filename)
  data = np.loadtxt(filepath)
  size = int(np.ceil(data[-1, 2]/hop_len*sr))
  labels = np.zeros((size, 88), dtype=np.int8)

  for onset, offset, pitch in data:
    onset_frame = int(np.ceil(onset/hop_len*sr))
    start_frame = max(onset_frame-1, 0)
    end_frame = min(onset_frame+2, size)
    labels[onset_frame:end_frame, int(pitch)-21] = 1

  for i in range(size):
    pitches = np.where(labels[i]==1)[0]
    # raise ValueError(pitches)

    a = -1
    for pitch in pitches:
      if a==-1:
        if pitch+1 in areas[0]:
          a=0
        elif pitch+1 in areas[1]:
          a=1
        else:
          a=2
        
      else:
        if pitch+1 not in areas[a]:
          print(pitches)
          break