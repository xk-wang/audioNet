class CQT_CONFIG(object):
  def __init__(self) -> None:
    self.sr = 16000
    self.mono = True
    self.hop_len = 512
    self.fmin = 27.5
    self.n_bins = 352
    self.bins_per_octave = 48
    self.T = 5

    self.spec_len = 5
    # don't use blurr label
  
  def __new__(cls, *args, **kwargs):
    if not hasattr(cls,'instance'):
        cls.instance=super(CQT_CONFIG, cls).__new__(cls)
    return cls.instance

class OMAPS_CONFIG(object):
  def __init__(self) -> None:
      self.train_dir = '/home/data/wxk/lab_dataset/OMAPS/train/mp3'
      self.valid_dir = '/home/data/wxk/lab_dataset/OMAPS/test/mp3'
      self.train_img_dir = '/home/data/wxk/lab_dataset/OMAPS/train/cut_images'
      self.valid_img_dir = '/home/data/wxk/lab_dataset/OMAPS/test/cut_images'
      self.label_dir = '/home/data/wxk/lab_dataset/OMAPS/label'
      self.train_save_dir = '/home/data/wxk/lab_dataset/OMAPS_data/train'
      self.valid_save_dir = '/home/data/wxk/lab_dataset/OMAPS_data/valid'      

      self.FPS = 30
      self.process = 16

      self.train_audios = 80
      self.valid_audios = 26
    
  def __new__(cls, *args, **kwargs):
    if not hasattr(cls,'instance'):
        cls.instance=super(OMAPS_CONFIG, cls).__new__(cls)
    return cls.instance


class TEST_CONFIG(object):
  def __init__(self) -> None:
    self.data_dir = '/home/data/wxk/lab_dataset/OMAPS/test/mp3'
    self.img_dir = '/home/data/wxk/lab_dataset/OMAPS/test/cut_images'
    self.save_dir = '/home/data/wxk/lab_dataset/OMAPS_data/test'
    self.save_res_dir = '/home/data/wxk/lab_dataset/OMAPS_data/res_save'
    
    self.FPS = 30
    self.process = 16

    self.audios = 26

  def __new__(cls, *args, **kwargs):
    if not hasattr(cls,'instance'):
        cls.instance=super(TEST_CONFIG, cls).__new__(cls)
    return cls.instance

cqt_config = CQT_CONFIG()
omaps_config = OMAPS_CONFIG()
test_config = TEST_CONFIG()
