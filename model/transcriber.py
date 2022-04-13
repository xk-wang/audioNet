from sklearn import metrics
import torch.nn.functional as F
from torch import nn
import torch
import sys
import numpy as np
eps = sys.float_info.epsilon

# only consider about the onset branch
class AudioNet(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.onset_3dconv = nn.Sequential(
            nn.Conv3d(input_features, output_features // 4, (3, 3, 3), padding=(0, 1, 1)), # D, H, W
            nn.BatchNorm3d(output_features // 4),
            nn.ReLU(),
            nn.Conv3d(output_features // 4, output_features // 2, (3, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(output_features // 2),
            nn.ReLU()
        )
        
        self.onset_2dconv = nn.Sequential(
          nn.Conv2d(output_features // 2, output_features // 2, (3, 3)),
          nn.BatchNorm2d(output_features // 2),
          nn.ReLU(),
          nn.MaxPool2d((1, 2)),
          nn.Conv2d(output_features // 2, output_features, (3, 3)),
          nn.BatchNorm2d(output_features),
          nn.ReLU(),
          nn.MaxPool2d((1, 2)),
          nn.Conv2d(output_features, output_features, (3, 3), padding=(1, 0)),
          nn.BatchNorm2d(output_features),
          nn.ReLU(),
          nn.MaxPool2d((1, 2)),
        )

        # using gap
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.onset_fc = nn.Sequential(
          nn.Linear(output_features, 512),
          nn.Dropout(0.5),
          nn.Linear(512, 88),
        )
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        

    def forward(self, spec):
        # normalize [N, 5, 5, 352]
        base, _ = torch.max(spec, dim=2, keepdim=True)
        base, _ = torch.max(base, dim=3, keepdim=True)
        spec = spec / (base+eps)
        spec = spec.unsqueeze(1)
        features_3d = self.onset_3dconv(spec)
        features_3d = features_3d.squeeze(2)
        features_2d = self.onset_2dconv(features_3d)
        features_gap = self.gap(features_2d).squeeze()
        logits = self.onset_fc(features_gap)

        return logits

    def run_on_batch(self, batch):
      specs, labels = batch
      labels = labels.float()
      logits = self(specs)
      metrics = self.get_metrices(logits, labels, threshold=0.5, pos_weight=4)
      return metrics


    def get_metrices(self, logits, labels, threshold=0.5, pos_weight=4):
      probs = self.sigmoid(logits)
      predictions = (probs>threshold).float()
      # raise ValueError(predictions.shape, labels.shape)
      TP = int(torch.sum(predictions*labels))
      FP = int(torch.sum(predictions*(1-labels)))
      FN = int(torch.sum((1-predictions)*labels))

      p = TP / (TP+FP+eps)
      r = TP / (TP+FN+eps)
      f = 2 * p * r / (p + r + eps)

      mask = torch.where(labels==1, pos_weight, 1)
      loss = self.bce_loss(logits, labels)
    #   loss = torch.binary_cross_entropy_with_logits(onset_logits, onset_labels, reduction='none')
      mask_loss = torch.sum(mask*loss)

      return {"loss" : mask_loss, "p": p, "r": r, "f": f, "TP": TP, "FP": FP, "FN": FN, "samples": int(logits.shape[0])}

    def log(self, metrics, step=None):
      if step is not None:
            print("step: %d loss: %-6.2f F1: %-4.2f%% P: %-4.2f%% R: %-4.2f%%"%(
                step, metrics["loss"], metrics["f"]*100, metrics["p"]*100, metrics["r"]*100
            ))
      else:
          print("final loss: %-6.2f F1: %-4.2f%% P: %-4.2f%% R: %-4.2f%%"%(
              metrics["loss"], metrics["f"]*100, metrics["p"]*100, metrics["r"]*100
          ))