from time import time
import sys
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader
from data import EvaluateDataset
from model import AudioNet
from data import omaps_config

eps = sys.float_info.epsilon
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# implement a moving average metrics
class MovingMetrics:
    def __init__(self) -> None:
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.loss = 0
        self.samples = 0

    def step(self, metrices):
        samples = metrices["samples"]
        TP = metrices["TP"]
        FP = metrices["FP"]
        FN = metrices["FN"]
        loss = metrices["loss"]

        self.TP += TP
        self.FP += FP
        self.FN += FN
        self.samples += samples
        self.loss += loss

    def average_metrics(self):
        p = self.TP / (self.TP + self.FP + eps)
        r = self.TP / (self.TP + self.FN + eps)
        f = 2 * p * r / (p+r+eps)
        loss = self.loss / (self.samples+eps)

        return {"p": p, "r": r, "f": f, "loss": loss}
        

    def log(self, metrics, step=None):
        if step is None:
            print("final loss: %-6.2f F1: %-4.2f%% P: %-4.2f%% R: %-4.2f%%"%(
                metrics["loss"], metrics["f"]*100, metrics["p"]*100, metrics["r"]*100
            ))
        else:
            print("step %-4d loss: %-6.2f F1: %-4.2f%% P: %-4.2f%% R: %-4.2f%%"%(
                step, metrics["loss"], metrics["f"]*100, metrics["p"]*100, metrics["r"]*100
            ))
    
    def zeros(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.loss = 0
        self.samples = 0


class Evaluate:
    def __init__(self, model, dataloader=None, data_config=None, is_training=False, 
                log_interval = None, device=DEFAULT_DEVICE) -> None:
        if(dataloader is None and data_config is None):
            raise ValueError("You must provide dataloader or data_config!")
        self.model = model
        self.model.eval()
        self.dataloader = dataloader

        if self.dataloader is None:
            dataset = EvaluateDataset(data_config, is_training=is_training, device=device)
            self.dataloader = DataLoader(dataset, 64, shuffle=False, drop_last=False)
        
        self.moving_metrics = MovingMetrics()
        self.log_interval = log_interval if log_interval is not None else 1

    def evaluate(self):
        self.moving_metrics.zeros()
        with torch.no_grad():
            t = time()
            for i, batch in enumerate(self.dataloader):
                metrics = self.model.run_on_batch(batch)
                self.moving_metrics.step(metrics)
                if (i+1) % self.log_interval == 0:
                    duration = time() - t
                    t = time()
                    print("====== evaluating results: %.2fs ======"%duration)
                    self.moving_metrics.log(metrics, i+1)
        average_metrics = self.moving_metrics.average_metrics()
        self.moving_metrics.log(average_metrics)
        return metrics


if __name__ == '__main__':
    device = "cuda:2"
    model_path = '/home/wxk/py/audioNet/runs/transcriber-220412-16/model-37500.pt'
    # model = AudioNet(1, 1024)
    # model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = torch.load(model_path, map_location='cpu') # ignore the training cpu number
    model.to(device)
    
    evaluator = Evaluate(model, data_config=omaps_config, log_interval=100, device=device)
    evaluator.evaluate()
