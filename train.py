import os
from time import time
from datetime import datetime

import torch

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import Evaluate, MovingMetrics
from model import AudioNet

from data import OMAPSDataset
from data import omaps_config

def cycle(iterable):
    while True:
        for item in iterable:
            yield item

# ex = Experiment('train_transcriber')
# @ex.config
def get_config():
    # strftime('%y%m%d-%H%M%S')
    logdir = '/home/wxk/py/audioNet/runs/transcriber-' + datetime.now().strftime('%y%m%d-%H')
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    # these parameters will be get after getting the data num
    # the trainset: 293955 train neg 251612 validset: 95550 
    # 3691MiB 
    # 1e-5 loss降得太慢
    # 换用第二种思路，直接去掉3D卷积直接使用5帧的频谱数据来预测中间帧的模型结果
    # 现在的5帧模型实际的感受野还要多4帧
    # 预测完了之后的第一步是将模型的帧数进行调整
    # 第二步是将GRU引入音频模型中进行时序相关性学习
    # 考虑是否添加音频的分区, 引入3类别的多标签
    # 考虑音频onset-aware分支

    iterations = 120000
    resume_iteration = None
    checkpoint_interval = 2500
    train_on = 'OMAPS'
    batch_size = 64

    learning_rate = 1e-3
    learning_rate_decay_steps = 2500
    learning_rate_decay_rate = 0.98

    train_log_interval = 50
    valid_log_interval = 200
    validation_interval = 2500

    # ex.observers.append(FileStorageObserver.create(logdir))

    return logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, \
           learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, train_log_interval, \
           valid_log_interval, validation_interval


# @ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, learning_rate,
          learning_rate_decay_steps, learning_rate_decay_rate, train_log_interval, valid_log_interval, validation_interval):
    # print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    if train_on == 'OMAPS':
        dataset = OMAPSDataset(omaps_config, is_trainset=True, device=device)
        validation_dataset = OMAPSDataset(omaps_config, is_trainset=False, device=device)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(validation_dataset, batch_size, shuffle=False, drop_last=False)

    if resume_iteration is None:
        model = AudioNet(1, 1024).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = AudioNet(1, 1024).to(device)
        model.load_state_dict(torch.load(model_path))
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    # summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = range(resume_iteration + 1, iterations + 1)
    # moving_metrics = MovingMetrics(threshold=0.5, pos_weight=4)
    evaluator = Evaluate(model, dataloader=valid_loader, is_training=False, 
                         log_interval = valid_log_interval, device=device)


    t = time()
    duration_saving = 0
    duration_evaluating = 0
    for i, batch in zip(loop, cycle(loader)):
        # should also be evaluated during training
        metrics = model.run_on_batch(batch)
        loss = metrics["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % train_log_interval == 0:
            duration = time() - t - duration_saving - duration_evaluating
            t = time()
            print("====== training results: %.2fs ======"%duration)
            model.log(metrics, i)
            writer.add_scalar("loss", metrics["loss"], global_step=i)
            writer.add_scalar("f", metrics["f"], global_step=i)
            writer.add_scalar("p", metrics["p"], global_step=i)
            writer.add_scalar("r", metrics["r"], global_step=i)

        t_saving = time()
        if i % checkpoint_interval == 0:
            print("====== saving checkpoints ======")
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
        duration_saving = time() - t_saving

        t_evaluating = time()
        # if i % validation_interval == 0:

        #     print("====== evaluating results ======")
        #     model.eval()
        #     with torch.no_grad():
        #         metrics = evaluator.evaluate()
        #         writer.add_scalar('validation/average_loss', metrics["average_loss"], global_step=i)
        #         writer.add_scalar('validation/p', metrics["p"], global_step=i)
        #         writer.add_scalar('validation/r', metrics["r"], global_step=i)
        #         writer.add_scalar('validation/f', metrics["f"], global_step=i)

        #     model.train()
        duration_evaluating = time() - t_evaluating

        


if __name__ == '__main__':
    configs = get_config()
    train(*configs)