from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data import DataLoader
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list


# new added
from Syncnet import SyncNetPerception
from dataset.dataset_syncnet_train import DINetDataset
import logging
import pdb



parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16


logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data, train_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    criterionMSE = nn.MSELoss().cuda()

    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (source_clip, deep_speech_full, clip_flag) in prog_bar:
        # for iteration, data in enumerate(train_data_loader):
        #     source_clip, deep_speech_full, clip_flag = data
            # if clip_flag:
            #     real_tensor = torch.tensor(1.0).cuda()
            # else:
            #     real_tensor = torch.tensor(0.0).cuda()
            print("source_clip1.shape: ", source_clip.shape)
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_full = deep_speech_full.float().cuda()
            clip_flag = clip_flag.cuda()
            
            print("source_clip2.shape: ", source_clip.shape)
            
            source_clip = torch.cat(torch.split(source_clip, hparams.syncnet_batch_size, dim=0), 1)
            
            print("source_clip3.shape: ", source_clip.shape)
            
            source_clip_mouth = source_clip[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
                                            train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]
            print("source_clip_mouth.shape: ", source_clip_mouth.shape)
            
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            source_clip_mouth = source_clip_mouth.to(device)
            deep_speech_full = deep_speech_full.to(device)

            # a, v = model(mel, x)
            # y = y.to(device)
            # loss = cosine_loss(a, v, y)

            sync_score = model(source_clip_mouth, deep_speech_full)
            print("sync_score shape: ", sync_score.shape)
            print("clip_flag.shape: ", clip_flag.shape)
            loss = criterionMSE(sync_score, clip_flag)
            # loss_sync = criterionMSE(sync_score, real_tensor.expand_as(sync_score)) * opt.lamb_syncnet_perception 
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
    import pdb
    pdb.set_trace()

    # load training data
    train_data_json_path = "/workspace/DINet/asserts/training_data/training_json_debug.json"
    train_data_augment_num = 160
    train_data_mouth_region_size = 256
    
    hparams.syncnet_batch_size = 6
    
    train_data = DINetDataset(train_data_json_path, train_data_augment_num, train_data_mouth_region_size)
    train_data_loader = DataLoader(dataset=train_data,  batch_size=hparams.syncnet_batch_size, shuffle=True,drop_last=True)
    train_data_length = len(train_data_loader)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    # model = SyncNet().to(device)
    model = SyncNetPerception("").to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data, train_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
