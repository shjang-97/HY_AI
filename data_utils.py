import os
import time
import argparse
import math
import random

import torch
from torch.utils.data import DataLoader

from dataloader import DBRLoader, DBRCollate




def read_dic(path, ext=None):
    # dictionary 내에 있는 모든 파일 읽어 리스트로 출력
    import fnmatch
    files = []
    for root, dirnames, filenames in os.walk(path):
        if ext != None:
            for filename in fnmatch.filter(filenames, '*.wav'):
                files.append(os.path.join(root, filename))
        else:
            for filename in filenames:
                files.append(os.path.join(root, filename))
    return files


def prepare_dataloaders(hparams):
    filelist = read_dic(hparams.filepath, ext='.wav')  # filepath : '/media/sh/DB/AI_dbr'

    random.shuffle(filelist)
    len_files = len(filelist)

    trainlist = filelist[:int(len_files * 0.9)]
    testlist = filelist[int(len_files * 0.9):]

    trainset = DBRLoader(trainlist, hparams)
    testset = DBRLoader(testlist, hparams)

    collate_fn = DBRCollate(hparams.n_frames_per_step)

    train_sampler = None
    shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=16, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, testset, collate_fn

