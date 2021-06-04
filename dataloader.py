import os
import random
import numpy as np
import torch
import torch.utils.data
import numpy as np
from scipy.io.wavfile import read
import torch


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class DBRLoader(torch.utils.data.Dataset):
    def __init__(self, datalist, hparams):
        self.datalist = datalist
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.stft = STFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        '''
        max_wav_value=32768.0,
        sampling_rate=16000,  # sox 이용해서 downsampling, mono wav로 변환 후 사용
        filter_length=1024,
        hop_length=80,
        win_length=160,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        '''

        random.seed(1234)
        random.shuffle(self.datalist)

    def get_mel_label_pair(self, audiopath):
        label = self.get_label(audiopath.split('/')[-2])
        mel = self.get_mel(audiopath)
        return (label, mel)

    def get_label(self, label):
        if 'dog' in label:
            l = 0
        elif 'bird' in label:
            l = 1
        else:  # rain
            l = 2

        onehot = [0] * 3
        onehot[l] = 1
        return torch.LongTensor(onehot)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)

        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)  # 참고용: mel spectrogram으로 변환하는 코드 찾아서 사용할 것
        melspec = torch.squeeze(melspec, 0)

        return melspec

    def __getitem__(self, index):
        return self.get_mel_label_pair(self.datalist[index])

    def __len__(self):
        return len(self.datalist)


class DBRCollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)

        label_len = input_lengths[0]

        label = torch.LongTensor(len(batch), label_len)
        label.zero_()
        for i in range(len(ids_sorted_decreasing)):
            l = batch[ids_sorted_decreasing[i]][0]
            label[i, :] = l

        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel

            output_lengths[i] = mel.size(1)

        return label, mel_padded, output_lengths
