import os
from dataclasses import dataclass
from typing import Optional, List, Callable, Any

import numpy as np
import torch
import torchaudio
from einops import rearrange
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import Tensor
from torch.nn import functional as F, Identity
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import Dataset

from utils import read_json, read_video, padding_video, padding_audio, resize_video, iou_with_anchors, resample_frames
import json
from torch.nn.utils.rnn import pad_sequence
import time


@dataclass
class Metadata:
    file: str
    n_fakes: int
    fake_periods: List[List[int]]
    duration: float
    original: Optional[str]
    modify_video: bool
    modify_audio: bool
    split: str
    video_frames: int
    audio_channels: int
    audio_frames: int


class LAVDF(Dataset):

    def __init__(self, subset: str, root: str = "data", frame_padding: int = 500,
        max_duration: int = 40, fps=25,
        video_transform: Callable[[Tensor], Tensor] = Identity(),
        audio_transform: Callable[[Tensor], Tensor] = Identity(),
        metadata: Optional[List[Metadata]] = None,
        get_meta_attr: Callable[[Metadata, Tensor, Tensor, Tensor], List[Any]] = None
    ):
        self.subset = subset
        self.root = root
        self.video_padding = frame_padding
        self.audio_padding = int(frame_padding / fps * 15999)
        self.max_duration = max_duration
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.get_meta_attr = get_meta_attr

        if subset == 'train':
            suffix = 'set_label.txt'
        elif subset == 'val':
            suffix = "set_label.txt"
        elif subset == 'test':
            suffix = "set_nolabel.txt"

        with open(os.path.join(root, subset + suffix), "r") as f:
            lines = f.readlines()
        self.wav_ids = [line.strip() for line in lines]

    def __getitem__(self, idx: int) -> List[Tensor]:


        files = self.wav_ids[idx]
        path = files.split(',')[0]
        name, ext = os.path.splitext(path)
        #fu_label = int(files.split(',')[1])
        if len(files.split(',')) > 1:
            fu_label = int(files.split(',')[1])
        else:
            fu_label = 1
        #time1 = time.time()
        #video, audio, info = read_video(os.path.join(self.root, self.subset+ 'set_25fps16k', path))

        #/mnt/disk_work1/liumiao/phase1/rgbfeat_trainset/
        video_name = os.path.join(self.root, 'rgbfeat_' + self.subset + 'set', name + '.npy')
        video_feats = np.load(video_name).astype(np.float32)
        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose()))


        audio_name = os.path.join(self.root, 'audiofeat_' + self.subset + 'set', name + '.npy')
        audio_feats = np.load(audio_name).astype(np.float32)
        audio_feats = torch.from_numpy(np.ascontiguousarray(audio_feats.transpose()))

        if audio_feats.shape[1] != video_feats.shape[1]:

            resize_feats = F.interpolate(
                audio_feats.unsqueeze(0),
                size=video_feats.shape[1],
                mode='linear',
                align_corners=False
            )
            audio_feats = resize_feats.squeeze(0)

        #time2 = time.time()
        #print('read_time', time2-time1)
        #video = resample_frames(video_o, 30, 25)
        #time3 = time.time()
        #print('sample_time', time3-time2)
        #print('audio', audio.shape)
        #print('video', video.shape)
        #print('resize_time', time4-time3)
        n_frames = video_feats.shape[1]
        #print(n_frames)

        #print('video_feats', video_feats.shape)
        if self.video_padding > n_frames:
            video_feats = video_feats.transpose(0,1)
            audio_feats = audio_feats.transpose(0,1)
            video_feats = padding_audio(video_feats, target=self.video_padding)
            audio_feats = padding_audio(audio_feats, target=self.video_padding)
            video_feats = video_feats.transpose(0,1)
            audio_feats = audio_feats.transpose(0,1)

        return video_feats, audio_feats, n_frames, fu_label, path

    def __len__(self) -> int:
        return len(self.wav_ids)

    @staticmethod
    def _get_log_mel_spectrogram(audio: Tensor) -> Tensor:
        ms = torchaudio.transforms.MelSpectrogram(n_fft=321, n_mels=64)
        spec = torch.log(ms(audio[:, 0]) + 0.01)
        #assert spec.shape == (64, 2048), "Wrong log mel-spectrogram setup in Dataset"
        return spec


def _default_get_meta_attr(meta: Metadata, video: Tensor, audio: Tensor, label: Tensor) -> List[Any]:
    return [meta.video_frames]


class LAVDFDataModule(LightningDataModule):
    train_dataset: LAVDF
    dev_dataset: LAVDF
    test_dataset: LAVDF
    metadata: List[Metadata]

    def __init__(self, root: str = "data", frame_padding=512, max_duration=40, batch_size=1, num_workers=0,
        take_train: int = None, take_dev: int = None, take_test: int = None,
        cond: Optional[Callable[[Metadata], bool]] = None,
        get_meta_attr: Callable[[Metadata, Tensor, Tensor, Tensor], List[Any]] = _default_get_meta_attr
    ):
        super().__init__()
        self.root = root
        self.frame_padding = frame_padding
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.take_train = take_train
        self.take_dev = take_dev
        self.take_test = take_test
        self.cond = cond
        self.get_meta_attr = get_meta_attr

    def collater(self, samples):
        video_list = []
        audio_list = []
        n_frames_list = []
        fu_label_list = []
        path_list = []        

        for video, audio, n_frames, fu_label, path in samples:
            video_list.append(video.transpose(0,1))
            audio_list.append(audio.transpose(0,1))
            n_frames_list.append(n_frames)
            fu_label_list.append(fu_label)
            path_list.append(path)

        video_list = pad_sequence(video_list, batch_first=True).transpose(1,2)
        audio_list = pad_sequence(audio_list, batch_first=True).transpose(1,2)

        fu_label_list = torch.tensor(fu_label_list)
        n_frames_list = torch.tensor(n_frames_list)
        
        return video_list, audio_list, n_frames_list, fu_label_list, path_list

    def setup(self, stage: Optional[str] = None) -> None:

        #self.train_dataset = LAVDF("train", self.root, self.frame_padding, self.max_duration, get_meta_attr=self.get_meta_attr
        #)
        #self.val_dataset = LAVDF("val", self.root, self.frame_padding, self.max_duration, get_meta_attr=self.get_meta_attr
        #)
        self.test_dataset = LAVDF("test", self.root, self.frame_padding, self.max_duration, get_meta_attr=self.get_meta_attr
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collater,
            sampler=RandomSampler(self.train_dataset, num_samples=self.take_train, replacement=False), pin_memory=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collater, pin_memory=True)
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collater, num_workers=self.num_workers)
