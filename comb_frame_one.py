# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch
import glob
from mmaction.datasets.pipelines import Compose
#from mmengine.dataset import Compose
from mmaction.models import build_model


from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm

from utils import padding_audio
import shutil
class CombinedModel(torch.nn.Module):
    def __init__(self, model, model_a, model_v):
        super(CombinedModel, self).__init__()
        # 你可以根据你的需要来组合这些层
        self.model = model
        self.model_a = model_a
        self.model_v = model_v

    def forward(self, audio, video):
        # 你可以根据需要定义模型的前向传播过程
        audio_feats = self.model_a(audio)
        video_feats = self.model_v(video, return_loss=False)
        #print('type', video_feats.dtype)
        #print(video_feats)
        #print('type', audio_feats.dtype)
        #print(audio_feats)
        audio_feats = audio_feats.squeeze(0).transpose(0,1)
        #video_feats = video_feats.transpose(0,1)
        video_feats = torch.from_numpy(np.ascontiguousarray(video_feats.transpose()))

        #print(audio_feats.shape)
        #print(video_feats.shape)

        if audio_feats.shape[1] != video_feats.shape[1]:
            resize_feats = F.interpolate(
                audio_feats.unsqueeze(0),
                size=video_feats.shape[1],
                mode='linear',
                align_corners=False
            )
            audio_feats = resize_feats.squeeze(0)

        n_frames = video_feats.shape[1]
        #print(n_frames)
        if 500 > n_frames:
            video_feats = video_feats.transpose(0,1)
            audio_feats = audio_feats.transpose(0,1)
            video_feats = padding_audio(video_feats, target=500)
            audio_feats = padding_audio(audio_feats, target=500)
            video_feats = video_feats.transpose(0,1)
            audio_feats = audio_feats.transpose(0,1)

        audio_feats = audio_feats.unsqueeze(0).cuda()
        video_feats = video_feats.unsqueeze(0).cuda()

        n_frames = torch.tensor(n_frames).unsqueeze(0).cuda()

        out = self.model(video_feats, audio_feats, n_frames)

        return out

class NetworkCommonMixIn():
    """Common mixin for network definition."""

    def load_weight(self, weight_file, device, state_dict=None, key_check=True):
        """Utility to load a weight file to a device."""

        state_dict = state_dict or torch.load(weight_file, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove unneeded prefixes from the keys of parameters.
        if key_check:
            weights = {}
            for k in state_dict:
                m = re.search(r'(^fc\.|\.fc\.|^features\.|\.features\.)', k)
                if m is None: continue
                new_k = k[m.start():]
                new_k = new_k[1:] if new_k[0] == '.' else new_k
                weights[new_k] = state_dict[k]
        else:
            weights = state_dict
        # Load weights and set model to eval().
        self.load_state_dict(weights)
        self.eval()
        logging.info(f'Using audio embbeding network pretrained weight: {Path(weight_file).name}')
        return self

    def set_trainable(self, trainable=False):
        for p in self.parameters():
            p.requires_grad = trainable


class AudioNTT2020Task6(nn.Module, NetworkCommonMixIn):
    """DCASE2020 Task6 NTT Solution Audio Embedding Network."""

    def __init__(self, n_mels, d):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x):
        x = self.features(x)       # (batch, ch, mel, time)       
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x = self.fc(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='Extract TSN Feature')
    # parser.add_argument('data_prefix', default='', help='dataset prefix')
    parser.add_argument('--output-prefix', default='', help='output prefix')
    parser.add_argument(
        '--data-list',
        help='video list of the dataset, the format should be '
        '`frame_dir num_frames output_file`')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=1,
        help='the sampling frequency of frame in the untrimed video')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--ckpt', help='checkpoint for feature extraction')
    parser.add_argument(
        '--part',
        type=int,
        default=0,
        help='which part of dataset to forward(alldata[part::total])')
    parser.add_argument(
        '--total', type=int, default=1, help='how many parts exist')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')
    args = parser.parse_args()
    return args


def inference(data_prefix,store_path,start_rate=0,end_rate=1):
    #warnings.filterwarnings('ignore')
    args = parse_args()
    args.is_rgb = args.modality == 'RGB'
    args.clip_len = 1 if args.is_rgb else 5
    args.input_format = 'NCHW' if args.is_rgb else 'NCHW_Flow'
    rgb_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False)
    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    args.in_channels = args.clip_len * (3 if args.is_rgb else 2)
    # max batch_size for one forward
    args.batch_size = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_re = torch.load('./ckpt/combined_model.pth')
    model_re.to(device)
    #print(model_re)

    # define the data pipeline for Untrimmed Videos
    data_pipeline = [
        dict(
            type='UntrimmedSampleFrames',
            clip_len=args.clip_len,
            frame_interval=args.frame_interval,
            start_index=1),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=256),
        dict(type='Normalize', **args.img_norm_cfg),
        dict(type='FormatShape', input_format=args.input_format),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    #data = open(args.data_list).readlines()
    
    # data = [x.strip() for x in data]
    #data =[os.path.splitext(os.path.basename(x.strip()))[0] for x in data]
    #data = data[args.part::args.total]
    # data = data[50760:]

    # enumerate Untrimmed videos, extract feature from each of them

    # 过滤出以 .mp4 结尾的文件

    stats = [-7.9842134, 3.9270191]
    #cfg = load_yaml_config('config.yaml')
    to_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        n_mels=64,
        f_min=60,
        f_max=7800,
    )
    normalizer = PrecomputedNorm(stats)


    #frame_dir = item.split('.')[0]
    
    #print(frame_dir)
    
    #input_mp4 = osp.basename(frame_dir) + '.mp4'
    #input_path = osp.join(args.data_prefix, input_mp4)
    input_mp4 = data_prefix
    input_path = input_mp4


    frame_dir = store_path
    name = osp.basename(input_mp4).split('.')[0]

    out_wav = name + '.wav'
    out_mp4 = name + '_25fps.mp4'
    out_mp4_path = osp.join(frame_dir, 'tmps', out_mp4)
    out_wav_path = osp.join(frame_dir, 'tmps', out_wav)
    out_frame_path = osp.join(frame_dir, 'rawframes', name)

    if not osp.isdir(out_frame_path):
        #print(f'Creating folder: {out_frame_path}')
        os.makedirs(out_frame_path)

    if not osp.isdir(osp.join(frame_dir, 'tmps')):
        os.makedirs(osp.join(frame_dir, 'tmps'))

    import subprocess
    ffprobe_cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'
    p = subprocess.Popen(
        ffprobe_cmd.format(input_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True)
    out, err = p.communicate()
    duration_info = float(str(out, 'utf-8').strip())
    start_time=str(int(start_rate*duration_info))
    duration_time=str(int(end_rate*duration_info-start_rate*duration_info))
    cmd1 = f'ffmpeg -y -loglevel quiet -i {input_path} -ss {start_time} -t {duration_time} -r 25 -ac 1 -ar 16000 {out_mp4_path} -map 0:a -ac 1 -ar 16000 {out_wav_path}'
    #print(cmd1)
    r1 = os.popen(cmd1)

    r1.close()


    shutil.rmtree(out_frame_path)

    vr = mmcv.VideoReader(out_mp4_path)

    for i, vr_frame in enumerate(vr):
        if vr_frame is not None:
            w, h, _ = np.shape(vr_frame)
            if args.new_short == 0:
                if args.new_width == 0 or args.new_height == 0:
                    # Keep original shape
                    out_img = vr_frame
                else:
                    out_img = mmcv.imresize(
                        vr_frame,
                        (args.new_width, args.new_height))
            else:
                if min(h, w) == h:
                    new_h = args.new_short
                    new_w = int((new_h / h) * w)
                else:
                    new_w = args.new_short
                    new_h = int((new_w / w) * h)
                out_img = mmcv.imresize(vr_frame, (new_h, new_w))
            mmcv.imwrite(out_img,
                         f'{out_frame_path}/img_{i + 1:05d}.jpg')

    #print('video_reader', len(video_reader))
    length = len(glob.glob(os.path.join(out_frame_path,'img_*.jpg' if args.is_rgb else 'flow_x_*.jpg')))
    #output_file = osp.join(args.output_prefix, output_file)
    # assert output_file.endswith('.pkl')
    #if  osp.exists(output_file):
        # print("skip")
    #    prog_bar.update()
    #    continue
    length = int(length)
    #print(length)
    # prepare a pseudo sample

    tmpl = dict(
        frame_dir=out_frame_path,
        total_frames=length,
        filename_tmpl=args.f_tmpl,
        start_index=1,
        modality=args.modality)

    sample = data_pipeline(tmpl)
    imgs = sample['imgs']
    shape = imgs.shape
    # the original shape should be N_seg * C * H * W, resize it to N_seg *
    # 1 * C * H * W so that the network return feature of each frame (No
    # score average among segments)
    imgs = imgs.reshape((shape[0], 1) + shape[1:])
    imgs = imgs.cuda()
    
    wav, sr = torchaudio.load(out_wav_path) # a sample from SPCV2 for now





    lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())

    #print(lms)
    #print(device)
    with torch.no_grad():       
        #video_feats = model_re.model_v.forward(video_deal.to(device), return_loss=False)
        #audio_feats = model_re.model_a(lms.unsqueeze(0).to(device))
        model_re.eval()
        # torch.onnx.export(model_re, (lms.unsqueeze(0).to(device), imgs), './onnx/onnx_model.onnx')
        out = model_re(lms.unsqueeze(0).to(device), imgs)
        out = torch.nn.functional.softmax(out)[0][1]
        print(out.item())
        #print(out)
        return out.item()


if __name__ == '__main__':
    inference('video/cabcdfd7108889050a36355e627bae0c.mp4',store_path='./store_folder')
