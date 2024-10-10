from typing import Dict, Optional, Union, Sequence, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.ffv_pt import Metadata
from model.blocks import Contraction2

from model.attentionLayer import attentionLayer, attentionLayer_mask
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.                                                                                      
    
    See description of make_non_pad_mask.                                                                                                       
    
    Args:
        lengths (torch.Tensor): Batch of lengths (B,).                                                                                          
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.                                                                            
    
    Examples:                  
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask

class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''          
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear = nn.Linear(d_input, output_size)
        
    def forward(self, x, n_wins):
        #print(x.shape[1])       
        #mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = torch.arange(x.shape[1]).unsqueeze(0) < n_wins.unsqueeze(1).to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), n_wins.unsqueeze(1))   
            
        x = self.linear(x)
        
        return x

class Deepfakecla_pt_mask_bgru(LightningModule):
    def __init__(self,
        v_encoder: str = "talk", a_encoder: str = "se", frame_classifier: str = "lr",
        ve_features=(64, 96, 128, 128), ae_features=(32, 64, 64), v_cla_feature_in=256, a_cla_feature_in=256,
        boundary_features=(512, 128), boundary_samples=10, temporal_dim=512, max_duration=40,
        weight_frame_loss=2., weight_modal_bm_loss=1., weight_contrastive_loss=0.1, contrast_loss_margin=0.99,
        weight_decay=0.0001, learning_rate=0.0002, distributed=False
    ):
        super().__init__()
        #self.save_hyperparameters()

        if v_encoder == "c3d":
            self.video_encoder = C3DVideoEncoder(n_features=ve_features)
        elif v_encoder == "talk":
            self.video_encoder = visualtalenet()
        elif v_encoder == "light":
            self.video_encoder = visual_encoder()
        elif v_encoder == "pre":
            #self.video_encoder = Conv1d_block(4098, 256)
            self.video_encoder = Contraction2(2048, 256, 128, 0)
            #self.video_encoder = Res1dNet12()
            
        if a_encoder == "cnn":
            self.audio_encoder = CNNAudioEncoder(n_features=ae_features)
        elif a_encoder == "se":
            self.audio_encoder = seaudioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
        elif a_encoder == "pre":
            self.audio_encoder = Contraction2(2048, 256, 128, 0)
            #self.audio_encoder = Res1dNet12()
            
        #self.fu_classifier = nn.Linear(256, 1)
        assert v_cla_feature_in == a_cla_feature_in
        #self.frame_loss = BCEWithLogitsLoss()
        self.frame_loss = CrossEntropyLoss()

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed

        self.crossA2V = attentionLayer_mask(d_model = 256, nhead = 8, positional_emb_flag=False, dropout=0.1)
        self.crossV2A = attentionLayer_mask(d_model = 256, nhead = 8, positional_emb_flag=False, dropout=0.1)
        self.selfAV = attentionLayer_mask(d_model = 512, nhead = 8, positional_emb_flag=False, dropout=0.1)

        self.pool = PoolAvg(512, 2)
        #self.pool = PoolAvg(512, 1)
        #self.pool = PoolAtt(512, 2)
        #self.pool = PoolAvgmax(512, 2)
        #self.pool = PoolMax(512, 2) 
        #self.pool = PoolAttFF(512, 256, 2) 
        #self.pool = Poolsta(512, 512, 2) 
        #self.pool = PoolLastStepBi(512, 2) 
        #self.GRU = BGRU(512)
        #self.lstm = nn.LSTM(
        self.lstm = nn.GRU(
                input_size = 512,
                hidden_size = 256,
                num_layers = 1,
                dropout = 0.1,
                batch_first = True,
                bidirectional = True
                ) 


    def forward(self, video: Tensor, audio: Tensor, n_frames):
        # encoders
        #print("video", video.shape)
        #print("audio", audio.shape)
        if self.training:
            mask = make_pad_mask(n_frames)
        else:
            if n_frames.max().item() < 500:
                max_len=500
            else:
                max_len= n_frames.max().item()
            mask = make_pad_mask(n_frames, max_len)
        #print(max_len)
        #print(mask)
        mask = mask.unsqueeze(1).bool()

        v_features,_ = self.video_encoder(video, mask)
        #a_features = self.audio_encoder(audio)
        a_features,_ = self.audio_encoder(audio, mask)
        #print(v_features)
        #print(a_features)
        v_features = v_features.transpose(1,2)
        a_features = a_features.transpose(1,2)

        #fu_features = torch.cat((a_features_c, v_features_c), 2) 
        mask = mask.squeeze(1)

        a_features_c = self.crossA2V(src = a_features, tar = v_features, mask=mask)
        v_features_c = self.crossV2A(src = v_features, tar = a_features, mask=mask)

        fu_features = torch.cat((a_features_c, v_features_c), 2) 
        fu_features = self.selfAV(src = fu_features, tar = fu_features, mask=mask) 
        #print(fu_features)

        fu_features = pack_padded_sequence(
                fu_features,
                n_frames.cpu(),
                batch_first=True,
                enforce_sorted=False
                )             
        
        fu_features = self.lstm(fu_features)[0]
        
        fu_features, _ = pad_packed_sequence(
            fu_features, 
            batch_first=True, 
            padding_value=0.0,
            total_length=n_frames.max())  
        #print(fu_features)

        #fu_features = self.GRU(fu_features)

        #fu_features = fu_features.transpose(1, 2)
        #fu_cla = self.fu_classifier(fu_features)
        fu_cla = self.pool(fu_features, n_frames)
        
        return fu_cla 


    def loss_fn(self, fu_label: Tensor, fu_cla) -> Dict[str, Tensor]:

        #fu_frame_loss = self.frame_loss(fu_cla.squeeze(1), fu_label.float())
        fu_frame_loss = self.frame_loss(fu_cla, fu_label)

        loss = fu_frame_loss

        return {
            "loss": loss 
        }

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None
    ) -> Tensor:
        video, audio, n_frames, fu_label, _ = batch

        fu_cla = self(video, audio, n_frames)
        loss_dict = self.loss_fn(fu_label, fu_cla)

        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        video, audio, n_frames, fu_label, _ = batch
        #print(fu_label)
        fu_cla = self(video, audio, n_frames)
        loss_dict = self.loss_fn(fu_label, fu_cla)

        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)
        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }

    @staticmethod
    def get_meta_attr(meta: Metadata, video: Tensor, audio: Tensor):

        fu_label = 1.0 if meta.modify_audio or meta.modify_video else 0.0
        
        return [meta.video_frames, fu_label]

