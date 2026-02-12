import torch
import torch.nn as nn
from modules.coattention import co_attention

class FMNVD(torch.nn.Module):
    def __init__(self):
        super(FMNVD, self).__init__()

        self.text_dim = 768
        self.img_dim = 512
        self.motion_dim = 2048
        self.clip_dim = 512
        self.num_frames = 55
        self.speech_text_length = 64
        self.title_length = 32
        self.dim = 128
        self.dropout = 0.1
        self.num_heads = 4

        self.co_attention_ts = co_attention(
            d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
            d_model=self.dim,
            visual_len=self.speech_text_length, sen_len=self.title_length, 
            fea_v=self.dim, fea_s=self.dim, pos=False
        )
        
        self.co_attention_tv = co_attention(
            d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
            d_model=self.dim,
            visual_len=self.num_frames, sen_len=self.title_length, 
            fea_v=self.dim, fea_s=self.dim, pos=False
        )

        self.trm = nn.TransformerEncoderLayer(d_model=self.dim, nhead=2, batch_first=True)

        self.linear_title = nn.Sequential(
            torch.nn.Linear(self.text_dim, self.dim), 
            torch.nn.ReLU(), 
            nn.Dropout(p=0.1)
        )
        
        self.linear_speech = nn.Sequential(
            torch.nn.Linear(self.text_dim, self.dim), 
            torch.nn.ReLU(),
            nn.Dropout(p=self.dropout)
        )
        
        self.linear_motion = nn.Sequential(
            torch.nn.Linear(self.motion_dim, self.dim), 
            torch.nn.ReLU(), 
            nn.Dropout(p=0.1)
        )
        
        self.linear_clip = nn.Sequential(
            torch.nn.Linear(self.clip_dim, self.dim), 
            torch.nn.ReLU(), 
            nn.Dropout(p=0.1)
        )
        
        self.linear_attn = nn.Sequential(
            torch.nn.Linear(self.dim * 2, 1),
            torch.nn.Sigmoid()
        )

        self.classifier = nn.Linear(self.dim, 2)

    def forward(self, **kwargs):
        title = kwargs['title']
        fea_title = self.linear_title(title)

        audio_transcript = kwargs['audio_transcript']
        fea_speech = self.linear_speech(audio_transcript)
        
        motion = kwargs['motion']
        fea_motion = self.linear_motion(motion)
        
        clip = kwargs['frames']
        fea_clip = self.linear_clip(clip)

        fea_motion_mean = torch.mean(fea_motion, dim=1, keepdim=True) 
        fea_clip_mean = torch.mean(fea_clip, dim=1, keepdim=True) 
        
        combined = torch.cat([fea_motion_mean, fea_clip_mean], dim=-1)
        motion_weight = self.linear_attn(combined)
        
        fea_visual = motion_weight * fea_motion_mean + (1 - motion_weight) * fea_clip_mean

        fea_speech, fea_title = self.co_attention_ts(
            v=fea_speech, s=fea_title, 
            v_len=fea_speech.shape[1], s_len=fea_title.shape[1]
        )
        
        fea_visual, fea_title = self.co_attention_tv(
            v=fea_visual, s=fea_title, 
            v_len=fea_visual.shape[1], s_len=fea_title.shape[1]
        )

        fea_visual = torch.mean(fea_visual, -2)
        fea_title = torch.mean(fea_title, -2)
        fea_speech = torch.mean(fea_speech, -2)

        fea_title = fea_title.unsqueeze(1)
        fea_visual = fea_visual.unsqueeze(1)
        fea_speech = fea_speech.unsqueeze(1)

        fea = torch.cat((fea_title, fea_speech, fea_visual), 1)
        
        fea = self.trm(fea)
        fea = torch.mean(fea, -2)
        output = self.classifier(fea)

        return output