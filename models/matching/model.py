import torch 
from torch import nn 
import torchvision

from transformers import *
from .normalization import FrozenBatchNorm2d 

class VisualModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        
        try:
            model = getattr(torchvision.models, cfg.VISUAL_MODEL.NAME)(pretrained=True)
        except AttributeError as e:
            print(e)
            print("There is no model %s in torchvision." % cfg.VISUAL_MODEL.NAME)
        
        backbone_dim = model.fc.in_features
        model.fc = nn.Identity() 
        self.model = model 
        if not cfg.VISUAL_MODEL.FINE_TUNING:
            for param in model.parameters():
                param.requires_grad = False

        hdim = cfg.VISUAL_MODEL.HIDDEN_DIM
        fdim = cfg.CROSS_FEATURE_DIM
        dropout = cfg.VISUAL_MODEL.DROPOUT
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim, hdim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, fdim),
        )

    def forward(self, img):
        x = self.model(img)
        x = self.mlp(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x

class LangModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        arch = cfg.LANG_MODEL.NAME
        try:
            bert = AutoModel.from_pretrained(arch)
            self.tokenizer = AutoTokenizer.from_pretrained(arch)
        except AttributeError as e:
            print(e)
            print("There is no model %s in Transformer." % arch)
        bert.init_weights()

        if not cfg.LANG_MODEL.FINE_TUNING:
            for param in bert.parameters():
                param.requires_grad = False
                
        backbone_dim = bert.config.hidden_size
        self.backbone = bert

        hdim = cfg.LANG_MODEL.HIDDEN_DIM
        fdim = cfg.CROSS_FEATURE_DIM
        dropout = cfg.LANG_MODEL.DROPOUT
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim, hdim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, fdim),
        )

    def forward(self, captions, device = 0):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            x = self.backbone(**tokenized).last_hidden_state
            x = self.mlp(x)
            x = x / x.norm(2, dim=-1, keepdim=True)
            return x
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        
        

class JointModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.visual_model = VisualModel(cfg)
        self.lang_model = LangModel(cfg)
    
    def forward(self, v_in, l_in):
        self.eval()
        v_out = self.visual_model(v_in) 
        l_out = self.lang_model(l_in)
        return v_out, l_out 
