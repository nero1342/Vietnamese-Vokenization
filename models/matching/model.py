import torch 
from torch import nn 
from torchvision import models 

from transformers import *
from .normalization import FrozenBatchNorm2d 

LANG_MODELS = {
          'bert':    (BertModel,       BertTokenizer,       'bert-base-uncased'),
          'bert-large':  (BertModel,       BertTokenizer,       'bert-large-uncased'),
          'gpt':     (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          'gpt2':    (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          'ctrl':    (CTRLModel,       CTRLTokenizer,       'ctrl'),
          'xl':      (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          'xlnet':   (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          'xlm':     (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          'distil':  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
          'roberta': (RobertaModel,    RobertaTokenizer,    'roberta-base'),
          'xlm-roberta': (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
}

class VisualModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        
        try:
            model = getattr(models, cfg.VISUAL_MODEL.NAME)(
                replace_stride_with_dilation=[False, False, False],
                pretrained=True, 
                norm_layer=FrozenBatchNorm2d)

        except AttributeError as e:
            print(e)
            print("There is no model %s in torchvision." % cfg.VISUAL_MODEL.NAME)
        
        
        backbone_dim = model.fc.in_features 
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
        x = self.mlp(x)         # [b, dim]
        return x

class LangModel(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        arch = cfg.LANG_MODEL.NAME
        Model, Tokenizer, weight = LANG_MODELS[arch]
        bert = Model.from_pretrained(
            weight,
            output_hidden_states=True
        )
        bert.init_weights()

        if not cfg.VISUAL_MODEL.FINE_TUNING:
            for param in bert.parameters():
                param.requires_grad = False
                
        backbone_dim = bert.config.hidden_size
        self.backbone = bert

        layers = cfg.LANG_MODEL.LAYERS
        self.layers = sorted(layers)

        hdim = cfg.LANG_MODEL.HIDDEN_DIM
        fdim = cfg.CROSS_FEATURE_DIM
        dropout = cfg.LANG_MODEL.DROPOUT
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim * len(self.layers), hdim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, fdim),
        )

    def forward(self, captions, device = 0):
        if isinstance(captions[0], str):
            tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(device)
            x = self.text_encoder(**tokenized)
            # sequence_output, pooled_output, (hidden_states), (attentions) --> seq_output
            if type(self.backbone) is XLNetModel:
                output, hidden_states = x[:2]
            else:
                output, pooled_output, hidden_states = x[:3]

            # gather the layers
            if type(self.backbone) is XLNetModel:
                x = torch.cat(list(hidden_states[layer].permute(1, 0, 2) for layer in self.layers), -1)
            else:
                x = torch.cat(list(hidden_states[layer] for layer in self.layers), -1)

            x = self.mlp(x)
            return x
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        
        

class JointModel(nn.Module):
    def __init__(self, visual_model, lang_model):
        super().__init__() 
        self.visual_model = visual_model
        self.lang_model = lang_model 
    
    def forward(self, v_in, l_in):
        v_out = self.visual_model(v_in) 
        l_out = self.lang_model(l_in)
        return v_out, l_out 
