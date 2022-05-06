import torch
from argparse import ArgumentParser
import torchvision.transforms as transforms 
from collections import OrderedDict
from tqdm import tqdm
import faiss 
import matplotlib.pyplot as plt 
import os 

from configs.config import Config
from datasets.matching import XMatching
from models.matching.model import JointModel

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        
    ]
)

def load_model(model, weight_path):
    state_dict = torch.load(weight_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[len("model."):] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def main(): 
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--weight", type=str, default="final.ckpt")
    parser.add_argument("--text", type=str, default="Xin_chào")
    args = parser.parse_args()
    cfg_path = args.config
    cfg = Config(cfg_path)
    
    # Load
    dataset = test_dataset = XMatching(cfg = cfg.DATASET.VAL, img_transform=transform)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=False, shuffle = False)

    weight_path = args.weight
    model = JointModel(cfg.MODEL).cuda()
    model.eval()
    model = load_model(model, weight_path)
    
    # Extract image features
    meta = dataset.data 
    voken = None
    for data in tqdm(dataloader):
        vis, lan = data
        with torch.no_grad():
            img_feats = model.visual_model(vis.cuda())
            if voken is None:
                voken = img_feats
            else:
                voken = torch.cat((voken, img_feats), dim = 0)
    
    # Indexing
    d = cfg.MODEL.CROSS_FEATURE_DIM
    xb = voken.detach().cpu().numpy()
    index = faiss.IndexFlatL2(d)   # build the index
    index.add(xb)                  # add vectors to the index

    # Visualize vokens
    text = args.text
    tokens = model.lang_model.tokenizer.tokenize(text)

    xq = model.lang_model([text])[0][1:-1].detach().cpu().numpy() 
    D, I = index.search(xq, 1) 

    
    os.makedirs('./visualization/' + text, exist_ok = True)
    for i, token in enumerate(tokens):
        if token != '.' and token != ',':
            img = dataset.get_image(I[i][0]).resize((224, 224)) 
            img.save(f'./visualization/{text}/{i:02d}_{token}.jpg')
            plt.imshow(img)
            plt.show() 
            print(token, D[i]) 