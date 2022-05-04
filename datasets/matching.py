import json 
from pathlib import Path 

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset 
from PIL import Image 
import numpy as np 
import torch 

CC_ROOT = 'data/cc'
COCO_ROOT = 'data/mscoco'
VG_ROOT = '/ssd-playpen/data/vg'
LXRT_ROOT = 'data/lxmert'


def make_uid(img_id, source, sent_id):
    """
    see the descriptions in function 'make_datum'
    """
    return "%s:%s:%s" % (img_id, source, sent_id)


def get_img_path(source, img_id):
    if source == 'cc':
        split_tag, _ = img_id.split('_')
        return "%s/images/%s/%s" % (CC_ROOT, split_tag, img_id)
    elif 'COCO' in img_id:
        _, split_tag, _ = img_id.split('_')
        return "%s/images/%s/%s" % (COCO_ROOT, split_tag, img_id + '.jpg')
    else:   # VG images
        return "%s/images/%s.jpg" % (VG_ROOT, img_id)

def make_data(source: str, img_id: str, sent_id: int, sent: str):
    uid = make_uid(img_id, source, sent_id)
    img_path = get_img_path(source, img_id)
    return {
        'uid': uid,
        'img_id': img_id,
        'img_path': img_path,
        'sent': sent,
    }

class XMatching(Dataset):
    def __init__(self, cfg, img_transform):
        pass 
        self.img_splits = cfg.IMG_SPLITS
        self.lang_splits = cfg.LANG_SPLITS

    def createIndex(self):
        metas = [] 
        self.data = [] 

        for img_split in self.img_splits:
            fname = img_split + ".json"
            metas.extend(json.load((Path(LXRT_ROOT) / fname).open()))

        for i, meta in enumerate(metas):
            img_id = meta['img_id']
            for lang_split in self.lang_splits:
                if lang_split in meta['sentf']:
                    sents = meta['sentf'][lang_split]
                    for j, sent in enumerate(sents):
                        self.data.append(make_data(lang_split, img_id, j, sent))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]
        uid, img_id, img_path, caption = meta['uid'], meta['img_id'], meta['img_path'], meta['sent'] 

        img = Image.open(img_path)
        if self.img_transform is not None:
            img = self.img_transform(img) 

        img = np.array(img)

        return img, caption

class XMatchingDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        train_dataset = XMatching(cfg = self.cfg.DATASET.TRAIN, transform=self.transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE)

    def val_dataloader(self):
        test_dataset = XMatching(cfg = self.cfg.DATASET.VAL, transform=self.transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE)