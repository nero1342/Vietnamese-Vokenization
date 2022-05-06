import json 
from pathlib import Path 

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset 
from PIL import Image 
import numpy as np 
import torch 
from torchvision import transforms

CC_ROOT = 'data/cc'
COCO_ROOT = 'data/mscoco'
VG_ROOT = '/ssd-playpen/data/vg'
LXRT_ROOT = 'data/lxmert-vi'


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
        self.img_transform = img_transform
        self.createIndex() 

    def createIndex(self):
        print("creating index...")
        metas = [] 
        self.data = [] 

        for img_split in self.img_splits:
            fname = Path(LXRT_ROOT) / (img_split + ".json")
            cur_meta = json.load((fname).open())
            print(f'Found {len(cur_meta)} images in {fname}')
            metas.extend(cur_meta)

        for i, meta in enumerate(metas):
            img_id = meta['img_id']
            for lang_split in self.lang_splits:
                if lang_split in meta['sentf']:
                    sents = meta['sentf'][lang_split]
                    for j, sent in enumerate(sents):
                        self.data.append(make_data(lang_split, img_id, j, sent))
        
        print(f"Found total {self.__len__()} image-caption pair in {LXRT_ROOT}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]
        uid, img_id, img_path, caption = meta['uid'], meta['img_id'], meta['img_path'], meta['sent'] 
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(e)
            print(img_path)
            return self.__getitem__((index + 1) % self.__len__())
        if self.img_transform is not None:
            img = self.img_transform(img) 

        img = np.array(img)

        return img, caption
    
    def get_meta(self, index):
        return self.data[index]
    
    def get_image(self, index):
        meta = self.data[index]
        uid, img_id, img_path, caption = meta['uid'], meta['img_id'], meta['img_path'], meta['sent'] 
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(e)
            print(img_path)
            return self.get_image((index + 1) % self.__len__())        
        return img 
class XMatchingDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        # self.save_hyperparameters()
        self.cfg = cfg
        self.train_transform = transforms.Compose(
            [
                # transforms.RandomAffine(degrees=45),
                # transforms.RandomAutocontrast(p=0.2),
                transforms.RandomResizedCrop((224, 224), scale = (0.5, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                
            ]
        )


    def train_dataloader(self):
        train_dataset = XMatching(cfg = self.cfg.DATASET.TRAIN, img_transform=self.train_transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE, num_workers=self.cfg.TRAIN.NUM_WORKERS, drop_last=True, shuffle = True)

    def val_dataloader(self):
        test_dataset = XMatching(cfg = self.cfg.DATASET.VAL, img_transform=self.val_transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.TRAIN.BATCH_SIZE, num_workers=self.cfg.TRAIN.NUM_WORKERS, drop_last=True, shuffle = False)