# Vietnamese-Vokenization
Implement Vietnamese Vokenization based on paper ["Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision"](https://arxiv.org/pdf/2010.06775.pdf) (Hao Tan and Mohit Bansal)

Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zoJ2Lc87kmqcNWkuC5qrMGq0LGZ6UyOZ?usp=sharing)

# Outline

# Installation
```
pip install -r requirements.txt
```
Require python 3.6 + (to support huggingface transformers).

# Contextualized Cross-Modal Matching (xmatching)

## Prepare datasets
```
sh prepare_datasets.sh 
```
or 
1. Download MS COCO images:
    ```
    # MS COCO (Train 13G, Valid 6G)
    mkdir -p data/mscoco
    wget http://images.cocodataset.org/zips/train2014.zip -P data/mscoco
    wget http://images.cocodataset.org/zips/val2014.zip -P data/mscoco
    unzip data/mscoco/train2014.zip -d data/mscoco/images/ && rm data/mscoco/train2014.zip
    unzip data/mscoco/val2014.zip -d data/mscoco/images/ && rm data/mscoco/val2014.zip
    ```
2. Download Vietnamese captions (split following the LXMERT project after [EN-VI translation](https://huggingface.co/NlpHUST/t5-en-vi-small) and [word-segmentation](https://github.com/VinAIResearch/PhoNLP#example-usage)), [drive link](https://drive.google.com/file/d/17BcPlyMXKkj0vz4cxK5TonjPwgSVsMwP/view):
    ```
    gdown --id 17BcPlyMXKkj0vz4cxK5TonjPwgSVsMwP
    unzip -q lxmert-vi.zip -d data/
    ```
## Training the Cross-Modal Matching Model
The model is trained on MS COCO with pairwise hinge loss
```
python train_net.py --config configs/default.yaml
```

## Pretrained [link](https://drive.google.com/file/d/1_x2JnOQub0fN1O1iEG4j8U2NiCwnUG4c/view?usp=sharing)
```
gdown --id 1_x2JnOQub0fN1O1iEG4j8U2NiCwnUG4c 
```
## Demo
In order to vokenize text ```"Xin_chào các bạn"```, we can run the following command:
```
python demo.py --config configs/default.yaml --weight <weight_path> \
            --text "Xin_chào các bạn"
```

# Task
&#9745; Contextualized Cross-Modal Matching

&#9744; Revokenization

&#9744; Downstream task