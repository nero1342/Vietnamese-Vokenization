mkdir -p data/mscoco
wget http://images.cocodataset.org/zips/train2014.zip -P data/mscoco
wget http://images.cocodataset.org/zips/val2014.zip -P data/mscoco
unzip data/mscoco/train2014.zip -d data/mscoco/images/ && rm data/mscoco/train2014.zip
unzip data/mscoco/val2014.zip -d data/mscoco/images/ && rm data/mscoco/val2014.zip
gdown --id 17BcPlyMXKkj0vz4cxK5TonjPwgSVsMwP
unzip -q lxmert-vi.zip -d data/
