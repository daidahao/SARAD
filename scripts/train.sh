# datasets
python src/train.py data=psm  trainer=gpu model.optimizer.lr=0.0003 model.net.num_layers=5
python src/train.py data=smd  trainer=gpu
python src/train.py data=swat trainer=gpu model.optimizer.lr=0.0001
python src/train.py data=hai  trainer=gpu model.optimizer.lr=0.0002 model.net.num_layers=5 model.detec_weight=0.8
