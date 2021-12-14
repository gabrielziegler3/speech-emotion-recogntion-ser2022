#!/bin/bash

pip install gdown


mkdir -p data/
cd data/

echo "Downloading Training Dataset Files..."
gdown --id 1N56YOgJ_plF4K8Eyh9hqiP0_O5L8uwya

echo "Unziping Training Dataset Files..."
unzip data_train.zip

