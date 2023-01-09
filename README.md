# Semantic Segmentation on CityScapes dataset using UNet model

# Usage

This section of the README walks through how to train, evaluate and predict from a model.

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `SemanticSegmentation` python package that the scripts depend on.

## Preparing Data

Download Cityscapes dataset from [link](https://www.cityscapes-dataset.com/downloads/)

Save it in the folder [dataset](dataset) in the structure:
```
.{DATASET}
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
```

## Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, training flags and data setting. Here are some reasonable defaults:

```
MODEL_FLAGS="--model 'unet' --num_classes 19 --num_filters 64 --num_channels 3 --bilinear True"
TRAIN_FLAGS="--epochs 50 --loss_type 'dice' --lr 0.005 --batch_size 4 --resume_step 0 --chkptfolder './chpkt/' --save_interval 5"
DATA_FLAGS="--data_dir 'path/to/dataset' --augment False"
```

Once you have setup your hyper-parameters, you can run an experiment like so:

```
python scripts/train.py $MODEL_FLAGS $TRAIN_FLAGS $DATA_FLAGS
```

## Evaluating

The above training script saves checkpoints to `.pt` files in the `--chkptfolder` directory. These checkpoints will have names like `model_epoch_50.pt` and `optim_epoch_50.pt`.

Once you have a path to your model, you can evaluate IoU, Dice score,... of the `val` data
```
MODEL_FLAGS="--model 'unet' --num_classes 19 --num_filters 64 --num_channels 3 --bilinear True --model_path './chkpt/model_epoch_50.pt'"
DATA_FLAGS="--data_dir 'path/to/dataset' --batch_size 4"
```

```
python scripts/evaluate.py $MODEL_FLAGS $DATA_FLAGS
```

## Predicting

You can segment the `val` data and save the result to `--result_folder`
```
MODEL_FLAGS="--model 'unet' --num_classes 19 --num_filters 64 --num_channels 3 --bilinear True --model_path './chkpt/model_epoch_50.pt'"
DATA_FLAGS="--data_dir 'path/to/dataset' --batch_size 4"
```

```
python scripts/predict.py --result_folder 'path/to/result_images' $MODEL_FLAGS $DATA_FLAGS
```