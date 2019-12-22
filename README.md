# Single Shot Multibox Detector - Facade Parsing
Semester Project at Swiss Data Science Center @EPFL

## Description
This project is a collaboration between the Civil Engineers and the Swiss Data Science Center(SDSC) at EPFL. The motivation behind the project is to help Civil Engineers in detecting the damage imposed on buildings by an earthquake. In this semester project, only a sub-part of the whole project has been tackled. By using deep learning, we automate the detection of important objects in an given image. As a deep learning method we are using the Single Shot MultiBox Detector. 

Find more details about the project in the [report.pdf](report.pdf).


## Install
To install the `ssd_project` library and use it within your python environment:
```
pip install ssd-project==0.1
```


## Examples on how to use 
Examples for various functions and sub-tasks of the project can be found in the Notebooks folder.

### Training
To train, the `train.py` script is ready to use:
```
usage: train.py [-h] [--epochs EPOCHS] [--split-seed SPLIT_RATIO]
                [--batch-train BATCH_SIZE] [--lr LR]
                [--model PRETRAINED_MODEL] [--path_imgs PATH_IMGS]
                [--path_bboxes PATH_BBOXES] [--path_labels PATH_LABELS]

Train an SSD model

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --split-seed SPLIT_RATIO
  --batch-train BATCH_SIZE
  --lr LR
  --model PRETRAINED_MODEL
  --path_imgs PATH_IMGS
  --path_bboxes PATH_BBOXES
  --path_labels PATH_LABELS
```
For example:
```
python3 train.py --epochs=500
python train.py --model saved_models/BEST_model_ssd300.pth.tar
```
For loading a pretrained a model python train.py --model saved_models/BEST_model_ssd300.pth.tar

When training a new model, the train.py script saves a specific structure, therefore to continue training from the best specific point, just give as input to the saved structure.  

### Predicting
For predictions, one should look into `notebooks/results.ipynb`

### Ground Truth & Aspect Ratios 
`notebooks/creation_ground_truth.ipynb` shows how the ground truth was derived.
`notebooks/Aspect_Ratios.ipynb` shows how aspect ratios and scales for prior box are derived

### Transofrmations
`notebooks/Data Augmentations Examples.ipynb` shows an example of all the transformations applied to one image.

 
## Outputs
The best model is saved in `saved_model/first_best_model_ssd300.pth.tar`


