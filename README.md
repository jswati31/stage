
# Spatial Attention and Gaussian Processes for Video Gaze Estimation

This repository is the official implementation of [STAGE](https://arxiv.org/abs/2404.05215). 

## Requirements
The code is tested with Python 3.7.10 and torch 1.12.1.

To install all the packages:

```setup
pip install -r requirements.txt
```


## Data Processing

1. Download [EVE Dataset](https://ait.ethz.ch/projects/2020/EVE/), [Gaze360](http://gaze360.csail.mit.edu/) and [EYEDIAP](https://www.idiap.ch/en/dataset/eyediap) datasets.
2. Preprocess Gaze360 and EYEDIAP using [GazeHub](https://phi-ai.buaa.edu.cn/Gazehub/).


## Training STAGE

* Download [GazeCLR](https://drive.google.com/file/d/10K_AwVH6H_0P77lR0XHl3iDsfiep2YTP/view) weights and store in "gazeclr_weights" folder.

For cross-dataset:
```
python main.py --config_json configs/<json_file> --save_path <path/to/save> --spatial_model <sam_variant>
```
For within-dataset:
```
python main_gaze360.py --config_json configs/<json_file> --save_path <path/to/save> --spatial_model <sam_variant>
```

## GP Personalization

Train GP Base Model on EVE:

```
python train_gp_basemodel.py --config_json configs/<json_file> --spatial_model <sam_variant> --load_checkpoint_path <STAGE_MODEL_PATH> --gp_model_name <GP_BASE_MODEL_NAME>
```

Adpating Base Model on EYEDIAP Participants:
```
python gp_personalization.py --config_json configs/<json_file> --k <number_of_shots> --spatial_model <sam_variant> --load_checkpoint_path <STAGE_MODEL_PATH> --gp_model_path <GP_BASE_MODEL_PATH>
```

## Pre-trained Models

You can download pretrained models [here](https://drive.google.com/drive/folders/1kV3K6OMwgddxKNbdHmRnG3ytAhAVvumT?usp=share_link).

## Questions?

For any inquiries, please contact us.
