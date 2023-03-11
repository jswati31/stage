
# Spatial Attention and Gaussian Process for Video Gaze Estimation

This repository is the official implementation of [STAGE](). 

## Requirements
The code is tested with Python 3.7.10 and torch 1.18.1.

To install all the packages:

```setup
pip install -r requirements.txt
```


## Data Processing

1. Download [EVE Dataset](https://ait.ethz.ch/projects/2020/EVE/), [Gaze360](http://gaze360.csail.mit.edu/) and [EYEDIAP](https://www.idiap.ch/en/dataset/eyediap) datasets.
2. Preprocess Gaze360 and EYEDIAP using [GazeHub](https://phi-ai.buaa.edu.cn/Gazehub/).


## Training

For cross-dataset:
```
python main.py --config_json configs/<json_file> --save_path <path/to/save> --spatial_model <sam_variant>
```
For within-dataset:
```
python main_gaze360.py --config_json configs/<json_file> --save_path <path/to/save> --spatial_model <sam_variant>
```

## Pre-trained Models

You can download pretrained models here:

- [STAGE-Tx-ProposedSAM]()
- [STAGE-LSTM-ProposedSAM]()

## Questions?

For any inquiries, please contact us.
