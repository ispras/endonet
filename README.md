# EndoNet: model for automatic calculation histoscore on histological slides

This project hosts the code for the pretraining section of article **EndoNet: model for automatic calculation histoscore on histological slides**

Model weights can be downloaded from https://nextcloud.ispras.ru/index.php/s/Srgi6nsrXgQd7tQ

After downloading models should be placed into
supplementary/pretrained_checkpoint.pth
supplementary/baseline_checkpoint.pth

## Info
Supplementary folder contains testing dataset, corresponding YAML files, baseline and pretrained models checkpoints and configs for them. Correctness of scripts inside this project depends on a supplementary folder. Please, do not edit any content inside. 

## Requirements
Python 3.8 or Python 3.9

## Installation
a. Create virtual environment

```python3 -m venv pretraining_endonet```

b. Activate this virtual environment

```source pretraining_endonet/bin/activate```

c. Navigate to root folder of this project

```cd article_demo```

d. Install requirements to venv

```pip install -r requirements.txt```

e. (optional) Requires GPU that supports CUDA and installation process heavily depends on your hardware and OS. Providing links for each operational system family:

[Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

[Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)

[Mac(deprecated)](https://developer.nvidia.com/nvidia-cuda-toolkit-11_6_0-developer-tools-mac-hosts)

## Usage
### Calculating metrics and confidence intervals
Please, note that method for confidence intervals is not deterministic and results could slightly change with each launch.

To calculate metrics and confidence intervals, use the following commands:

a. Activate virtual environment

```source <path to virtual environment>/bin/activate```

b. Navigate to project's path

```cd <path to project>```

c. Run python script itself

```python3 <path to this project>/demo.py```

d. Wait until it finishes computations. It could take a few minutes, because we are calculating metrics across the whole testing dataset and a lot of confidence intervals with a lot of resamples (10000 x 10000)

e. It will print out metrics for each model. Average precision for stroma and epithelium and mean Average Precision for both classes (mAP). Besides that, it will print left and right border for 

### Inference model to get heatmaps and keypoints
To inference model, the first thing that we need is an image. You could either take one from testing dataset, or get another, somewhere. 
![Inference signature](supplementary/inference_signature.png?raw=true)
Also, please note, that here we use inline arguments, to specify image path (and, if we need, we could specify model_path and config path, with adding ```--model_path=<new_path_to_model>``` or ```--config_path=<new_path_to_config>```. By default, that arguments are set to path with config and model inside supplementary directory) 

Then, run commands:

a. Activate virtual environment 

```source <path to virtual environment>/bin/activate```

b. Navigate to project's path

```cd <path to project>```

c. Run python script with arguments

```python3 <path to this project>/inference.py --image_path="<path to image>"```

d. Inside root folder you will find heatmaps_epithelium.png, heatmaps_stroma.png and output.txt. First two files are corresponding to visualised heatmaps for each class, and third is every keypoint that model has detected. 

#### Folder structure
At last, the folder structure looks like this:

```
├── heatmaps_epithelium.png (appears only after executing inference.py and contains heatmaps for epithelium)
├── heatmaps_stroma.png (appears only after executing inference.py and contains heatmaps for stroma)
├── output.txt (appears only after executing inference.py and contains keypoints)
├── demo.py
├── dependencies.py
├── inference.py
├── README.md
├── requirements.txt
└── supplementary
    ├── baseline_cfg.json
    ├── baseline_checkpoint.pth
    ├── dataset
    │   ├── endonuke_data
    │   │   ├── images (Contains images)
    │   │   ├── images.txt
    │   │   ├── labels (Contains labels)
    │   │   └── labels.txt
    │   └── staining_ds
    │       ├── images (Contains images)
    │       ├── images.txt
    │       ├── labels (Contains labels)
    │       └── labels.txt
    ├── pretrained_checkpoint.pth
    ├── pretrained_config.json
    └── test.yaml
```
