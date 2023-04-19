# mica-MovieCLIP
This repository contains the codebase for MovieCLIP: Visual Scene Recognition in Movies

## **Installation**

* Install the environment for training the baseline LSTM models using the following commands:
  ```
  conda create -n py37env python=3.7
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
  pip install -r requirements.txt --use-deprecated=legacy-resolver
  ```
* Install [**CLIP**](https://github.com/openai/CLIP) dependencies using the following commands:

  ```
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git
  ```

## **Data setup**

* Please refer to [**README.md**](https://github.com/usc-sail/mica-MovieCLIP/blob/main/split_files/README.md) under the ```data_splits``` folder for instructions on using the MovieCLIP dataset.

## **Visual scene tagging**

* Please refer to [**README.md**](https://github.com/usc-sail/mica-MovieCLIP/blob/main/preprocess_scripts/visual_scene_tagging/README.md) under the ```preprocess_scripts/visual_scene_tagging``` folder for instructions on using the CLIP model for tagging the visual scenes in the MovieCLIP dataset.

## **To Dos**

- [x] Add the dataset link and instructions for using the MovieCLIP dataset
- [x] Add code for tagging using the CLIP model
- [ ] Add code for training the baseline LSTM models
- [ ] Add code for openmmlab setup and Swin-B model inference


If you find this repository useful, please cite the following paper:
```bibtex
@InProceedings{Bose_2023_WACV,
    author    = {Bose, Digbalay and Hebbar, Rajat and Somandepalli, Krishna and Zhang, Haoyang and Cui, Yin and Cole-McLaughlin, Kree and Wang, Huisheng and Narayanan, Shrikanth},
    title     = {MovieCLIP: Visual Scene Recognition in Movies},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {2083-2092}
}
```

For any questions, please open an issue and feel free to contact Digbalay Bose (dbose@usc.edu)
