# mica-MovieCLIP
This repository contains the codebase for MovieCLIP: Visual Scene Recognition in Movies

## **Installation**

Install the environment for training the baseline LSTM models using the following commands:
```
conda create -n py37env python=3.7
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

Install [**CLIP**](https://github.com/openai/CLIP) dependencies using the following commands:
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```



If you find this repository useful, please cite the following paper:
```bibtex
@misc{MovieCLIP,
  doi = {10.48550/ARXIV.2210.11065},
  url = {https://arxiv.org/abs/2210.11065},
  author = {Bose, Digbalay and Hebbar, Rajat and Somandepalli, Krishna and Zhang, Haoyang and Cui, Yin and Cole-McLaughlin, Kree and Wang, Huisheng and Narayanan, Shrikanth},
  title = {MovieCLIP: Visual Scene Recognition in Movies},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
