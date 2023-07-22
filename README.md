# DiffMIC

DiffMIC is a project to adapt [Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) to general medical image classification by dual-granularity conditional guidance.
The method is elaborated in the paper [DiffMIC: Dual-Guidance Diffusion Network for Medical Image Classification](https://arxiv.org/abs/2303.10610).

## A Quick Overview 

<img width="800" height="400" src="https://github.com/scott-yjyang/DiffMIC/blob/main/figs/framework.png">

## News
- 23-06-05. This paper has been early accepted by MICCAI 2023. Code is coming and welcome to taste it.
- 23-06-06. This project is still quickly updating 🌝. Check TODO list to see what will be released next.

## Requirement

``conda env create -f environment.yml``

## Datasets

1. Download [HAM10000](https://challenge.isic-archive.com/data/#2018) or [APTOS2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) dataset. Your dataset folder under "your_data_path" should be like:

dataset/isic2018/

     images/...
     
     ISIC2018_Task3_Training_GroundTruth.csv
     
     isic2018_train.pkl

     isic2018_test.pkl

dataset/aptos/

     train/...
     
     train.csv
     
     aptos_train.pkl

     aptos_test.pkl

.pkl file contains the list of data whose element is a dictionary with the format as ``{'img_root':image_path,'label':label}``

## Run

2. For Training! run: ``bash training_scripts/run_isic.sh`` where the first command line is used ``python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config configs/${TASK}.yml --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS}``

3. For Testing! run: ``bash training_scripts/run_isic.sh`` where the second command line is used ``python main.py --device ${DEVICE_ID} --thread ${N_THREADS} --loss ${LOSS} --config $EXP_DIR/${MODEL_VERSION_DIR}/logs/ --exp $EXP_DIR/${MODEL_VERSION_DIR} --doc ${TASK} --n_splits ${N_SPLITS} --test --eval_best``

The configuration for each of the above-listed tasks (including data file location, training log and evaluation result directory settings, neural network architecture, optimization hyperparameters, etc.) are provided in the corresponding files in the ``configs`` directory


### TODO LIST

- [ ] Release PMG2000 dataset and config
- [x] Release HAM10000, APTOS2019 dataloaders and configs
- [x] Dataset splits
- [x] Release training scripts
- [x] Release evaluation
- [ ] Upload the checkpoints of HAM10000, APTOS2019
- [ ] configuration


## Be a part of DiffMIC !
Welcome to contribute to DiffMIC. Any technique that can improve the performance or speed up the algorithm is appreciated🙏. I am writing DiffMIC V2, aiming at top journals. I'm glad to list the contributors as my co-authors🤗.


## Thanks

Code is largely based on [XzwHan/CARD](https://github.com/XzwHan/CARD), [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion), [MedSegDiff](https://github.com/WuJunde/MedSegDiff/tree/master), [nyukat/GMIC](https://github.com/nyukat/GMIC)


## Cite
If you find this code useful, please cite
~~~
@article{yang2023diffmic,
  title={DiffMIC: Dual-Guidance Diffusion Network for Medical Image Classification},
  author={Yang, Yijun and Fu, Huazhu and Aviles-Rivero, Angelica and Sch{\"o}nlieb, Carola-Bibiane and Zhu, Lei},
  journal={arXiv preprint arXiv:2303.10610},
  year={2023}
}
~~~



