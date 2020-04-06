# EAD2020
The code accompanying our submission to the EndoCV 2020 challenge. Full paper [here](https://arxiv.org/abs/2003.10129v1).

### How to Train:
 1. **Downloading and Preparing Dataset** : Download script.sh file  and run `bash script.sh` on terminal. This will install dependencies, download dataset and extract them to following folders:
	 - EndoCV/EAD2020-Phase-II-Segmentation-VALIDATION
	 - EndoCV/EAD2020-Phase-II_HOLDOUT
	 - EndoCV/ead2020_semantic_segmentation_TRAIN

2. Clone this repository using command `git clone https://github.com/ubamba98/EAD2020.git`
3. Run Sample.ipynb notebook to start training. 
4. Trained model weights and logs will get saved in models and logs directory respectively.
