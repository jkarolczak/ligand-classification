# ligands-classification

## Environement
### Docker
**Building Docker image from Dockerfile**
No additional configuration is required, simply execute in ligands-classification folder:
> `docker build . -t <name-of-image>`  
  
**Running existing Docker image**
The recommended way of running our Docker image would be:
> `sudo docker run --rm -it --init --gpus=all --ipc=host --volume="<path-to-folder>:/app" -w="/app" <name-of-image> /bin/bash`  

with <path-to-folder> pointing to the folder containing data folder and cloned ligands-classification repository

## Frameworks
* [MinkowskiEngine](https://nvidia.github.io/MinkowskiEngine/index.html) (recommended to use)
  * [example](https://github.com/NVIDIA/MinkowskiEngine/blob/432ce88ab1735b84a2d4fb21a5d45af98a968f1b/examples/classification_modelnet40.py) (classification using Sparse CNN)

* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [torchsparse](https://reposhub.com/python/deep-learning/mit-han-lab-torchsparse.html)

# Papers
* [Automatic recognition of ligands in electron density by machine learning](https://academic.oup.com/bioinformatics/article/35/3/452/5055122)
* [Recognizing and validating ligands with CheckMyBlob](https://academic.oup.com/nar/article/49/W1/W86/6255698)
