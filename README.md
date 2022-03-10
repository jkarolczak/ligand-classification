# Ligand classification using deep neural networks

## Environement

### Docker

**Building Docker image from Dockerfile**
No additional configuration is required, simply execute in ligands-classification folder:
> `docker build . -t <name-of-image>`

**Running existing Docker image**
The recommended way of running our Docker image would be:
> `sudo docker run --rm -it --init --gpus=all --ipc=host --volume="<path-to-folder>:/app" -w="/app" <name-of-image> /bin/bash`

with <path-to-folder> pointing to the folder containing data folder and cloned ligands-classification repository

