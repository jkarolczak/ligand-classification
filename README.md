[![Streamlit - Demo](https://img.shields.io/badge/Streamlit-Demo-informational)](https://ligands.cs.put.poznan.pl)
[![Zenodo - Data](https://img.shields.io/badge/Zenodo-Data-informational)](https://zenodo.org/records/10908325)
![example workflow](https://github.com/jkarolczak/ligands-classification/actions/workflows/python-app.yml/badge.svg)


# Deep Learning Methods for Ligand Identification in Density Maps
[Jacek Karolczak](https://github.com/jkarolczak), [Anna Przybyłowska](https://github.com/annprzy), [Konrad Szewczyk](https://github.com/konradszewczyk), [Witold Taisner](https://github.com/wtaisner), [John M. Heumann](https://github.com/jmheumann), Michael H.B. Stowell, [Michał Nowicki](https://github.com/MichalNowicki?tab=repositories), [Dariusz Brzezinski](https://github.com/dabrze)

Accurately identifying ligands plays a crucial role in structure-guided drug design. 
Based on density maps from X-ray diffraction or cryogenic-sample electron microscopy (cryoEM), scientists verify whether small-molecule ligands bind to active sites. 
However, the interpretation of density maps is challenging, and cognitive bias can sometimes mislead investigators into modeling fictitious compounds. 
Ligand identification can be aided by automatic methods, but existing approaches are available only for X-ray diffraction. 
Here, we propose to identify ligands using a deep learning approach that treats density maps as 3D point clouds. 
We show that the proposed model is on par with existing methods for X-ray crystallography while also being applicable to cryoEM density maps. 
Our study demonstrates that electron density map fragments can be used to train models that can be applied to cryoEM structures, but also highlights challenges associated with the standardization of electron microscopy maps and the quality assessment of cryoEM ligands.

In the repository, we provide the code for the experiments conducted in the paper, including models implementations and transformations for generating datasets.
To reproduce the results, use scripts from the `scripts` directory. 
Configuration files for the experiments are available in the `cfg` directory.

Weights of the model that revealed as the best in the paper are published as `model.pt` ([link](https://github.com/jkarolczak/ligand-classification/blob/main/model.pt)).

## Environment setup

### Docker

To simplify the setup and ensure consistency, to , we provide a Docker configuration that includes all necessary dependencies.

#### Prerequisites

Ensure you have the following installed:

- [Docker](https://docs.docker.com/engine/install/) (>= 20.0.0)
- [Docker Compose](https://docker-docs.netlify.app/compose/install/#install-compose) (>= 2.0.0)
- (Optional) For GPU support:
  - [CUDA](https://developer.nvidia.com/cuda-downloads) (>= 11.3)
  - [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
  - [cuDNN](https://developer.nvidia.com/cudnn) (>= 8.0)
  - [NVIDIA Docker Runtime](https://developer.nvidia.com/nvidia-container-runtime)

#### Steps to Start

1. Clone this repository.
2. Set the necessary permissions: `sudo chmod 744 ./start.sh ./stop.sh`
3. Configure the environment by editing the `docker/.env` file:
   - Adjust `PYTORCH`, `CUDA`, and `CUDNN` settings if needed (for GPU use).
   - Set the `DATA_PATH` to point to your data directory. Default is `../../data/`.
4. Start the container:
   - For GPU use: `./start.sh`
   - For CPU use: `./start.sh cpu`
5. To stop the container:
   - For GPU use: `./stop.sh`
   - For CPU use: `./stop.sh cpu`

## Demo

The best model from the paper can be tested without the need to install anything. 
The model is deployed as a Streamlit app under the link [ligands.xyz](https://ligands.xyz).

## Data

All the data necessary to reproduce results is available at [Zenodo](https://zenodo.org/record/10908325).

Repository with code for extracting ligands from CryoEM difference maps is a submodule of this repository, but can be also found [here](https://github.com/dabrze/cryo-em-ligand-cutter/tree/6032b5701cad7a4db86f780b91c2078907e36e42).

Additionally, the processed data used for training the model that revealed as the best in the paper is available [here](https://ligands.blob.core.windows.net/ligands/blobs_uniform_2000_max.tar.gz).

## Citation
```
Space reserved for bibtex entry
```