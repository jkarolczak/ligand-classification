# ligands-classification

## Environement

## Frameworks
* [MinkowskiEngine](https://nvidia.github.io/MinkowskiEngine/index.html) (recommended to use)
  * [example](https://github.com/NVIDIA/MinkowskiEngine/blob/432ce88ab1735b84a2d4fb21a5d45af98a968f1b/examples/classification_modelnet40.py) (classification using Sparse CNN)

Build docker image
```
git clone https://github.com/NVIDIA/MinkowskiEngine
mv MinkowskiEngine/ minkowskiengine/
cd minkowskiengine
docker build -t MinkowskiEngine docker
```

Run docker container
```
docker run -dit --name minkowskiengine minkowskiengine
```

Inspect containers
```
docker container ls -a
```

Start docker container
```
docker container start minkowskiengine
```

Stop docker container
```
docker container stop minkowskiengine
```

* [SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
* [torchsparse](https://reposhub.com/python/deep-learning/mit-han-lab-torchsparse.html)

# Papers
* [Automatic recognition of ligands in electron density by machine learning](https://academic.oup.com/bioinformatics/article/35/3/452/5055122)
* [Recognizing and validating ligands with CheckMyBlob](https://academic.oup.com/nar/article/49/W1/W86/6255698)
