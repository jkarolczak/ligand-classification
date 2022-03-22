
#!/usr/bin/env python3
"""Setup testing enviroment"""

import io
from setuptools import setup

with io.open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

requirements = [
    addict==2.4.0,
#     ipython==8.1.1,
#     MinkowskiEngine==0.5.4
#     neptune_client==0.15.1
    numpy==1.22.3,
    pandas==1.4.1,
#     plotly==5.6.0,
    PyYAML==6.0,
    scikit_image==0.19.2,
    scikit_learn==1.0.2,
    skimage==0.0,
#     torch==1.10.2+cpu
#     torchmetrics==0.7.2
    yapf==0.32.0
]


setup(
    name="Ligand classification",
    version=0.1,
    description="Ligand classification using sparse convolutional neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
)
