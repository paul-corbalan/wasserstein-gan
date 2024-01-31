# Wasserstein GAN
This project consisted in studying GANs in the case of Wasserstein distance, as part of the fifth-year course at INSA Toulouse in Applied Mathematics of High Dimensional and Deep Learning.

Here are the members of our group, listed alphabetically by surname:
- Paul Corbalan
- Nicolas Gonel
- Oihan Joyot
- Tristan Portugues
- Florian Zorzynski

Our project is largely inspired by the following resources, which are the initial article of our project as well as the corresponding code.
- Article: [[1701.07875] Wasserstein GAN (arxiv.org)](https://arxiv.org/abs/1701.07875)
- Code: [martinarjovsky/WassersteinGAN (github.com)](https://github.com/martinarjovsky/WassersteinGAN)


## Installation
It's important to note that Python 3.11 was used for this project, particularly for compatibility with the PyTorch library, so we recommend using this version.
1. To install Python 3.11, we recommend using Anaconda, by executing the following command:
    ```shell
    conda create -n wasserstein-gan python=3.11
    ```
2. To activate the environment, simply run the following command:
    ```shell
    conda activate wasserstein-gan
    ```
3. To install the project's dependencies, simply run the following command:
    ```shell
    pip install -r requirements.txt
    ```


## Use
Details of the experiments are given in the [Jupiter Notebook](./notebook.ipynb). However, they can be reproduced simply by executing the following commands:
- For training :
    ```shell
    python main.py
    ```
- For image generation:
    ```shell
    python generate.py
    ```
