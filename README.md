# Wasserstein GAN
L'objectif de ce projet était d'étudier les GANs dans le cas de la distance de Wasserstein.

Voici les membres de notre groupe classés par ordres alphabétiques pour leur nom de famille :
- Paul Corbalan
- Nicolas Gonel
- Oihan Joyot
- Tristan Portugues
- Florian Zorzynski

Notre projet s'inspire grandement des ressources suivantes qui sont l'article initial de notre projet ainsi que le code correspondant.
- Article : [[1701.07875] Wasserstein GAN (arxiv.org)](https://arxiv.org/abs/1701.07875)
- Code : [martinarjovsky/WassersteinGAN (github.com)](https://github.com/martinarjovsky/WassersteinGAN)


## Installation
Il est important de noter que Python 3.11 a été utilisé pour ce projet notamment au niveau de la compatibilité avec la librairie PyTorch, il est donc recommandé d'utiliser cette version.
1. Pour installer Python 3.11, il est recommandé de le faire avec Anaconda, en exécutant la commande suivante :
    ```shell
    conda create -n wasserstein-gan python=3.11
    ```
2. Pour activer l'environnement, il suffit d'exécuter la commande suivante :
    ```shell
    conda activate wasserstein-gan
    ```
3. Pour installer les dépendances du projet, il suffit d'exécuter la commande suivante :
    ```shell
    pip install -r requirements.txt
    ```


## Utilisation
Le détail des expériences est détaillé dans le [Jupiter Notebook](./notebook.ipynb). Cependant il est possible de reproduire celle-ci simplement en exécutant les commandes suivantes :
- Pour l'entraînement :
    ```shell
    python main.py
    ```
- Pour la génération d'images :
    ```shell
    python generate.py
    ```
