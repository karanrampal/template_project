# Template for projects

This is a template, for future projects, which can be hosted on Github. After cloning and manipulating the git directory (deleting :wink:) it can be used as starting point for any other project.

Contains starter code for model creation, training, testing and evaluation using Pytorch. Hence no need to create these from scratch everytime, just modify the relevant files for the particular project.

## Directory structure
Structure of the project
```
.github/
    workflows/
        main.yaml
experiments/
    base_model/
        params.json
    param_search/
        params.json
model/
    __init__.py
    data_loader.py
    net.py
tests/
    __init__.py
.gitignore
evaluate.py
LICENSE
Makefile
README.md
requirements.txt
search_hyperparams.py
synthesize_results.py
tox.ini
train.py
utils.py
visualization.py
```

## Usage
The simplest way to use this repository as a template for a project is to clone it and then delete the `.git` directory. Then git can be re-initialized,
```
git clone <url> <newprojname>
cd <newprojname>
rm -r .git
git init
```
To run visualization,
```
python visualization.py
tensorboard --logdir=experiments/
```

## Requirements
I used Anaconda with python3,

```
conda create -n <yourenvname> python=<3.x>
conda activate <yourenvname>
conda install -n <yourenvname> --file requirements.txt
```

## Reference
Borrowed heavily and modified slightly from the lovely [Stanford course](https://github.com/cs230-stanford "Stanford's deep learning course") with code written by Surag Nair, Olivier Moindrot and Guillaume Genthial.

*I don't know how to cite them, hopefully this is enough as reference, for me to not get in any kind of trouble.*
