# Backpropagation for a Multi-Perceptron Layer
This repository contains the implementation for the second ADM project, in which we implement the backpropagation algorithm for learning in a Multi-Perceptron layer network. The structure of the repository is as follows:

```
├── README.md
├── requirements.txt
├── results
│   ├── RandGradientCheck.png
│   ├── random_gradcheck.txt
│   └── random_gradcheck_plot.py
├── src
│   ├── __init__.py
│   ├── activation_functions.py
│   ├── ann.py
│   └── loss_functions.py
├── test
│   ├── __init__.py
│   ├── binary_category_test.py
│   ├── error_stress_test.py
│   ├── gradient_check_test.py
│   ├── mnist_test.py
│   └── randomized_gradient_check_test.py
```
## Get Started
This project was implemented in python, so one must have Python3 installed to run it, and it is recommended to have pip too. To avoid issues with the syntax, we recommend to have Python on version 3.11. 

To make sure that you have the correct version of python, you can run the following command:
```bash
python --version
```
if you don't have conda installed visit: https://www.anaconda.com/download/

It is recommended to use conda (or miniconda) to create a new environment with the correct version of python. To do this, you can run the following command:

```bash
conda create -n ADM python=3.11
conda activate ADM
```

Before running the project, one must install all the necessary requirements. This are all listed in the ```requirements.txt``` file. To install them it's enough to run the following command:
```bash
pip install -r ./requirements.txt
```

To run the codes, one can go into the ```test``` folder. There we have 5 different scripts:
- ```randomized_gradient_check_test.py```
- ```error_stress_test.py```
- ```binary_category_test.py``` 
- ```mnist_test.py```
- ```gradient_check_test.py```
The ones mentioned in the report are the first three, the others are auxiliar. These scripts use the implemented neural network that can be found in the ```src``` folder. To run them it is enough to call python3.
