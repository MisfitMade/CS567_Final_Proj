# CS567_Final_Proj

## Example of setting up a conda environment for this project
#### Maybe not neccesary, but guarantees you'll be able to use, debug, run the code + keep you from creating a clashing Python version dependency between Python projects. Have conda installed, then in a conda terminal:
* `conda create -n 567FP python=3.10`
* `conda activate 567FP`
* `pip install tensorflow tensorflow_hub jupyterlab numpy matplotlib pandas scikit-learn`

Now, when you work in the project and/or run its code, do so in this 567FP environment/conda space.

## Files
* `FP567_Lib.py` -- This is a python file where we can define functions and use them throughout the repo. It is a library of functions we defined to help with cleaning, processing data, etc.
* `FP567_Training.ipynb` -- This is a notebook for peicing together the parts we made into a model, training it, inspecting how well it does.
* `Embedding_Updates.ipynb` -- This is a notebook used for cleaning, sorting, embedding and reducing the dimension of the updates.
* `FP567_Forcasting.ipynb` -- This is a notebook used for loading in saved models, using them to forcast the state of the market around certain dates, plus inspect the true state of the market around that date and compare.
* `extract price.py` -- This is a python file that returns the price 30 days before and 30 days after the given date
## General setup before running any code
* Python
* Tensorflow
* Numpy
* Matplotlib
## Using the CS567_Final_Proj code
