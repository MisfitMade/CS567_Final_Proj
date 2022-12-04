# CS567_Final_Proj

## Example of setting up a conda environment for this project
#### Maybe not neccesary, but guarantees you'll be able to use, debug, run the code + keep you from creating a clashing Python version dependency between Python projects. Have conda installed, then in a conda terminal:
* `conda create -n 567FP python=3.9`
* `conda activate 567FP`
* `pip install tensorflow==2.11.0 tensorflow_hub eyeD3 jupyterlab numpy matplotlib numba librosa pandas scikit-learn`

Now, when you work in the project and/or run its code, do so in this 567FP environment/conda space.

## Files
* `FP567_Lib.py` -- This is a python file where we can define functions and use them throughout the repo. It is a library of functions we defined to help with cleaning, processing data, etc.
* `FP567.ipynb` -- This is a notebook for peicing together the parts we made into a model, training it, inspecting it.
* `Embedding_Updates.ipynb` -- This is a notebook used for cleaning, sorting, embedding and reducing the dimension of the updates.

## General setup before running any code

## Using the CS567_Final_Proj code
