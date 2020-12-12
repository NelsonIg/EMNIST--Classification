# EMNIST -Classification
This project aims to implement a simple classification of characters using common classifiers. Basics of machine learning and machine vision are used.
## Files
1. src/ML-Models.ipynp: training and testing of models
2. src/char_recognition.ipynp: classifying characters from provided image.
3. src/models: contains classifiers
## Installation
Don't use python 3.9. Sklearn has at this moment some trouble with it.
If tkinter is not installed by default, then get it for your environment. Anaconda is recommended, since it has tk by default. You will need Jupyter as well for training models, which comes with Anaconda too.  
If Anaconda is installed:
```
conda create -n venv python=3.8
conda activate venv
pip install -r requirements.txt
```
## Example Output  
Here the `bal_MLP_clf` is used to classify digits and letters. But be free to train new models!
![ ](https://github.com/NelsonIg/EMINST-Classification/blob/master/images/example_inp.jpg)
![ ](https://github.com/NelsonIg/EMINST-Classification/blob/master/images/example_out.jpg)

