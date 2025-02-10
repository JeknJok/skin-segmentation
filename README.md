# project

Datasets used:
#1.
http://cs-chan.com/downloads_skin_dataset.html
Citation:
A Fusion Approach for Efficient Human Skin Detection
W.R. Tan, C.S. Chan, Y. Pratheepan and J. Condell
IEEE Transactions on Industrial Informatics, vol.8(1):138-147 (T-II 2012)

#2.
...

COMPILING AND TRAINING MODEL

to train model, first, run:

> pip install -r requirements.txt

to install all dependencies.

then, run the train.py
to test, define the img file path in test.py then run it. You will see a side-by-side comparison of an original image, true mask, and predicted mask (pred_mask is the one that the model produce.)
