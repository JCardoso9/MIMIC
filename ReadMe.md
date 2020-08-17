# Deep Learning for the automatic generation of chest x-ray diagnosis reports


Code for my Msc. thesis on the automatic generation of medical reports for chest X-ray images using deep learning.

We employed encoder-decoder models and compared generating words using a softmax output layer vs a vector output layer and then choosing the nearest word embedding of the generated vector.

The base architecture follows the Show, Attend and Tell model implemented in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.

The hierarchical model is inspired by Jing et al, 2018.

We also created baselines such as random sampling and nearest neighbour searching using the image representations from the CNN component and then computing comparisons through FAISS https://github.com/facebookresearch/faiss

Currently writing article and finishing dissertation
