# LeNet-5 from Scratch
LeNet-5 Convolution Neural Network built primarily using NumPy and applied on the MNIST Handwritten Digit Dataset.

## 1. Project Structure
```
decision-tree-from-scratch/
├── dataset/
│   ├── MNIST/
│   │   ├── train-images-idx3-ubyte # MNIST Training Images
│   │   ├── train-labels-idx1-ubyte # MNIST Training Labels
│   │   ├── t10k-images-idx3-ubyte  # MNIST Testing Images
│   │   └── t10k-labels-idx1-ubyte  # MNIST Testing Labels
│   │
│   └── download.py                 # Download MNIST Dataset
│
├── utils/
│   ├── cnn_layers.py               # All Layers of a CNN
│   ├── lenet5_model.py             # LeNet-5 Model
│   └── custom_cnn_model.py         # CustomCNN Model
│
├── model_data/
│   ├── lenet5_data_0.86.pkl        # LeNet-5 Model with 0.86% Testing Error Rate
│   └── customCNN_data_0.62.pkl     # CustomCNN Model with 0.62% Testing Error Rate
│
├── lenet5_mnist.py                 # Train and Test LeNet-5 on MNIST Dataset
└── custom_cnn_mnist.py             # Train and Test CustomCNN on MNIST Dataset
```

## 2. The Structure of LeNet-5
<img src="/images/lenet5_structure.png"/>

The LeNet-5 has the following structure:<br/>
`C1 -> ReLU1 -> S2 -> C3 -> ReLU2 -> S4 -> C5 -> ReLU3 -> F6 -> ReLU4 -> F7 (softmax)`
- C1 is a convolutional layer having 6 `5x5x1` filters and no zero-padding.
- S2 is a max-pooling layer of size `2x2` and stride `2`.
- C3 is a convolutional layer having 16 `5x5x6` filters and no zero-padding.
- S4 is a max-pooling layer of size `2x2` and stride `2`.
- C5 is a convolutional layer having 120 `5x5x16` filters and no zero-padding.
- F6 is a fully-connected layer having `84` neurons, each of which takes `120` feature inputs.
- F7 is a fully-connected layer having `10` neurons, each of which takes `84` feature inputs. The softmax activation function is applied on the output vector of size `10` from the `10` neurons. This softmax vector corresponds to `10` classes for digit `0` to `9`.
- The cost function of this model is cross-entropy.

## 3. LeNet-5 on MNIST Handwritten Digit Dataset
- The original size of an image in the MNIST dataset is `28x28`. Before training on LeNet-5, all MNIST images are added a zero-padding of size `2`, after which they all have the size of `32x32`.
- The training has `20` epoches, and the learning rate decreases after each epoch.
- The learning rates are set as follows.
  - Epoch 1 and 2: `0.05`
  - Epoch 3, 4 and 5: `0.02`
  - Epoch 6, 7 and 8: `0.01`
  - Epoch 9, 10, 11 and 12: `0.005`
  - Epoch 13, 14, 15 and 16: `0.001`
  - Epoch 17, 18, 19 and 20: `0.0005`
- The batch size is `8` to allow more weigth and bias updates which may quickly reduce the error rate over `20` epoches.<br/>

After each epoch, the settings of the model will be extracted and stored in a `pkl` file using `pickle`. The best model achieves the error rate of `0.86%` on the testing dataset. The training was conducted using CPU only, and the model is evaluated on training and testing datasets once after finishing each epoch, so the total running time is around 3 hours.
<img src="/images/figure_lenet5.png" width="480"/>

## 4. The Structure of CustomCNN

## 5. CustomCNN on MNIST Handwritten Digit Dataset

## 6. References
1. [Matt Wang's repository](https://github.com/mattwang44/LeNet-from-Scratch)
2. [Classification and Loss Evaluation - Softmax and Cross Entropy Loss by Paras Dahal](https://deepnotes.io/softmax-crossentropy)
3. [The MNIST database of handwritten digits by Yann LeCun, Corinna Cortes and Christopher Burges](http://yann.lecun.com/exdb/mnist/)
