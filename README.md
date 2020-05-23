# LeNet-5 from Scratch
LeNet-5 Convolution Neural Network built primarily using NumPy and applied on the MNIST Handwritten Digit Dataset.
## 1. Repository Structure
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
├── images/
│   ├── figure_lenet5.png
│   └── figure_custom_cnn.png
│
├── lenet5_mnist.py                 # Train and Test LeNet-5 on MNIST Dataset
├── custom_cnn_mnist.py             # Train and Test CustomCNN on MNIST Dataset
└── error_rate_plot.py
```
## 2. The Structure of LeNet-5
- The LeNet-5 has the following structure:<br/>
`C1 -> ReLU1 -> S2 -> C3 -> ReLU2 -> S4 -> C5 -> ReLU3 -> F6 -> ReLU4 -> F7 (softmax)`
  - C1 is a convolutional layer having 6 `5x5x1` filters and no zero-padding. Input: `32x32x1`, Output: `28x28x6`.
  - S2 is a max-pooling layer of size `2x2` and stride `2`. Input: `28x28x6`, Output: `14x14x6`.
  - C3 is a convolutional layer having 16 `5x5x6` filters and no zero-padding. Input: `14x14x6`, Output: `10x10x16`.
  - S4 is a max-pooling layer of size `2x2` and stride `2`. Input: `10x10x16`, Output: `5x5x16`.
  - C5 is a convolutional layer having 120 `5x5x16` filters and no zero-padding. Input: `5x5x16`, Output: `1x1x120`.
  - F6 is a fully-connected layer having `84` neurons, each of which takes `120` feature inputs.
  - F7 is a fully-connected layer having `10` neurons, each of which takes `84` feature inputs. The softmax activation function is applied on the output vector of size `10` from the `10` neurons. This softmax vector corresponds to `10` classes for digit `0` to `9`.
  - The cost function of this model is cross-entropy.
## 3. Training and Testing on MNIST Handwritten Digit Dataset
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

After each epoch, the settings of the model will be extracted and stored in a .pkl file using `pickle`. The best model achieves the error rate of `0.86%` on the testing dataset. The training was conducted using CPU only, and the model is evaluated on training and testing datasets once after finishing each epoch, so the total running time is around 3 hours.

<img src="/images/figure_lenet5.png" width="480"/>

## CustomCNN on the MNIST Handwritten Digit Dataset
```
---------- epoch 1 begin ----------
learning rate: 0.05
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [17:32<00:00,  3.56it/s]
Done, total cost of epoch 1: 70693.40160759492
Time used: 1052.3372430801392 sec
0/1 error(s) of training set: 1715 / 60000
0/1 error(s) of testing set: 269 / 10000
---------- epoch 1 end ------------
---------- epoch 2 begin ----------
learning rate: 0.05
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:57<00:00,  3.69it/s]
Done, total cost of epoch 2: 3933.0811879639864
Time used: 1018.2483661174774 sec
0/1 error(s) of training set: 711 / 60000
0/1 error(s) of testing set: 117 / 10000
---------- epoch 2 end ------------
---------- epoch 3 begin ----------
learning rate: 0.02
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:56<00:00,  3.69it/s]
Done, total cost of epoch 3: 1653.6516252292622
Time used: 1018.1669759750366 sec
0/1 error(s) of training set: 419 / 60000
0/1 error(s) of testing set: 95 / 10000
---------- epoch 3 end ------------
---------- epoch 4 begin ----------
learning rate: 0.02
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:55<00:00,  3.69it/s]
Done, total cost of epoch 4: 1216.7510394896653
Time used: 1016.6399211883545 sec
0/1 error(s) of training set: 249 / 60000
0/1 error(s) of testing set: 88 / 10000
---------- epoch 4 end ------------
---------- epoch 5 begin ----------
learning rate: 0.02
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:50<00:00,  3.71it/s]
Done, total cost of epoch 5: 980.3839502514301
Time used: 1011.6973021030426 sec
0/1 error(s) of training set: 206 / 60000
0/1 error(s) of testing set: 92 / 10000
---------- epoch 5 end ------------
---------- epoch 6 begin ----------
learning rate: 0.01
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:52<00:00,  3.70it/s]
Done, total cost of epoch 6: 597.3477996261356
Time used: 1014.2297012805939 sec
0/1 error(s) of training set: 98 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 6 end ------------
---------- epoch 7 begin ----------
learning rate: 0.01
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:49<00:00,  3.72it/s]
Done, total cost of epoch 7: 488.10392397333123
Time used: 1010.3244409561157 sec
0/1 error(s) of training set: 82 / 60000
0/1 error(s) of testing set: 72 / 10000
---------- epoch 7 end ------------
---------- epoch 8 begin ----------
learning rate: 0.01
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:44<00:00,  3.73it/s]
Done, total cost of epoch 8: 413.2828309970186
Time used: 1006.0620262622833 sec
0/1 error(s) of training set: 89 / 60000
0/1 error(s) of testing set: 65 / 10000
---------- epoch 8 end ------------
---------- epoch 9 begin ----------
learning rate: 0.005
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:44<00:00,  3.73it/s]
Done, total cost of epoch 9: 314.6321334892624
Time used: 1005.6263389587402 sec
0/1 error(s) of training set: 49 / 60000
0/1 error(s) of testing set: 72 / 10000
---------- epoch 9 end ------------
---------- epoch 10 begin ----------
learning rate: 0.005
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:42<00:00,  3.74it/s]
Done, total cost of epoch 10: 280.28351972679184
Time used: 1003.4593198299408 sec
0/1 error(s) of training set: 42 / 60000
0/1 error(s) of testing set: 70 / 10000
---------- epoch 10 end ------------
---------- epoch 11 begin ----------
learning rate: 0.005
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:41<00:00,  3.74it/s]
Done, total cost of epoch 11: 259.28408622777744
Time used: 1002.886332988739 sec
0/1 error(s) of training set: 39 / 60000
0/1 error(s) of testing set: 69 / 10000
---------- epoch 11 end ------------
---------- epoch 12 begin ----------
learning rate: 0.005
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:38<00:00,  3.75it/s]
Done, total cost of epoch 12: 239.00801157776317
Time used: 999.998584985733 sec
0/1 error(s) of training set: 40 / 60000
0/1 error(s) of testing set: 69 / 10000
---------- epoch 12 end ------------
---------- epoch 13 begin ----------
learning rate: 0.001
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:39<00:00,  3.75it/s]
Done, total cost of epoch 13: 200.7725918355597
Time used: 1001.1002180576324 sec
0/1 error(s) of training set: 33 / 60000
0/1 error(s) of testing set: 70 / 10000
---------- epoch 13 end ------------
---------- epoch 14 begin ----------
learning rate: 0.001
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:31<00:00,  3.78it/s]
Done, total cost of epoch 14: 194.61008906020797
Time used: 992.6318888664246 sec
0/1 error(s) of training set: 33 / 60000
0/1 error(s) of testing set: 69 / 10000
---------- epoch 14 end ------------
---------- epoch 15 begin ----------
learning rate: 0.001
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:36<00:00,  3.76it/s]
Done, total cost of epoch 15: 190.6921441341232
Time used: 997.4112141132355 sec
0/1 error(s) of training set: 32 / 60000
0/1 error(s) of testing set: 70 / 10000
---------- epoch 15 end ------------
---------- epoch 16 begin ----------
learning rate: 0.001
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:37<00:00,  3.76it/s]
Done, total cost of epoch 16: 186.93679232756148
Time used: 998.7812180519104 sec
0/1 error(s) of training set: 31 / 60000
0/1 error(s) of testing set: 71 / 10000
---------- epoch 16 end ------------
---------- epoch 17 begin ----------
learning rate: 0.0005
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:40<00:00,  3.75it/s]
Done, total cost of epoch 17: 182.0209171069152
Time used: 1002.478453874588 sec
0/1 error(s) of training set: 31 / 60000
0/1 error(s) of testing set: 70 / 10000
---------- epoch 17 end ------------
---------- epoch 18 begin ----------
learning rate: 0.0005
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:32<00:00,  3.78it/s]
Done, total cost of epoch 18: 180.13467852637987
Time used: 994.4903690814972 sec
0/1 error(s) of training set: 32 / 60000
0/1 error(s) of testing set: 71 / 10000
---------- epoch 18 end ------------
---------- epoch 19 begin ----------
learning rate: 0.0005
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:39<00:00,  3.75it/s]
Done, total cost of epoch 19: 178.5197966851922
Time used: 1001.3075678348541 sec
0/1 error(s) of training set: 30 / 60000
0/1 error(s) of testing set: 72 / 10000
---------- epoch 19 end ------------
---------- epoch 20 begin ----------
learning rate: 0.0005
batch size: 16
Training:
100%|█████████████████████████████████████████████| 3750/3750 [16:35<00:00,  3.77it/s]
Done, total cost of epoch 20: 177.2519038896825
Time used: 997.0255508422852 sec
0/1 error(s) of training set: 30 / 60000
0/1 error(s) of testing set: 71 / 10000
---------- epoch 20 end ------------
Total time used: 37806.881603717804 sec
Testing with model_data/customCNN_data_6.pkl:
error rate: 0.0062
```
## References
1. [Matt Wang's repository](https://github.com/mattwang44/LeNet-from-Scratch)
2. [Classification and Loss Evaluation - Softmax and Cross Entropy Loss by Paras Dahal](https://deepnotes.io/softmax-crossentropy)
3. [The MNIST database of handwritten digits by Yann LeCun, Corinna Cortes and Christopher Burges](http://yann.lecun.com/exdb/mnist/)
