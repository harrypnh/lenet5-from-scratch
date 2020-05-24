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
│   ├── lenet5_data_0.78.pkl        # LeNet-5 Model with 0.78% Testing Error Rate
│   └── customCNN_data_0.XX.pkl     # CustomCNN Model with 0.XX% Testing Error Rate
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

After each epoch, the settings of the model will be extracted and stored in a `pkl` file using `pickle`. The best model achieves the error rate of `0.78%` on the testing dataset. The training was conducted using CPU only, and the model is evaluated on training and testing datasets once after finishing each epoch, so the total running time is around 3 hours and a half.
<img src="/images/figure_lenet5.png" width="480"/>

## 4. The Structure of CustomCNN

## 5. CustomCNN on MNIST Handwritten Digit Dataset
<img src="/images/figure_custom_cnn.png" width="480"/>

## 6. References
1. [W.-H. Wang, LeNet5 Implementation FROM SCRATCH](https://github.com/mattwang44/LeNet-from-Scratch)
2. [P. Dahal, Classification and Loss Evaluation - Softmax and Cross Entropy Loss](https://deepnotes.io/softmax-crossentropy)
3. [Y. LeCun et al., THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)
4. [Y. LeCun et al., Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

## Appendix 1. LeNet-5 Training Details
```
---------- epoch 1 begin ----------
learning rate: 0.05
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:23<00:00, 13.32it/s]
Done, total cost of epoch 1: 9980.616932611361
0/1 error(s) of training set: 1058 / 60000
0/1 error(s) of testing set: 182 / 10000
Time used: 613.1700155735016 sec
---------- epoch 1 end ------------
---------- epoch 2 begin ----------
learning rate: 0.05
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:10<00:00, 13.61it/s]
Done, total cost of epoch 2: 4288.569345766769
0/1 error(s) of training set: 911 / 60000
0/1 error(s) of testing set: 174 / 10000
Time used: 602.498046875 sec
---------- epoch 2 end ------------
---------- epoch 3 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:18<00:00, 13.42it/s]
Done, total cost of epoch 3: 1746.70291781717
0/1 error(s) of training set: 321 / 60000
0/1 error(s) of testing set: 109 / 10000
Time used: 611.9014346599579 sec
---------- epoch 3 end ------------
---------- epoch 4 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:12<00:00, 13.57it/s]
Done, total cost of epoch 4: 1041.908711761913
0/1 error(s) of training set: 191 / 60000
0/1 error(s) of testing set: 89 / 10000
Time used: 604.4757525920868 sec
---------- epoch 4 end ------------
---------- epoch 5 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:12<00:00, 13.57it/s]
Done, total cost of epoch 5: 788.6754815299188
0/1 error(s) of training set: 217 / 60000
0/1 error(s) of testing set: 110 / 10000
Time used: 606.6137552261353 sec
---------- epoch 5 end ------------
---------- epoch 6 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:12<00:00, 13.57it/s]
Done, total cost of epoch 6: 399.7020344960303
0/1 error(s) of training set: 64 / 60000
0/1 error(s) of testing set: 84 / 10000
Time used: 603.1853880882263 sec
---------- epoch 6 end ------------
---------- epoch 7 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:11<00:00, 13.59it/s]
Done, total cost of epoch 7: 275.0486173883645
0/1 error(s) of training set: 36 / 60000
0/1 error(s) of testing set: 81 / 10000
Time used: 604.7320675849915 sec
---------- epoch 7 end ------------
---------- epoch 8 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:16<00:00, 13.48it/s]
Done, total cost of epoch 8: 192.17748230883083
0/1 error(s) of training set: 27 / 60000
0/1 error(s) of testing set: 82 / 10000
Time used: 611.4439840316772 sec
---------- epoch 8 end ------------
---------- epoch 9 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:17<00:00, 13.45it/s]
Done, total cost of epoch 9: 129.55022561354568
0/1 error(s) of training set: 21 / 60000
0/1 error(s) of testing set: 78 / 10000
Time used: 610.8770899772644 sec
---------- epoch 9 end ------------
---------- epoch 10 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:15<00:00, 13.51it/s]
Done, total cost of epoch 10: 111.32157223683312
0/1 error(s) of training set: 15 / 60000
0/1 error(s) of testing set: 81 / 10000
Time used: 609.8109068870544 sec
---------- epoch 10 end ------------
---------- epoch 11 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:12<00:00, 13.56it/s]
Done, total cost of epoch 11: 96.86704642049718
0/1 error(s) of training set: 12 / 60000
0/1 error(s) of testing set: 82 / 10000
Time used: 605.9128978252411 sec
---------- epoch 11 end ------------
---------- epoch 12 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:14<00:00, 13.53it/s]
Done, total cost of epoch 12: 85.1463094040894
0/1 error(s) of training set: 12 / 60000
0/1 error(s) of testing set: 82 / 10000
Time used: 609.6413428783417 sec
---------- epoch 12 end ------------
---------- epoch 13 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:14<00:00, 13.53it/s]
Done, total cost of epoch 13: 70.7034945142419
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 81 / 10000
Time used: 608.1183385848999 sec
---------- epoch 13 end ------------
---------- epoch 14 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:13<00:00, 13.56it/s]
Done, total cost of epoch 14: 68.59040131924829
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 81 / 10000
Time used: 608.3979847431183 sec
---------- epoch 14 end ------------
---------- epoch 15 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:13<00:00, 13.55it/s]
Done, total cost of epoch 15: 66.72820352795718
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 82 / 10000
Time used: 607.6273169517517 sec
---------- epoch 15 end ------------
---------- epoch 16 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:11<00:00, 13.59it/s]
Done, total cost of epoch 16: 65.04836694104898
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 81 / 10000
Time used: 606.1273622512817 sec
---------- epoch 16 end ------------

learning rate: 0.0005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:15<00:00, 13.49it/s]
Done, total cost of epoch 17: 63.06508614582355
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 81 / 10000
Time used: 607.7152042388916 sec
---------- epoch 17 end ------------
---------- epoch 18 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:14<00:00, 13.54it/s]
Done, total cost of epoch 18: 62.26914988251854
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 83 / 10000
Time used: 608.8773896694183 sec
---------- epoch 18 end ------------
---------- epoch 19 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:11<00:00, 13.61it/s]
Done, total cost of epoch 19: 61.51671766957748
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 83 / 10000
Time used: 605.7930634021759 sec
---------- epoch 19 end ------------
---------- epoch 20 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [09:16<00:00, 13.49it/s]
Done, total cost of epoch 20: 60.8437322474144
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 83 / 10000
Time used: 611.6129689216614 sec
---------- epoch 20 end ------------
Testing with model_data/lenet5_data_9.pkl:
error rate: 0.0078
```

## Appendix 2. CustomCNN Training Details
```
---------- epoch 1 begin ----------
learning rate: 0.05
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:47<00:00,  4.34it/s]
Done, total cost of epoch 1: 7512.089061950697
Time used: 1727.1575927734375 sec
0/1 error(s) of training set: 974 / 60000
0/1 error(s) of testing set: 179 / 10000
---------- epoch 1 end ------------
---------- epoch 2 begin ----------
learning rate: 0.05
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:01<00:00,  4.31it/s]
Done, total cost of epoch 2: 2701.030069792301
Time used: 1741.9387412071228 sec
0/1 error(s) of training set: 459 / 60000
0/1 error(s) of testing set: 104 / 10000
---------- epoch 2 end ------------
---------- epoch 3 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:59<00:00,  4.31it/s]
Done, total cost of epoch 3: 991.4760777114255
Time used: 1739.5507094860077 sec
0/1 error(s) of training set: 155 / 60000
0/1 error(s) of testing set: 81 / 10000
---------- epoch 3 end ------------
---------- epoch 4 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:47<00:00,  4.34it/s]
Done, total cost of epoch 4: 537.0618992298514
Time used: 1727.72611951828 sec
0/1 error(s) of training set: 80 / 60000
0/1 error(s) of testing set: 70 / 10000
---------- epoch 4 end ------------
---------- epoch 5 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:50<00:00,  4.33it/s]
Done, total cost of epoch 5: 322.92186974788046
Time used: 1730.7369742393494 sec
0/1 error(s) of training set: 120 / 60000
0/1 error(s) of testing set: 78 / 10000
---------- epoch 5 end ------------
---------- epoch 6 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:13<00:00,  4.28it/s]
Done, total cost of epoch 6: 157.8592532190945
Time used: 1753.4830327033997 sec
0/1 error(s) of training set: 19 / 60000
0/1 error(s) of testing set: 60 / 10000
---------- epoch 6 end ------------
---------- epoch 7 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:07<00:00,  4.29it/s]
Done, total cost of epoch 7: 100.79359826015427
Time used: 1747.4527938365936 sec
0/1 error(s) of training set: 13 / 60000
0/1 error(s) of testing set: 60 / 10000
---------- epoch 7 end ------------
---------- epoch 8 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:12<00:00,  4.28it/s]
Done, total cost of epoch 8: 78.0056792565656
Time used: 1753.0950682163239 sec
0/1 error(s) of training set: 8 / 60000
0/1 error(s) of testing set: 58 / 10000
---------- epoch 8 end ------------
---------- epoch 9 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:05<00:00,  4.30it/s]
Done, total cost of epoch 9: 51.035547905601796
Time used: 1745.3303685188293 sec
0/1 error(s) of training set: 5 / 60000
0/1 error(s) of testing set: 56 / 10000
---------- epoch 9 end ------------
---------- epoch 10 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:54<00:00,  4.32it/s]
Done, total cost of epoch 10: 43.13419022511757
Time used: 1734.9669053554535 sec
0/1 error(s) of training set: 4 / 60000
0/1 error(s) of testing set: 58 / 10000
---------- epoch 10 end ------------
---------- epoch 11 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:03<00:00,  4.30it/s]
Done, total cost of epoch 11: 36.884440836821824
Time used: 1743.7268526554108 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 11 end ------------
---------- epoch 12 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:04<00:00,  4.30it/s]
Done, total cost of epoch 12: 32.77747294422119
Time used: 1745.1878411769867 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 12 end ------------
---------- epoch 13 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:56<00:00,  4.32it/s]
Done, total cost of epoch 13: 27.821463542623405
Time used: 1736.5197653770447 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 13 end ------------
---------- epoch 14 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:45<00:00,  4.35it/s]
Done, total cost of epoch 14: 26.896923290489326
Time used: 1725.6785237789154 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 14 end ------------
---------- epoch 15 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:00<00:00,  4.31it/s]
Done, total cost of epoch 15: 26.312016395187154
Time used: 1740.493469953537 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 15 end ------------
---------- epoch 16 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:05<00:00,  4.30it/s]
Done, total cost of epoch 16: 25.66753420888172
Time used: 1745.4987199306488 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 16 end ------------
---------- epoch 17 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:56<00:00,  4.32it/s]
Done, total cost of epoch 17: 25.01816945154716
Time used: 1736.6064808368683 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 17 end ------------
---------- epoch 18 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [28:53<00:00,  4.33it/s]
Done, total cost of epoch 18: 24.735334201958555
Time used: 1733.32337474823 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 18 end ------------
---------- epoch 19 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:02<00:00,  4.30it/s]
Done, total cost of epoch 19: 24.48233744714694
Time used: 1743.0635526180267 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 19 end ------------
---------- epoch 20 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|██████████████████████████████████████████████████████| 7500/7500 [29:10<00:00,  4.29it/s]
Done, total cost of epoch 20: 24.235228818552656
Time used: 1750.2566900253296 sec
0/1 error(s) of training set: 3 / 60000
0/1 error(s) of testing set: 62 / 10000
---------- epoch 20 end ------------
Total time used: 37362.436579704285 sec
Testing with model_data/customCNN_data_9.pkl:
error rate: 0.0056
```