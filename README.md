# LeNet-5 from Scratch
LeNet-5 Convolution Neural Network built primarily using NumPy and applied on the MNIST Handwritten Digit Dataset.
## Training and Testing on MNIST Handwritten Digit Dataset
- The training has 20 epoches, and the learning rate decreases after each epoch.
- The learning rates are set as follows.
  - Epoch 1 and 2: 0.05
  - Epoch 3, 4 and 5: 0.02
  - Epoch 6, 7 and 8: 0.01
  - Epoch 9, 10, 11 and 12: 0.005
  - Epoch 13, 14, 15 and 16: 0.001
  - Epoch 17, 18, 19 and 20: 0.0005
- The batch size is 8 to allow more weigth and bias updates which may quickly reduce the error rate over 20 epoches.<br />

After each epoch, the settings of the model will be extracted and stored in a .pkl file using `pickle`. The best model achieves the error rate of `0.86%` on the testing dataset. The training was conducted using CPU only, and the model is evaluated on training and testing datasets once after finishing each epoch, so the total running is around 3 hours.

<img src="/images/figure_batch_size_8.png" width="480"/>

```
---------- epoch 1 begin ----------
learning rate: 0.05
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:03<00:00, 13.81it/s]
Done, total cost of epoch 1: 9301.628953437894
0/1 error(s) of training set: 1337 / 60000
0/1 error(s) of testing set: 240 / 10000
Time used: 594.6615204811096 sec
---------- epoch 1 end ------------
---------- epoch 2 begin ----------
learning rate: 0.05
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:04<00:00, 13.78it/s]
Done, total cost of epoch 2: 3860.164777672052
0/1 error(s) of training set: 679 / 60000
0/1 error(s) of testing set: 158 / 10000
Time used: 593.219001531601 sec
---------- epoch 2 end ------------
---------- epoch 3 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:06<00:00, 13.73it/s]
Done, total cost of epoch 3: 1639.490457199866
0/1 error(s) of training set: 273 / 60000
0/1 error(s) of testing set: 102 / 10000
Time used: 595.4691672325134 sec
---------- epoch 3 end ------------
---------- epoch 4 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:00<00:00, 13.87it/s]
Done, total cost of epoch 4: 1032.8087742720043
0/1 error(s) of training set: 176 / 60000
0/1 error(s) of testing set: 100 / 10000
Time used: 590.9740438461304 sec
---------- epoch 4 end ------------
---------- epoch 5 begin ----------
learning rate: 0.02
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:05<00:00, 13.74it/s]
Done, total cost of epoch 5: 750.0928390905117
0/1 error(s) of training set: 135 / 60000
0/1 error(s) of testing set: 97 / 10000
Time used: 595.2709658145905 sec
---------- epoch 5 end ------------
---------- epoch 6 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:02<00:00, 13.82it/s]
Done, total cost of epoch 6: 386.05077217256047
0/1 error(s) of training set: 55 / 60000
0/1 error(s) of testing set: 93 / 10000
Time used: 593.1165690422058 sec
---------- epoch 6 end ------------
---------- epoch 7 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [08:58<00:00, 13.93it/s]
Done, total cost of epoch 7: 256.35310094928525
0/1 error(s) of training set: 40 / 60000
0/1 error(s) of testing set: 92 / 10000
Time used: 588.3213109970093 sec
---------- epoch 7 end ------------
---------- epoch 8 begin ----------
learning rate: 0.01
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:04<00:00, 13.78it/s]
Done, total cost of epoch 8: 184.7650054495611
0/1 error(s) of training set: 28 / 60000
0/1 error(s) of testing set: 88 / 10000
Time used: 594.8452796936035 sec
---------- epoch 8 end ------------
---------- epoch 9 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [08:59<00:00, 13.90it/s]
Done, total cost of epoch 9: 118.05145979338414
0/1 error(s) of training set: 13 / 60000
0/1 error(s) of testing set: 89 / 10000
Time used: 590.5513620376587 sec
---------- epoch 9 end ------------
---------- epoch 10 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:04<00:00, 13.77it/s]
Done, total cost of epoch 10: 99.05161803056046
0/1 error(s) of training set: 12 / 60000
0/1 error(s) of testing set: 86 / 10000
Time used: 595.0000903606415 sec
---------- epoch 10 end ------------
---------- epoch 11 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [08:59<00:00, 13.90it/s]
Done, total cost of epoch 11: 87.91350042066031
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 90 / 10000
Time used: 590.5205237865448 sec
---------- epoch 11 end ------------
---------- epoch 12 begin ----------
learning rate: 0.005
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:06<00:00, 13.72it/s]
Done, total cost of epoch 12: 74.78545186783344
0/1 error(s) of training set: 11 / 60000
0/1 error(s) of testing set: 90 / 10000
Time used: 598.6651961803436 sec
---------- epoch 12 end ------------
---------- epoch 13 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:00<00:00, 13.88it/s]
Done, total cost of epoch 13: 63.89091310729966
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 91 / 10000
Time used: 591.5527698993683 sec
---------- epoch 13 end ------------
---------- epoch 14 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [08:59<00:00, 13.90it/s]
Done, total cost of epoch 14: 61.56923261306023
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 92 / 10000
Time used: 591.15025639534 sec
---------- epoch 14 end ------------
---------- epoch 15 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:00<00:00, 13.88it/s]
Done, total cost of epoch 15: 60.046941828972976
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 93 / 10000
Time used: 592.1293377876282 sec
---------- epoch 15 end ------------
---------- epoch 16 begin ----------
learning rate: 0.001
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:05<00:00, 13.74it/s]
Done, total cost of epoch 16: 58.52672434408471
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 91 / 10000
Time used: 598.4910805225372 sec
---------- epoch 16 end ------------
---------- epoch 17 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:00<00:00, 13.88it/s]
Done, total cost of epoch 17: 56.76377098991184
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 91 / 10000
Time used: 594.0652556419373 sec
---------- epoch 17 end ------------
---------- epoch 18 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:05<00:00, 13.74it/s]
Done, total cost of epoch 18: 56.082716533938076
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 91 / 10000
Time used: 597.767697095871 sec
---------- epoch 18 end ------------
---------- epoch 19 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:04<00:00, 13.77it/s]
Done, total cost of epoch 19: 55.46364412844892
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 91 / 10000
Time used: 596.5782310962677 sec
---------- epoch 19 end ------------
---------- epoch 20 begin ----------
learning rate: 0.0005
batch size: 8
Training:
100%|█████████████████████████████████████████████| 7500/7500 [09:03<00:00, 13.81it/s]
Done, total cost of epoch 20: 54.86490839051625
0/1 error(s) of training set: 10 / 60000
0/1 error(s) of testing set: 91 / 10000
Time used: 597.2083172798157 sec
---------- epoch 20 end ------------
Total time used: 11879.68366599083 sec
Testing on model_data_10.pkl:
error rate: 0.0086
```
## References
1. [Matt Wang's repository](https://github.com/mattwang44/LeNet-from-Scratch)
2. [Classification and Loss Evaluation - Softmax and Cross Entropy Loss by Paras Dahal](https://deepnotes.io/softmax-crossentropy)
3. [The MNIST database of handwritten digits by Yann LeCun, Corinna Cortes and Christopher Burges](http://yann.lecun.com/exdb/mnist/)
