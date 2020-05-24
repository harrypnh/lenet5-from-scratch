import numpy as np
import matplotlib.pyplot as plt

err_train_8 = [1058 / 60000, 911 / 60000, 321 / 60000, 191 / 60000, 217 / 60000,
               64 / 60000, 36 / 60000, 27 / 60000, 21 / 60000, 15 / 60000,
               12 / 60000, 12 / 60000, 11 / 60000, 11 / 60000, 11 / 60000,
               11 / 60000, 11 / 60000, 11 / 60000, 11 / 60000, 11 / 60000]

err_test_8 = [182 / 10000, 174 / 10000, 109 / 10000, 89 / 10000, 110 / 10000,
              84 / 10000, 81 / 10000, 82 / 10000, 78 / 10000, 81 / 10000,
              82 / 10000, 82 / 10000, 81 / 10000, 81 / 10000, 82 / 10000,
              81 / 10000, 81 / 10000, 83 / 10000, 83 / 10000, 83 / 10000]

err_train_custom_8 = [974 / 60000, 459 / 60000, 155 / 60000, 80 / 60000, 120 / 60000,
                       19 / 60000, 13 / 60000, 8 / 60000, 5 / 60000, 4 / 60000,
                       3 / 60000, 3 / 60000, 3 / 60000, 3 / 60000, 3 / 60000,
                       3 / 60000, 3 / 60000, 3 / 60000, 3 / 60000, 3 / 60000]

err_test_custom_8 = [179 / 10000, 104 / 10000, 81 / 10000, 70 / 10000, 78 / 10000,
                      60 / 10000, 60 / 10000, 58 / 10000, 56 / 10000, 58 / 10000,
                      62 / 10000, 62 / 10000, 62 / 10000, 62 / 10000, 62 / 10000,
                      62 / 10000, 62 / 10000, 62 / 10000, 62 / 10000, 62 / 10000]

epoches = 20
plt.xlabel("epoches")
plt.ylabel("error rate")
plt.plot(np.arange(1, epoches + 1), err_train_8)
plt.plot(np.arange(1, epoches + 1), err_test_8)
plt.legend(["training data",
            "testing data"], loc = "upper right")
plt.xticks(range(1, epoches + 1))
plt.title("LeNet-5 Error Rate during Training (batch size = 8)")
plt.savefig("figure_lenet5.png")
plt.clf()

plt.xlabel("epoches")
plt.ylabel("error rate")
plt.plot(np.arange(1, epoches + 1), err_train_custom_8)
plt.plot(np.arange(1, epoches + 1), err_test_custom_8)
plt.legend(["training data",
            "testing data"], loc = "upper right")
plt.xticks(range(1, epoches + 1))
plt.title("CustomCNN Error Rate during Training (batch size = 8)")
plt.savefig("figure_custom_cnn.png")
