import numpy as np
import matplotlib.pyplot as plt

err_train_8 = [1337 / 60000, 679 / 60000, 273 / 60000, 176 / 60000, 135 / 60000,
               55 / 60000, 40 / 60000, 28 / 60000, 13 / 60000, 12 / 60000,
               10 / 60000, 11 / 60000, 10 / 60000, 10 / 60000, 10 / 60000,
               10 / 60000, 10 / 60000, 10 / 60000, 10 / 60000, 10 / 60000]

err_test_8 = [240 / 10000, 158 / 10000, 102 / 10000, 100 / 10000, 97 / 10000,
              93 / 10000, 92 / 10000, 88 / 10000, 89 / 10000, 86 / 10000,
              90 / 10000, 90 / 10000, 91 / 10000, 92 / 10000, 93 / 10000,
              91 / 10000, 91 / 10000, 91 / 10000, 91 / 10000, 91 / 10000]
epoches = 20
epoches = np.arange(1, epoches + 1)
plt.xlabel("epoches")
plt.ylabel("error rate")
plt.plot(epoches, err_train_8)
plt.plot(epoches, err_test_8)
plt.legend(["training data (batch size = 8)",
            "testing data (batch size = 8)"], loc = "upper right")
plt.xticks(range(1, 20 + 1))
plt.savefig("images/figure_batch_size_8.png")