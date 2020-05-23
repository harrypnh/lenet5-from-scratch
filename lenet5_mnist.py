import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

from utils.lenet5_model import LeNet5
from tqdm import tqdm
import struct
import math

# read the images and labels
def readDataset(dataset_path):
    (image_path, label_path) = dataset_path
    with open(label_path, "rb") as label_file:
        magic, dataset_size = struct.unpack(">II", label_file.read(8))
        label_dataset = np.fromfile(label_file, dtype = np.int8)
    with open(image_path, "rb") as image_file:
        magic, dataset_size, rows, columns = struct.unpack(">IIII", image_file.read(16))
        image_dataset = np.fromfile(image_file, dtype = np.uint8).reshape(len(label_dataset), rows, columns)
    return (image_dataset, label_dataset)

# padding for the matrix of images
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, ), (pad, ), (pad, ), (0, )), "constant", constant_values = (0, 0))
    return X_pad

# normalise the dataset
def normalise(image):
    image -= image.min()
    image = image / image.max()
    image = (image - np.mean(image)) / np.std(image)
    return image

# generate random-shuffled mini-batches
def random_mini_batches(image, label, mini_batch_size = 256, one_batch = False):
    dataset_size = image.shape[0] # number of training examples
    mini_batches = []
    # shuffle (image, label)
    permutation = list(np.random.permutation(dataset_size))
    shuffled_image = image[permutation, :, :, :]
    shuffled_label = label[permutation]
    # extract only one batch
    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return (mini_batch_image, mini_batch_label)
    # partition (shuffled_image, shuffled_label). Minus the end case.
    complete_minibatches_number = math.floor(dataset_size / mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, complete_minibatches_number):
        mini_batch_image = shuffled_image[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_label = shuffled_label[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    # handle the end case (last mini-batch < mini_batch_size)
    if dataset_size % mini_batch_size != 0:
        mini_batch_image = shuffled_image[complete_minibatches_number * mini_batch_size: dataset_size, :, :, :]
        mini_batch_label = shuffled_label[complete_minibatches_number * mini_batch_size: dataset_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    return mini_batches

def load_dataset(test_image_path, test_label_path, train_image_path, train_label_path):
    train_dataset = (train_image_path, train_label_path)
    test_dataset = (test_image_path, test_label_path)
    # read data
    train_image, train_label = readDataset(train_dataset)
    test_image, test_label = readDataset(test_dataset)
    # data preprocessing
    train_image_normalised_pad = normalise(zero_pad(train_image[:, :, :, np.newaxis], 2))
    test_image_normalised_pad = normalise(zero_pad(test_image[:, :, :, np.newaxis], 2))
    return (train_image_normalised_pad, train_label), (test_image_normalised_pad, test_label)

def train(model, train_data, test_data, epoches, learning_rate_list, batch_size):
    # training loops
    start_time = time.time()
    error_rate_list = []
    for epoch in range(0, epoches):
        print("---------- epoch", epoch + 1, "begin ----------")
        learning_rate = learning_rate_list[epoch]
        # print information
        print("learning rate: {}".format(learning_rate))
        print("batch size: {}".format(batch_size))
        # loop over each batch
        start_time_epoch = time.time()
        cost = 0
        mini_batches = random_mini_batches(train_data[0], train_data[1], batch_size)
        print("Training:")
        for i in tqdm(range(len(mini_batches))):
            batch_image, batch_label = mini_batches[i]
            loss = model.forward_propagation(batch_image, batch_label, 'train')
            cost += loss
            model.back_propagation(learning_rate)
        print("Done, total cost of epoch {}: {}".format(epoch + 1, cost))
        error_train, _ = model.forward_propagation(train_data[0], train_data[1], 'test')
        error_test, _ = model.forward_propagation(test_data[0], test_data[1], 'test')
        error_rate_list.append([error_train / 60000, error_test / 10000])
        print("0/1 error(s) of training set:", error_train, "/", len(train_data[1]))
        print("0/1 error(s) of testing set:", error_test, "/", len(test_data[1]))
        print("Time used:", time.time() - start_time_epoch, "sec")
        print("---------- epoch", epoch + 1, "end ------------")
        with open("model_data/lenet5_data_" + str(epoch + 1) + ".pkl", "wb") as output:
            pickle.dump(model.extract_model(), output, pickle.HIGHEST_PROTOCOL)
    error_rate_list = np.array(error_rate_list).T
    print("Total time used:", time.time() - start_time, "sec")
    return error_rate_list

def test(model_path, test_data):
    # read model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    print("Testing with {}:".format(model_path))
    errors, predictions = model.forward_propagation(test_data[0], test_data[1], "test")
    print("error rate:", errors / len(predictions))

test_image_path = "dataset/MNIST/t10k-images-idx3-ubyte"
test_label_path = "dataset/MNIST/t10k-labels-idx1-ubyte"
train_image_path = "dataset/MNIST/train-images-idx3-ubyte"
train_label_path = "dataset/MNIST/train-labels-idx1-ubyte"
batch_size = 8
epoches = 20
learning_rate_list = np.array([5e-2] * 2 + [2e-2] * 3 + [1e-2] * 3 + [5e-3] * 4 + [1e-3] * 4 + [5e-4] * 4)
# model_path = "model_data/lenet5_data_0.78.pkl"

train_data, test_data = load_dataset(test_image_path, test_label_path, train_image_path, train_label_path)
model = LeNet5()
error_rate_list = train(model, train_data, test_data, epoches, learning_rate_list, batch_size)
# test(model_path, test_data)
test("model_data/lenet5_data_" + str(error_rate_list[1].argmin() + 1) + ".pkl", test_data)
x = np.arange(1, epoches + 1)
plt.xlabel("epoches")
plt.ylabel("error rate")
plt.plot(x, error_rate_list[0])
plt.plot(x, error_rate_list[1])
plt.legend(["training data", "testing data"], loc = "upper right")
plt.show()
