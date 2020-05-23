import numpy as np

# padding for the matrix of images
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, ), (pad, ), (pad, ), (0, )), "constant", constant_values = (0, 0))
    return X_pad

# initialisation of the weights & bias
def initialise(kernel_shape):
    bias_shape = (1, 1, 1, kernel_shape[-1]) if len(kernel_shape) == 4 else (kernel_shape[-1], )
    mu, sigma = 0, 0.01
    weight = np.random.normal(mu, sigma, kernel_shape)
    bias = np.ones(bias_shape) * 0.001
    return weight, bias

# softmax activation function for the output layer
def softmax(X):
    X_softmax = np.exp(X) / np.array([np.sum(np.exp(X), axis = 1)]).T
    return X_softmax

# Convolution Layer
class Conv_Layer(object):
    def __init__(self, kernel_shape, stride = 1, pad = 0):
        self.weight, self.bias = initialise(kernel_shape)
        self.stride = stride
        self.pad = pad
    
    def forward_propagation(self, input_map):
        self.input_map = input_map
        batch_size, height_input, width_input, _ = input_map.shape
        f, _, _, channel_output = self.weight.shape
        height_output = int((height_input + 2 * self.pad - f) / self.stride + 1)
        width_output = int((width_input + 2 * self.pad - f) / self.stride + 1)
        output_map = np.zeros((batch_size, height_output, width_output, channel_output))
        input_map_pad = zero_pad(input_map, self.pad)
        for height in range(height_output):
            for width in range(width_output):
                vertical_start, vertical_end = height * self.stride, height * self.stride + f
                horizontal_start, horizontal_end = width * self.stride, width * self.stride + f
                input_map_slice = input_map_pad[:, vertical_start: vertical_end, horizontal_start: horizontal_end, :]
                output_map[:, height, width, :] = np.tensordot(input_map_slice, self.weight, axes = ([1, 2, 3], [0, 1, 2])) + self.bias
        return output_map
    
    def back_propagation(self, d_output_map, learning_rate):
        f, _, _, channel_output = self.weight.shape
        _, height_output, width_output, channel_output = d_output_map.shape
        d_input_map = np.zeros(self.input_map.shape)
        d_weight = np.zeros(self.weight.shape)
        d_bias = np.zeros((1, 1, 1, channel_output))
        if self.pad != 0:
            input_map_pad = zero_pad(self.input_map, self.pad)
            d_input_map_pad = zero_pad(d_input_map, self.pad)
        else:
            input_map_pad = self.input_map
            d_input_map_pad = d_input_map
        for height in range(height_output):
            for width in range(width_output):
                vertical_start, vertical_end = height * self.stride, height * self.stride + f
                horizontal_start, horizontal_end = width * self.stride, width * self.stride + f
                input_map_slice = input_map_pad[:, vertical_start: vertical_end, horizontal_start: horizontal_end, :]
                d_input_map_pad[:, vertical_start: vertical_end, horizontal_start: horizontal_end, :] += np.transpose(np.dot(self.weight, d_output_map[:, height, width, :].T), (3, 0, 1, 2))
                d_weight += np.dot(np.transpose(input_map_slice, (1, 2, 3, 0)), d_output_map[:, height, width, :])
                d_bias += np.sum(d_output_map[:, height, width, :], axis = 0)
        d_input_map = d_input_map_pad if self.pad == 0 else d_input_map_pad[:, self.pad: -self.pad, self.pad: -self.pad, :]
        self.weight -= learning_rate * d_weight
        self.bias -= learning_rate * d_bias
        self.input_map = None
        return d_input_map

# ReLU Activation Layer
class ReLU_Layer(object):
    def forward_propagation(self, input_map):
        self.input_map = input_map
        return np.where(input_map > 0, input_map, 0)
    
    def back_propagation(self, d_output_map):
        d_input_map = np.multiply(d_output_map, np.where(self.input_map > 0, 1, 0))
        self.input_map = None
        return d_input_map

# Max-Pooling Layer
class MaxPool_Layer(object):
    def __init__(self, stride = 2, f = 2):
        self.stride = stride
        self.f = f

    def forward_propagation(self, input_map):
        self.input_map = input_map
        batch_size, height_input, width_input, channel = input_map.shape
        height_output = int(1 + (height_input - self.f) / self.stride)
        width_output = int(1 + (width_input - self.f) / self.stride)
        output_map = np.zeros((batch_size, height_output, width_output, channel))
        for height in range(height_output):
            for width in range(width_output):
                vertical_start, vertical_end = height * self.stride, height * self.stride + self.f
                horizontal_start, horizontal_end = width * self.stride, width * self.stride + self.f
                input_map_slice = input_map[:, vertical_start: vertical_end, horizontal_start: horizontal_end, :]
                output_map[:, height, width, :] = np.max(input_map_slice, axis = (1, 2))
        return output_map

    def back_propagation(self, d_output_map):
        _, height_output, width_output, _ = d_output_map.shape
        d_input_map = np.zeros(self.input_map.shape)
        for height in range(height_output):
            for width in range(width_output):
                vertical_start, vertical_end = height * self.stride, height * self.stride + self.f
                horizontal_start, horizontal_end = width * self.stride, width * self.stride + self.f
                input_map_slice = self.input_map[:, vertical_start: vertical_end, horizontal_start: horizontal_end, :]
                input_map_slice = np.transpose(input_map_slice, (1, 2, 3, 0))
                mask = input_map_slice == input_map_slice.max((0, 1))
                mask = np.transpose(mask, (3, 2, 0, 1))
                d_input_map[:, vertical_start: vertical_end, horizontal_start: horizontal_end, :] += np.transpose(np.multiply(d_output_map[:, height, width, :][:, :, np.newaxis, np.newaxis], mask), (0, 2, 3, 1))
        self.input_map = None
        return d_input_map

# Fully Connected Layer
class FC_Layer(object):
    def __init__(self, weight_shape):
        self.weight, self.bias = initialise(weight_shape)

    def forward_propagation(self, input_array):
        self.input_array = input_array
        return np.matmul(input_array, self.weight) + self.bias

    def back_propagation(self, d_output_array, learning_rate):
        d_input_array = np.matmul(d_output_array, self.weight.T)
        d_weight = np.matmul(self.input_array.T, d_output_array)
        d_bias = np.sum(d_output_array.T, axis = 1)
        self.weight -= learning_rate * d_weight
        self.bias -= learning_rate * d_bias
        self.input_array = None
        return d_input_array

# Fully Connected Output Layer
class FC_Output_Layer(object):
    def __init__(self, weight_shape):
        self.weight, self.bias = initialise(weight_shape)
    
    def forward_propagation(self, input_array, labels, mode):
        self.input_array = input_array
        self.labels = labels
        self.output_array = np.matmul(input_array, self.weight) + self.bias
        output = softmax(self.output_array)
        predictions = np.argmax(output, axis = 1)
        if mode == "train":
            cost_value = -np.log(output[range(output.shape[0]), labels])
            return np.sum(cost_value)
        elif mode == "test":
            error = np.sum(labels != predictions)
            return error, predictions
    
    def back_propagation(self, learning_rate):
        d_output_array = softmax(self.output_array)
        d_output_array[range(d_output_array.shape[0]), self.labels] -= 1
        d_output_array = d_output_array / d_output_array.shape[0]
        d_input_array = np.matmul(d_output_array, self.weight.T)
        d_weight = np.matmul(self.input_array.T, d_output_array)
        d_bias = np.sum(d_output_array.T, axis = 1)
        self.weight -= learning_rate * d_weight
        self.bias -= learning_rate * d_bias
        self.input_array, self.labels, self.output_array = None, None, None
        return d_input_array
