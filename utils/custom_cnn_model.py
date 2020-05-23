import numpy as np
from utils.cnn_layers import Conv_Layer, ReLU_Layer, MaxPool_Layer, FC_Layer, FC_Output_Layer

# CustomCNN Model
class CustomCNN(object):
    def __init__(self):
        kernel_shape = {"C1": (5, 5, 1, 32),
                        "C3": (5, 5, 32, 64),
                        "C5": (5, 5, 64, 128),
                        "F6": (128, 256),
                        "F7": (256, 10)}
        self.C1 = Conv_Layer(kernel_shape["C1"], pad = 2)
        self.ReLU1 = ReLU_Layer()
        self.S2 = MaxPool_Layer()
        self.C3 = Conv_Layer(kernel_shape["C3"])
        self.ReLU2 = ReLU_Layer()
        self.S4 = MaxPool_Layer()
        self.C5 = Conv_Layer(kernel_shape["C5"])
        self.ReLU3 = ReLU_Layer()
        self.F6 = FC_Layer(kernel_shape["F6"])
        self.ReLU4 = ReLU_Layer()
        self.F7 = FC_Output_Layer(kernel_shape["F7"])

    def forward_propagation(self, input_image, input_label, mode):
        C1_FP = self.C1.forward_propagation(input_image)
        ReLU1_FP = self.ReLU1.forward_propagation(C1_FP)
        S2_FP = self.S2.forward_propagation(ReLU1_FP)
        C3_FP = self.C3.forward_propagation(S2_FP)
        ReLU2_FP = self.ReLU2.forward_propagation(C3_FP)
        S4_FP = self.S4.forward_propagation(ReLU2_FP)
        C5_FP = self.C5.forward_propagation(S4_FP)
        ReLU3_FP = self.ReLU3.forward_propagation(C5_FP)
        ReLU3_FP = ReLU3_FP[:, 0, 0, :]
        F6_FP = self.F6.forward_propagation(ReLU3_FP)
        ReLU4_FP = self.ReLU4.forward_propagation(F6_FP)
        return self.F7.forward_propagation(ReLU4_FP, input_label, mode)

    def back_propagation(self, learning_rate):
        F7_BP = self.F7.back_propagation(learning_rate)
        ReLU4_BP = self.ReLU4.back_propagation(F7_BP)
        F6_BP = self.F6.back_propagation(ReLU4_BP, learning_rate)
        F6_BP = F6_BP[:, np.newaxis, np.newaxis, :]
        ReLU3_BP = self.ReLU3.back_propagation(F6_BP)
        C5_BP = self.C5.back_propagation(ReLU3_BP, learning_rate)
        S4_BP = self.S4.back_propagation(C5_BP)
        ReLU2_BP = self.ReLU2.back_propagation(S4_BP)
        C3_BP = self.C3.back_propagation(ReLU2_BP, learning_rate)
        S2_BP = self.S2.back_propagation(C3_BP)
        ReLU1_BP = self.ReLU1.back_propagation(S2_BP)
        self.C1.back_propagation(ReLU1_BP, learning_rate)
    
    def extract_model(self):
        temp_model = CustomCNN()
        temp_model.C1.weight = self.C1.weight
        temp_model.C1.bias = self.C1.bias
        temp_model.C1.stride = self.C1.stride
        temp_model.C1.pad = self.C1.pad
        temp_model.S2.stride = self.S2.stride
        temp_model.S2.f = self.S2.f
        temp_model.C3.weight = self.C3.weight
        temp_model.C3.bias = self.C3.bias
        temp_model.C3.stride = self.C3.stride
        temp_model.C3.pad = self.C3.pad
        temp_model.S4.stride = self.S4.stride
        temp_model.S4.f = self.S4.f
        temp_model.C5.weight = self.C5.weight
        temp_model.C5.bias = self.C5.bias
        temp_model.C5.stride = self.C5.stride
        temp_model.C5.pad = self.C5.pad
        temp_model.F6.weight = self.F6.weight
        temp_model.F6.bias = self.F6.bias
        temp_model.F7.weight = self.F7.weight
        temp_model.F7.bias = self.F7.bias
        return temp_model
