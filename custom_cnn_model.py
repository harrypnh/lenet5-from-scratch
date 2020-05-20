import numpy as np
from cnn_layers import Conv_Layer, ReLU_Layer, MaxPool_Layer, FC_Layer, FC_Output_Layer

# CustomCNN Model
class CustomCNN(object):
    def __init__(self):
        kernel_shape = {"C1": (1, 1, 1, 8),
                        "C3": (1, 1, 8, 16),
                        "C5": (1, 1, 16, 32),
                        "C7": (1, 1, 32, 64),
                        "C9": (1, 1, 64, 128),
                        "C11": (1, 1, 128, 256),
                        "F12": (256, 512),
                        "F13": (512, 128)
                        "F14": (128, 10)}
        self.C1 = Conv_Layer(kernel_shape["C1"])
        self.ReLU1 = ReLU_Layer()
        self.S2 = MaxPool_Layer()
        self.C3 = Conv_Layer(kernel_shape["C3"])
        self.ReLU2 = ReLU_Layer()
        self.S4 = MaxPool_Layer()
        self.C5 = Conv_Layer(kernel_shape["C5"])
        self.ReLU3 = ReLU_Layer()
        self.S6 = MaxPool_Layer()
        self.C7 = Conv_Layer(kernel_shape["C7"])
        self.ReLU4 = ReLU_Layer()
        self.S8 = MaxPool_Layer()
        self.C9 = Conv_Layer(kernel_shape["C9"])
        self.ReLU5 = ReLU_Layer()
        self.S10 = MaxPool_Layer()
        self.C11 = Conv_Layer(kernel_shape["C11"])
        self.ReLU6 = ReLU_Layer()
        self.F12 = FC_Layer(kernel_shape["F12"])
        self.ReLU7 = ReLU_Layer()
        self.F13 = FC_Layer(kernel_shape["F13"])
        self.ReLU8 = ReLU_Layer()
        self.F14 = FC_Output_Layer(kernel_shape["F14"])

    def forward_propagation(self, input_image, input_label, mode):
        C1_FP = self.C1.forward_propagation(input_image)
        ReLU1_FP = self.ReLU1.forward_propagation(C1_FP)
        S2_FP = self.S2.forward_propagation(ReLU1_FP)
        C3_FP = self.C3.forward_propagation(S2_FP)
        ReLU2_FP = self.ReLU2.forward_propagation(C3_FP)
        S4_FP = self.S4.forward_propagation(ReLU2_FP)
        C5_FP = self.C5.forward_propagation(S4_FP)
        ReLU3_FP = self.ReLU3.forward_propagation(C5_FP)
        S6_FP = self.S6.forward_propagation(ReLU3_FP)
        C7_FP = self.C7.forward_propagation(S6_FP)
        ReLU4_FP = self.ReLU4.forward_propagation(C7_FP)
        S8_FP = self.S8.forward_propagation(ReLU4_FP)
        C9_FP = self.C9.forward_propagation(S8_FP)
        ReLU5_FP = self.ReLU5.forward_propagation(C9_FP)
        S10_FP = self.S10.forward_propagation(ReLU5_FP)
        C11_FP = self.C11.forward_propagation(S10_FP)
        ReLU6_FP = self.ReLU6.forward_propagation(C11_FP)
        ReLU6_FP = ReLU6_FP[:, 0, 0, :]
        F12_FP = self.F12.forward_propagation(ReLU6_FP)
        ReLU7_FP = self.ReLU7.forward_propagation(F12_FP)
        F13_FP = self.F13.forward_propagation(ReLU7_FP)
        ReLU8_FP = self.ReLU8.forward_propagation(F13_FP)
        return self.F14.forward_propagation(ReLU8_FP, input_label, mode)

    def back_propagation(self, learning_rate):
        F14_BP = self.F14.back_propagation(learning_rate)
        ReLU8_BP = self.ReLU8.back_propagation(F14_BP)
        F13_BP = self.F13.back_propagation(ReLU8_BP, learning_rate)
        ReLU7_BP = self.ReLU7.back_propagation(F13_BP)
        F12_BP = self.F12.back_propagation(ReLU7_BP, learning_rate)
        F12_BP = F12_BP[:, np.newaxis, np.newaxis, :]
        ReLU6_BP = self.ReLU6.back_propagation(F12_BP)
        C11_BP = self.C11.back_propagation(ReLU6_BP, learning_rate)
        S10_BP = self.S10.back_propagation(C11_BP)
        ReLU5_BP = self.ReLU5.back_propagation(S10_BP)
        C9_BP = self.C9.back_propagation(ReLU5_BP, learning_rate)
        S8_BP = self.S8.back_propagation(C9_BP)
        ReLU4_BP = self.ReLU4.back_propagation(S8_BP)
        C7_BP = self.C7.back_propagation(ReLU4_BP, learning_rate)
        S6_BP = self.S6.back_propagation(C7_BP)
        ReLU3_BP = self.ReLU3.back_propagation(S6_BP)
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
        temp_model.S6.stride = self.S6.stride
        temp_model.S6.f = self.S6.f
        temp_model.C7.weight = self.C7.weight
        temp_model.C7.bias = self.C7.bias
        temp_model.C7.stride = self.C7.stride
        temp_model.C7.pad = self.C7.pad
        temp_model.S8.stride = self.S8.stride
        temp_model.S8.f = self.S8.f
        temp_model.C9.weight = self.C9.weight
        temp_model.C9.bias = self.C9.bias
        temp_model.C9.stride = self.C9.stride
        temp_model.C9.pad = self.C9.pad
        temp_model.S10.stride = self.S10.stride
        temp_model.S10.f = self.S10.f
        temp_model.C11.weight = self.C11.weight
        temp_model.C11.bias = self.C11.bias
        temp_model.C11.stride = self.C11.stride
        temp_model.C11.pad = self.C11.pad
        temp_model.F12.weight = self.F12.weight
        temp_model.F12.bias = self.F12.bias
        temp_model.F13.weight = self.F13.weight
        temp_model.F13.bias = self.F13.bias
        temp_model.F14.weight = self.F14.weight
        temp_model.F14.bias = self.F14.bias
        return temp_model
