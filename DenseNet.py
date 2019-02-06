import torch
import torch.nn as nn

# TODO Include dropout
# Implemented DenseNet from https://arxiv.org/pdf/1608.06993.pdf

# DenseBlock input should have k layers input and k layers output
class DenseBlock(nn.Module):
    def __init__(self, n_layers, k):
        super(DenseBlock, self).__init__()
        self.activation = nn.ReLU()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        # Each layer in the block takes in (i+1) * k inputs and produces k outputs
        for i in range(n_layers):
            batch_norm = nn.BatchNorm2d(num_features=(i+1)*k).cuda()
            bottleneck = nn.Conv2d((i+1)*k, k, kernel_size=1, stride=1, padding=0).cuda()
            conv = nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1).cuda()
            self.layers.extend([batch_norm, self.activation, bottleneck, conv])

    def forward(self, x):
        state = x
        i = 0
        while i < self.n_layers:
            l1 = self.layers[i](state)
            l2 = self.layers[i+1](l1)
            l3 = self.layers[i+2](l2)
            l4 = self.layers[i+3](l3)
            state = torch.cat([state, l4], dim=1)
            i += 4

        return state


# DenseNet Model
class DenseNet(nn.Module):
    def __init__(self, input_size=28, output_size=10, n_layers=4, n_dense_blocks=2, k=8, k0=1, type="nominal"):
        super(DenseNet, self).__init__()
        self.k = k
        self.k0 = k0
        self.input_filters = k0+k
        self.dense_size = input_size
        self.n_dense_blocks = n_dense_blocks
        self.type = type

        initial_conv = nn.Conv2d(k0, k, kernel_size=1, stride=1, padding=0).cuda()
        self.layers = nn.ModuleList([initial_conv])

        for i in range(self.n_dense_blocks):
            self.dense_block = DenseBlock(n_layers=n_layers, k=k)
            self.transition_conv = nn.Conv2d(k*2, k, kernel_size=1, stride=1, padding=0).cuda()
            self.transition_pool = nn.AvgPool2d(kernel_size=2, stride=2).cuda()
            self.dense_size = int(self.dense_size/2)
            self.layers.extend([self.dense_block, self.transition_conv, self.transition_pool])

        self.semi_final_fc = nn.Linear(int(((input_size/(2**n_dense_blocks))**2)*k), output_size*8).cuda()
        self.final_fc = nn.Linear(output_size*8, output_size).cuda()
        self.softmax = nn.Softmax()

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        semi_fc = self.semi_final_fc(x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]))
        output = self.final_fc(semi_fc)

        if(self.type is "nominal"):
            output = self.softmax(output)
        return output









