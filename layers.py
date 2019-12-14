import torch
import torch.nn as nn
import torch.nn.functional as F


def get_vector_norm(x):
    return torch.norm(x, dim=-1, keepdim=True)


def squash(x):
    norm = get_vector_norm(x)

    norm_square = norm ** 2
    numerator =  norm_square/(1 + norm)
    denominator = (1 + norm_square)

    scaling_factor = numerator/denominator

    return scaling_factor * (x / norm)
    

class ConvLayer(nn.Module):

    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):

        super().__init__()

        self.layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1
        )

    def forward(self, x):
        return self.layer(x)


class PrimaryCapsLayer(nn.Module):

    def __init__(self, output_dim=8, num_capsules=32, in_channels=256, kernel_size=9, stride=2):

        super().__init__()

        self.output_dim = output_dim
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=output_dim*num_capsules,
            kernel_size=kernel_size,
            stride=stride
        )
    
    def forward(self, x):

        op = self.conv_layer(x)
        batch_size = x.shape[0]

        op = squash(op)

        return op.view(batch_size, -1, self.output_dim)


class DigiCapsLayer(nn.Module):

    def __init__(self, num_capsules=10, input_dim=8, output_dim=16, num_iterations=3):
        super().__init__()

        self.num_capsules = num_capsules
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.num_iterations = num_iterations

        self.transform_dim = nn.Linear(
            in_features=input_dim,
            out_features=output_dim
        )
    

    def forward(self,u):

        # Modify the input from dim 8 to dim 16
        u = self.transform_dim(u)

        # Get bs and cs
        bs = torch.zeros(
            u.shape[0], u.shape[1], self.num_capsules
        )
        print("b shape", bs.shape)
        # Dynamic Routing
        
        for i in range(self.num_iterations):
            cs = torch.softmax(bs, dim=1)
            print("C shape", cs.shape)
            s = torch.matmul(u.transpose(2,1),cs)
            v = squash(s)

            print("U shape", u.shape)
            print("V shape", v.shape)
            if i < self.num_iterations -1:
                print("B loop shape", bs.shape)
                bs += torch.matmul(u,v)

        return v.transpose(2,1)


class Decoder(nn.Module):
    
    def __init__(self, out_features, in_features):
        super().__init__()

        self.out_features = out_features
        self.in_features = in_features

        self.layer = nn.Sequential([
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.out_features),
            nn.Sigmoid()
        ])

    def forward(self, x, mask):
        masked_input = mask * x
        


