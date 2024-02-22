import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from infras.fno_utilities import *
import numpy as np


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO1d_Emsemble_vnet(nn.Module):
    def __init__(self, modes, width, fids_list):
        super(FNO1d_Emsemble_vnet, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8  # pad the domain if input is non-periodic
        self.fids_list = fids_list

        # self.p = nn.Linear(4, self.width)  # input channel_dim is 2: (u0(x), x)
        self.p = nn.Linear(3, self.width)  # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 2)  # output channel_dim is 1: u1(x)

        self.vnet_conv_list = nn.ModuleList([MLP(self.width, 1, self.width * 2) for fid in self.fids_list])
        self.vnet_fc_list = nn.ModuleList()
        for fid in self.fids_list:
            self.vnet_fc_list.append(DenseNet([fid, 128, 128, 1], nonlinearity=torch.nn.ReLU))


        # self.vnet_conv = MLP(self.width, 1, self.width * 2)
        # # self.vnet_fc = DenseNet(self.width, 1, self.width * 2)
        # print(self.vnet_conv)


    def forward(self, x, fid):
        grid = self.get_grid(x.shape, x.device)
        # cprint('p', grid.shape)
        # cprint('p', x.shape)
        # pe_x = self.get_pe_onehot(x.shape, fid, x.device)
        pe_x = self.get_pe_tri(x.shape, fid, self.fids_list[-1], x.device)
        # cprint('p', grid.shape)
        # cprint('p', pe_x.shape)
        x = torch.cat((x, grid, pe_x), dim=-1)
        # x = torch.cat((x, grid), dim=-1)
        # cprint('g', x.shape)
        # print(self.p)
        # print(self.p.weight.shape)
        # print(self.p.bias.shape)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2


        rho_x = self.vnet_conv_list[self.fids_list.index(fid)](x)
        rho_x = self.vnet_fc_list[self.fids_list.index(fid)](rho_x.view(rho_x.shape[0],-1))
        x = self.q(x)

        x = x.permute(0, 2, 1)

        # cprint('r', x.shape)
        # cprint('b', rho_x.shape)

        return x, rho_x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def get_pe_onehot(self, shape, fid, device):
        # cprint('g', fid)
        batchsize, size_x = shape[0], shape[1]
        # pe_x = (self.fids_list.index(fid)*torch.ones(size_x)).to(float)
        # pe_x = torch.tensor(np.ones(size_x)*self.fids_list.index(fid), dtype=torch.float)
        # pe_x = pe_x.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        # cprint('g', pe_x.shape)
        # # print(pe_x)
        # return pe_x.to(device)

        pe_x_1 = torch.tensor(np.ones(size_x), dtype=torch.float)
        pe_x_1 = pe_x_1.reshape(1, size_x, 1).repeat([batchsize, 1, 1])

        pe_x_2 = torch.tensor(np.zeros(size_x), dtype=torch.float)
        pe_x_2 = pe_x_2.reshape(1, size_x, 1).repeat([batchsize, 1, 1])

        if fid == 33:
            pe_x = torch.cat((pe_x_1, pe_x_2), dim=-1)
        elif fid == 129:
            pe_x = torch.cat((pe_x_2, pe_x_1), dim=-1)
        else:
            raise Exception('error')

        return pe_x.to(device)



    # def get_pe_onehot(self, shape, fid, device):
    #     # cprint('g', fid)
    #     batchsize, size_x = shape[0], shape[1]
    #     # pe_x = (self.fids_list.index(fid)*torch.ones(size_x)).to(float)
    #
    #     pe_x = pe_x.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
    #     # cprint('g', pe_x.shape)
    #     # print(pe_x)
    #     return pe_x.to(device)


    def get_pe_tri(self, shape, fid, fid_hf, device):
        # cprint('g', fid)
        batchsize, size_x = shape[0], shape[1]

        # cprint('r', fid)
        # cprint('g', fid_hf)

        pe = torch.tensor(np.arange(size_x), dtype=torch.float)
        pe = torch.sin(fid/(10000 ** (pe/fid_hf)))
        # print(pe)

        pe_x = pe.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        # print(pe_x.shape)
        return pe_x.to(device)



