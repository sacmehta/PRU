import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

__author__ = "Sachin Mehta"
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Sachin Mehta"

class PyramidalTransform(nn.Module):
    '''
    This class implements the pyramidal transform
    '''
    def __init__(self, ninp, nhid, k=3):
        super(PyramidalTransform, self).__init__()
        assert nhid % pow(2, k) == 0, 'Output dimensions should be divisible by number of pyramid levels'
        assert ninp % pow(2, k) == 0, 'Input dimensions should be divisible by number of pyramid levels'

        self.inputSize = ninp
        self.outSize = nhid
        self.k = k

        self.kernelSize = [1] * (k)
        kern = 3
        for i in reversed(range(1, k)):
            self.kernelSize[i] = kern
            kern +=2
        self.padding = [int((size - 1)/2) for size in self.kernelSize]

        # dimnesions for the input vector (pyramid shape)
        inpDims = list() #self.sizeList(ninp)
        a = np.zeros(ninp, dtype=bool)
        for i in range(k):
            p = 2 ** i
            inpDims.append(len(a[::p]))
        del a

        # dimnesions for the output vector (pyramid shape)
        a = np.zeros((nhid), dtype=bool)
        outDims = list()
        for i in range(k):
            p = 2 ** (i+1)
            outDims.append(len(a[::p]))
        outDims[0] += int(nhid - np.sum(outDims))
        del a

        self.pyramid = nn.ModuleList()
        for i in range(k):
            self.pyramid.append(nn.Linear(inpDims[i], 4 * outDims[i])) # Multiplication with 4 because LSTM has 4 gates

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        for layer in self.pyramid:
            init.orthogonal(layer.weight)
            init.constant(layer.bias, val=0)

    def forward(self, input):
        pyr_out2 = None
        pyr_out3 = None
        pyr_out4 = None
        in_copy = None

        for i, layer in enumerate(self.pyramid):
            if i > 0:
                # down-sample the input
                input = F.avg_pool1d(input, kernel_size=self.kernelSize[i], stride=2,
                                     padding=self.padding[i])
            in2 = input.view(input.size(0) * input.size(1), input.size(2))
            if i == 0:
                in_copy = in2.clone()
                #transform and chunk it into 4 outputs, each for the 4 gates of LSTM
                pyr_out1, pyr_out2, pyr_out3, pyr_out4 = torch.chunk(layer(in2), 4, dim=1)
            else:
                # transform and chunk it into 4 outputs, each for the 4 gates of LSTM
                val1, val2, val3, val4 = torch.chunk(layer(in2), 4, dim=1)
                pyr_out1 = torch.cat([pyr_out1, val1], 1)
                pyr_out2 = torch.cat([pyr_out2, val2], 1)
                pyr_out3 = torch.cat([pyr_out3, val3], 1)
                pyr_out4 = torch.cat([pyr_out4, val4], 1)
                del val1, val2, val3, val4
        # residual link
        if in_copy.size() == pyr_out1.size():
            pyr_out1 = pyr_out1 + in_copy
            pyr_out2 = pyr_out2 + in_copy
            pyr_out3 = pyr_out3 + in_copy
            pyr_out4 = pyr_out4 + in_copy
        return pyr_out1, pyr_out2, pyr_out3, pyr_out4

    def __repr__(self):
        s = '{name}({inputSize}, {outSize}, {k})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class GroupedLinear(nn.Module):
    '''
        This class implements the Grouped Linear Transform
    '''
    def __init__(self, nInp, groups=1):
        super(GroupedLinear, self).__init__()
        # print(nInp, groups)
        assert (nInp % groups == 0), "Input dimensions must be divisible by groups"
        self.nInp = nInp
        self.outDim = int(nInp / groups)
        self.nOut = self.outDim
        self.groups = groups
        self.W = torch.nn.Parameter(
            torch.Tensor(groups, self.outDim, 4 * self.outDim))  # Multiplication with 4 because LSTM has 4 gates
        self.bias = torch.nn.Parameter(torch.FloatTensor(groups, 4 * self.outDim))
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Initialize params
        '''
        init.orthogonal(self.W.data)
        init.constant(self.bias.data, val=0)

    def forward(self, input):# input should be of the form bsz x nInp
        '''
        We make use of batch matrix multiplication (BMM) in GLT. BMM takes mat1 (b x n x m) and mat2 (b x m x p)
        as input and produces as an output of dimensions b x n x p. To be computationally efficient, we set b as
        number of groups, n as batch size and p as output dims.
        To minimize the memory alignment operations, we assume that batch-size is along second dimension of the
        input vector i.e input vector is of size 1 x bSz x nInp. Therefore, we can concat the input vector obtained
        after grouping (or split) operation along the first dimension (represented by index 0). This way we minimize
        the vector alignment operations which might be computationally expensive in 3-D and 4-D dimensional vector space.
        '''
        bsz = input.size(1) #batch is a second dimension
        # reshape the input so that size is g x BSz X nInp
        input_res_tupple = torch.split(input, self.outDim, dim=2)#torch.chunk(input, self.groups, dim=1)
        input_res = torch.cat(input_res_tupple, 0)
        del input, input_res_tupple
        out = torch.bmm(input_res, self.W)  # multiply with Weights
        out = out.transpose(0, 1) # so that batch is last
        out = torch.add(out, self.bias)
        out1, out2, out3, out4 = torch.split(out, self.outDim, dim=2) #split for 4 gates

        # Reshape the outputs
        out1 = out1.contiguous().view(bsz, -1)
        out2 = out2.contiguous().view(bsz, -1)
        out3 = out3.contiguous().view(bsz, -1)
        out4 = out4.contiguous().view(bsz, -1)

        return out1, out2, out3, out4

    def __repr__(self):
        s = '{name}({nInp}, {nInp}, {groups})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
