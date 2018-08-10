from PRUTransforms import *

__author__ = "Sachin Mehta"
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Sachin Mehta"

class PRU(nn.Module):
    '''
    This class implements the Pyramidal recurrent unit with LSTM gating structure.
    x_t is processed using pyramidal transform while h_{t-1} is processed using grouped linear transform.
    Note that this will be slower than LSTM because it does not use cuDNN.
    '''
    def __init__(self, input_size, hidden_size, k=3, groups=3, **kwargs):
        super(PRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pt = PyramidalTransform(input_size, hidden_size, k)
        self.glt = GroupedLinear(hidden_size, groups=groups)

    def forward(self, input, hidden):
        def recurrence(input_, hx):
            """Recurrence helper."""
            h_0, c_0 = hx[0], hx[1]

            # input vector is processed by Pyramidal Transform
            i2h_f, i2h_g, i2h_i, i2h_o = self.pt(input_)
            # previous hidden state is processed by the Grouped Linear Transform
            h2h_f, h2h_g, h2h_i, h2h_o = self.glt(h_0)

            # input to LSTM gates
            f = i2h_f + h2h_f
            g = i2h_g + h2h_g
            i = i2h_i + h2h_i
            o = i2h_o + h2h_o

            # outputs
            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
            return h_1, c_1

        input = input.transpose(0, 1)# batch is always first
        output = []
        steps = range(input.size(1))
        for i in steps:
            size_inp = input[:,i].size()
            input_t = input[:, i].view(size_inp[0], 1, size_inp[1]) # make input as
            hidden = recurrence(input_t, hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.stack(output, 1) # stack the all output tensors, so that dims is 1 X Se X B X D
        output = torch.squeeze(output, 0) #remove the first dummy dim
        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)