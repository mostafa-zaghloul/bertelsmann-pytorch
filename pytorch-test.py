import torch


def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))


### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

features = torch.randn((1, 3))
n_input = features.shape[1]
n_hidden = 2
n_out = 1

w1 = torch.randn(n_input,n_hidden)
w2 = torch.randn(n_hidden,n_out)
b1 = torch.randn(1,n_hidden)
b2 = torch.randn(1,n_out)

h = activation(torch.mm(features,w1) + b1)
output = activation(torch.mm(h,w2) + b2)
print(output)
