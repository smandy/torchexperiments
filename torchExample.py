
import torch

if 0:
    z = torch.zeros(5,3)
    print(z)
    print(z.dtype)

def exp_adder(x,y,z):
    return 2 * x.exp() + 3 * y

inputs = (torch.rand(1), torch.rand(1), torch.rand(1))

print(inputs)

print( 2 * inputs[0].exp() )

print(torch.autograd.functional.jacobian(exp_adder, inputs))

a = torch.Tensor( [1,2,3])
b = torch.Tensor( [2,3,4])

print(a.cross(b))

