import torch
torch.set_default_dtype(torch.float64)

# Maps and gradients
# All vectors here are and must be column vectors.
# All tensors here: vectors and arrays must be 2d tensors.
# Given rx: dim of x, ry: dim of y.

rx = 2
ru = 2
ry = 2

# Noise covariances:
Q = 2 * torch.diag(torch.ones(rx,))
R = torch.diag(torch.ones(ry,))

def stateDynamics(x,u):
    x = torch.atleast_1d(x.squeeze())
    u = torch.atleast_1d(u.squeeze())
    f = torch.zeros(rx,)
    f[0] = 0.9 * x[0] + 0.2 * torch.exp(x[1]) + u[0]
    f[1] = 0.96 * x[1] + u[1]
    return torch.atleast_2d(f.squeeze()).T


def measurementDynamics(x, u):
    x = torch.atleast_1d(x.squeeze())
    u = torch.atleast_1d(u.squeeze())
    gx = torch.zeros(ry,)
    gx[0] = x[1] * torch.tanh(0.5*(x[0]-5))
    gx[1] = 1/27 * x[0] ** 3
    return torch.atleast_2d(gx.squeeze()).T

# We follow control theory notation for compactness: f is the stateDynamics function, g is the measurementDynamics function
def f_Jacobian(x, u):
    f_x, _ = torch.autograd.functional.jacobian(stateDynamics, inputs=(x, u))
    return torch.atleast_2d(f_x.squeeze())
def g_Jacobian(x, u):
    g_x, _ = torch.autograd.functional.jacobian(measurementDynamics, inputs=(x, u))
    return torch.atleast_2d(g_x.squeeze())