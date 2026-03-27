import numpy as np 
import torch 
import math

def kovasznay_solution(x, y, Re = 20):
    pi = math.pi
    lmbda = Re/2 - math.sqrt((Re**2)/4  + 4*pi**2)

    u = 1 - torch.exp(lmbda*x)*torch.cos(2*pi*y)
    v = (lmbda/(2*pi))*torch.exp(lmbda*x)*torch.sin(2*pi*y)
    p = 0.5*(1 - torch.exp(2*lmbda*x))

    return u, v, p