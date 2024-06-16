import torch
import matplotlib
from pan_integration.core.solvers import PanSolver

def Lorenz(input):
    dxdt = input[...,0]
    dydt - input[...,1]
