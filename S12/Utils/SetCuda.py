# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:01:39 2020

@author: 20115260
"""

import torch


def set_seed(seed, cuda):
    """ Setting the seed makes the results reproducible. """
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def initialize_cuda(seed):
    """ Check if GPU is availabe and set seed. """

    # Check CUDA availability
    cuda = torch.cuda.is_available()
    print('GPU Available?', cuda)

    # Initialize seed
    set_seed(seed, cuda)

    # Set device
    device = torch.device("cuda" if cuda else "cpu")

    return cuda, device
