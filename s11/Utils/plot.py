# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:03:46 2020

@author: Anjan 
"""

import matplotlib.pyplot as plt

def plot_metric(values, metric ='test Loss'):
    # Initialize a figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11, 9))
    
    # Plot values
    values_plt, = plt.plot(values)
    

    # Set plot title
    plt.title(f'{metric}')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    
    

