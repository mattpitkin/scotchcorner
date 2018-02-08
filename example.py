#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
from scotchcorner import scotchcorner as sc

# number of parameters
ndims = 4

# ratio of joint plots to histogram plots
ratio = 3

# create some data to plot
x = np.zeros((2000,ndims))
x[:,0:2] = np.random.randn(2000,2)
x[0:500,2] = -1. + 0.25*np.random.randn(500)
x[500:2000,2] = 3. + 1.25*np.random.randn(1500)
x[0:1000, 3] = 1.*np.random.randn(1000)
x[1000:2000, 3] = 3.+0.75*np.random.randn(1000)
x[:,0] = x[:,0]*1e-6
x[:,1] = x[:,1]*1e-8
x[:,3] = x[:,3]*1e-10

showlims = 'both'

# set parameter names (last one contains units in square brackets)
labels = [r'$\textrm{b}$', r'$x$', r'$\phi$', r'$y$ [$\textrm{s}$]']

# set histogram options
histops = {'histtype': 'stepfilled',
           'color': 'darkslategrey',
           'edgecolor': 'black',
           'linewidth': 1.5}

# set extents of parameters
limits = [(-5., 5.), (None, None), (None, None), (None, None)]

# create plot
p = sc(x, labels=labels, truths=[0., 1.52e-8, 0., 0.], showlims=showlims,
       hist_kwargs=histops, datatitle='Data 1', figsize=(10,10), limits=limits)

# add some more data
z = np.random.randn(2000,ndims)
z[:,0] = z[:,0]*1e-6
z[:,1] = z[:,1]*1e-8
z[:,3] = z[:,3]*1e-10
p.add_data(z, hist_kwargs={'histtype': 'step', 'color': 'blue', 'linewidth': 1}, datatitle='Data 2',
           showcontours=True, contour_kwargs={'colors': 'blue'})

# save figure
p.savefig('example.png')

