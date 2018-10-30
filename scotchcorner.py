from __future__ import print_function, division

__version__ = "0.2.1"
__author__ = "Matthew Pitkin (matthew.pitkin@glasgow.ac.uk)"
__copyright__ = "Copyright 2016 Matthew Pitkin, Ben Farr and Will Farr"

import numpy as np
import pandas as pd
import scipy.stats as ss
import math

import matplotlib as mpl
from matplotlib import pyplot as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import matplotlib.gridspec as gridspec
from matplotlib import transforms as mtransforms
from matplotlib import patheffects as PathEffects


# A bounded KDE class (inherited from the SciPy Gaussian KDE class) created by
# Ben Farr @bfarr
class Bounded_2d_kde(ss.gaussian_kde):
    """
    Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain (by `Ben Farr <https://github.com/bfarr>`_).
    """

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None,
                 *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`scipy.stats.gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.

        :param ylow: The lower y domain boundary.

        :param yhigh: The upper y domain boundary.
        """
        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'

        super(Bounded_2d_kde, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    @property
    def ylow(self):
        """The lower bound of the y domain."""
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain."""
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        x, y = pts.T
        pdf = super(Bounded_2d_kde, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, y])

        if self.xhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, y])

        if self.ylow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2*self.ylow - y])

        if self.yhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2*self.yhigh - y])

        if self.xlow is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x,
                                                             2*self.ylow - y])

            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x,
                                                             2*self.yhigh - y])

        if self.xhigh is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x,
                                                             2*self.ylow - y])
            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x,
                                                             2*self.yhigh - y])

        return pdf

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results


class scotchcorner(object):
    """
    Create a corner-style plot.

    Parameters
    ----------
    data : :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        A (`N` x `ndims`) array of values for the `ndims` parameters
    bins : int, optional, default: 20
        The number of bins in the 1D histogram plots
    ratio : int, optional, default: 3
        The ratio of the size of 1D histograms to the size of the joint plots
    labels : list, optional
        A list of names for each of the `ndims` parameters.  These are used for
        the axes labels. If `data` is a :class:`pandas.DataFrame` then the
        column names of that will be used instead.
    truths : list, optional, default: None
        A list of the true values of each parameter
    datatitle : string, optional
        A title for the data set to be added as a legend
    showlims : string, optional, default: None
        Show edges/borders at the plots limits. Use 'hist' for limits on the 1D
        histogram plots, 'joint' for borders around 2D joint plots, or 'both'
        for borders on the 1D and 2D plots. The default (None) is for no
        borders.
    limlinestyle : default: 'dotted'
        The line style for the plot borders
    hist_kwargs : dict
        A dictionary of keywords arguments for the histogram function
    truth_kwargs : dict
        A dictionary of keyword arguments for plotting true values
    showpoints: bool, default: True
        Show the data points in the 2D joint parameter plots
    thinpoints : float, default: 1.0
        Thin-down the number of points being plotted in the 2D scatter plots by
        this factor
    scatter_kwargs : dict
        A dictionary of keyword arguments for the scatter plot function
    showcontours : bool, default: False
        Show KDE probability contours for the 2D joint parameter plots (with
        levels defined by `contour_levels`)
    contour_kwargs : dict
        A dictionary of keyword argumemts for the contour plot function
    contour_levels : list, default: [0.5, 0.9]
        A list of values between 0 and 1 indicating the probability contour
        confidence intervals to plot (defaulting to 50% and 90% contours)
    show_level_labels : bool, default: True
        Add labels on the contours levels showing their probability
    use_math_text : bool, default: True
        Use math text scientific notation for parameter tick mark labelling
    limits : list, default: None
        A list of tuples giving the lower and upper limits for each parameter.
        If limits for some parameters are not known/required then an empty
        tuple (or `None` within a two value tuple) must be placed in the list
        for that parameter
    contour_limits : list, default: None
        A list of tuples giving the lower and upper limits for each parameter
        for use when creating credible interval contour for joint plots. If
        limits for some parameters are not known/required then an empty tuple
        (or `None` within a two value tuple) must be placed in the list for
        that parameter
    subtract_truths : list or tuple, optional, default: None
        A list/tuple of indices of parameters for which you want to show the
        distribution centred such that true value is zero. This is only
        relevent if `truths` are supplied.
    figsize : tuple
        A two value tuple giving the figure size
    mplparams : dict
        A dictionary containing matplotlib configuration values

    """
    def __init__(self, data, bins=20, ratio=3, labels=None, truths=None,
                 datatitle=None, showlims=None, limlinestyle='dotted',
                 showpoints=True, showcontours=False, hist_kwargs={},
                 truths_kwargs={}, scatter_kwargs={}, contour_kwargs={},
                 contour_levels=[0.5, 0.9], show_level_labels=True,
                 use_math_text=True, limits=None, contour_limits=None,
                 figsize=None, mplparams=None, thinpoints=1.0,
                 subtract_truths=None):
        # get number of dimensions in the data
        self.ndims = data.shape[1]  # get number of dimensions in data
        self.ratio = ratio
        if isinstance(data, pd.DataFrame):
            self.labels = data.columns
        else:
            self.labels = labels
        self.truths = truths          # true values for each parameter in data
        self.truths_kwargs = truths_kwargs
        if self.truths is not None:
            # must be same number of true values as parameters
            if len(self.truths) != self.ndims:
                self.truths = None
        self.subtract_truths = subtract_truths
        self.levels = contour_levels
        self.showpoints = showpoints
        self.thinpoints = thinpoints
        self.thinpermutation = None
        self.showcontours = showcontours
        self.scatter_kwargs = scatter_kwargs
        self.contour_kwargs = contour_kwargs
        self.show_level_labels = show_level_labels
        self.legend_labels = []
        self.use_math_text = use_math_text
        self.limits = limits  # a list of tuples giving the lower and upper limits for each parameter - if some values aren't given then an empty tuple must be placed in the list for that value
        self.contourlimits = contour_limits  # a list of tuples giving the lower and upper limits for each parameter for use in credible interval contours - if some values aren't given then an empty tuple must be placed in the list for that value

        # default figure size (numbers "stolen" from those used in corner.py that are, to quote, "Some magic numbers for pretty axis layout."
        factor = 2.0           # size of one side of one panel
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
        whspace = 0.05         # w/hspace size
        K = self.ndims - 1. + (1./self.ratio)  # different from corner.py to account for histogram ratio
        plotdim = factor * K + factor * (K - 1.) * whspace
        dim = lbdim + plotdim + trdim
        self.figsize = (dim, dim)  # default figure size
        if figsize is not None:
            if isinstance(figsize, tuple):
                if len(figsize) == 2:
                    self.figsize = figsize

        # set plot parameters
        if mplparams == None:
            # set default parameters
            self.mplparams = {
                'text.usetex': True,    # use LaTeX for all text
                'axes.linewidth': 0.5,  # set axes linewidths to 0.5
                'axes.grid': False,     # add a grid
                'font.family': 'sans-serif',
                'font.sans-serif': 'Avant Garde, Helvetica, Computer Modern Sans serif',
                'font.size': 15,
                'legend.fontsize': 'medium',
                'legend.frameon': False,
                'axes.formatter.limits': (-3, 4)}
        else:
            self.mplparams = mplparams

        mpl.rcParams.update(self.mplparams)

        # set default hist_kwargs
        self.hist_kwargs = {'bins': bins,
                            'histtype': 'stepfilled',
                            'color': 'lightslategrey',
                            'alpha': 0.4,
                            'edgecolor': 'lightslategray',
                            'linewidth': 1.5}
        for key in hist_kwargs.keys():  # set any values input
            self.hist_kwargs[key] = hist_kwargs[key]

        if bins != 20:
            if isinstance(bins, int) and bins > 0:
                self.hist_kwargs['bins'] = bins

        # create figure
        self._fig = pl.figure(figsize=self.figsize)
        self.histhori = []
        self.histhori_indices = list(range(0,self.ndims-1))  # indexes of parameters in horizontal histograms
        self.histvert = []
        self.histvert_indices = list(range(1,self.ndims))  # indexes of parameters in vertical histograms
        self.jointaxes = []
        self.jointaxes_indices = []
        self._axes = {}  # dictionary of axes keyed to parameter names if available

        # format the figure (again this is stolen from corner.py)
        lb = lbdim / dim
        tr = (lbdim + plotdim) / dim
        self._fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                                  wspace=whspace, hspace=whspace)

        # create grid
        gridsize = self.ratio*(self.ndims-1) + 1
        gs = gridspec.GridSpec(gridsize, gridsize, wspace=0.1, hspace=0.1)

        # empty axes to hold any legend information (if not just a 2D plot)
        if data.shape[1] > 2:
            self.legendaxis = self._fig.add_subplot(gs[0:ratio,((self.ndims-2)*ratio+1):(1+(self.ndims-1)*ratio)])
            for loc in ['top', 'right', 'left', 'bottom']:
                self.legendaxis.spines[loc].set_visible(False)  # remove borders
            pl.setp(self.legendaxis.get_xticklabels(), visible=False)  # remove xtick labels
            pl.setp(self.legendaxis.get_yticklabels(), visible=False)  # remove ytick labels
            self.legendaxis.tick_params(bottom=False, top=False, left=False,
                                        right=False)  # remove tick marks

        # create figure axes
        for i in range(self.ndims-1):
            # vertical histogram (and empty axes)
            axv = self._fig.add_subplot(gs[i*ratio:(i+1)*ratio,0])
            if showlims in ['hist', 'both']:
                for loc in ['top', 'bottom']:
                    axv.spines[loc].set_alpha(0.2)
                    axv.spines[loc].set_linestyle(limlinestyle)
            else:
                axv.spines['top'].set_visible(False)     # remove top border
                axv.spines['bottom'].set_visible(False)  # remove bottom border
            axv.spines['right'].set_visible(False)       # remove right border
            axv.set_xticklabels([])
            axv.set_xticks([])
            axv.yaxis.set_ticks_position('left')  # just show ticks on left
            self.histvert.append(axv)
            self.histvert_indices.append(i+1)

            # horizontal histograms
            axh = self._fig.add_subplot(gs[-1,(i*ratio+1):(1+(i+1)*ratio)])
            axh.spines['top'].set_visible(False)    # remove top border
            if showlims in ['hist', 'both']:
                for loc in ['left', 'right']:
                    axh.spines[loc].set_alpha(0.2)
                    axh.spines[loc].set_linestyle(limlinestyle)
            else:
                axh.spines['left'].set_visible(False)   # remove left border
                axh.spines['right'].set_visible(False)  # remove right border
            axh.set_yticklabels([])
            axh.set_yticks([])
            axh.xaxis.set_ticks_position('bottom')  # just show ticks on bottom
            self.histhori.append(axh)

            # joint plots
            for j in range(i+1):
                axj = self._fig.add_subplot(gs[i*ratio:(i+1)*ratio,(j*ratio+1):(1+(j+1)*ratio)],
                                            sharey=self.histvert[i], sharex=self.histhori[j])
                if data.shape[1] == 2:
                    # use this as the legend axis
                    self.legendaxis = axj
                if showlims in ['joint', 'both']:
                    for loc in ['top', 'right', 'left', 'bottom']:
                        axj.spines[loc].set_alpha(0.2)  # show border, but with alpha = 0.2
                        axj.spines[loc].set_linestyle(limlinestyle)
                else:
                    for loc in ['top', 'right', 'left', 'bottom']:
                        axj.spines[loc].set_visible(False) # remove borders

                pl.setp(axj.get_xticklabels(), visible=False)  # remove xtick labels
                pl.setp(axj.get_yticklabels(), visible=False)  # remove ytick labels
                axj.tick_params(bottom=False, top=False, left=False,
                                right=False)  # remove tick marks 
                self.jointaxes.append(axj)

        # check for alpha of filled histogram plot
        if self.hist_kwargs['histtype'] == 'stepfilled':
            self._check_alpha()

        # create plots
        self._add_plots(data, label=datatitle)

    def add_data(self, data, hist_kwargs={}, datatitle=None, showpoints=True,
                 showcontours=False, scatter_kwargs={}, contour_kwargs={},
                 truths=None, truths_kwargs={}, contour_levels=[0.5, 0.9],
                 limits=None, contour_limits = None, show_level_labels=True,
                 thinpoints=1.0):
        """
        Add another data set to the plots, `hist_kwargs` are required.
        """

        if data.shape[1] != self.ndims:
            raise("Error... number of dimensions not the same")

        # update with any newly supplied histogram keyword arguments
        for key in hist_kwargs:
            self.hist_kwargs[key] = hist_kwargs[key]
        
        if 'bins' not in self.hist_kwargs:
            # set default number of bins to 20
            self.hist_kwargs['bins'] = 20
        if 'linewidth' not in self.hist_kwargs:
            self.hist_kwargs['linewidth'] = 1.5
        self.truths = truths
        if self.truths is not None:
            if len(self.truths) != self.ndims:  # must be same number of true values as parameters
                self.truths = None
        self.scatter_kwargs = scatter_kwargs
        self.levels = contour_levels
        self.showpoints = showpoints
        self.showcontours = showcontours
        self.contour_kwargs = contour_kwargs
        self.truths_kwargs = truths_kwargs
        self.show_level_labels = show_level_labels
        self.contourlimits = contour_limits
        self.limits = limits
        
        if self.showpoints:
            if thinpoints != self.thinpoints:
                self.thinpoints = thinpoints
            self.thinpermutation = None

        self._add_plots(data, label=datatitle)

    def _add_plots(self, data, label=None):
        """
        Add histogram and joint plots to the figure using data
        
        Label is a legend label if required.
        """

        # make sure data has enough dimensions (and convert pandas.DataFrame to numpy ndarray)
        data = np.atleast_1d(data)

        # set default truth style
        if self.truths is not None:
            if 'color' not in self.truths_kwargs:
                if 'color' in self.hist_kwargs:
                    self.truths_kwargs['color'] = self.hist_kwargs['color']
                elif 'edgecolor' in self.hist_kwargs:
                    self.truths_kwargs['color'] = self.hist_kwargs['edgecolor']
                else:
                    self.truths_kwargs['color'] == 'k'

            if 'linestyle' not in self.truths_kwargs:
                self.truths_kwargs['linestyle'] = '--'

            if 'linewidth' not in self.truths_kwargs:
                self.truths_kwargs['linewidth'] = 1.5

        # the vertical histogram
        subval = 0. # value to subtract from distribution (to centre truth value at 0)
        if self.subtract_truths is not None and self.truths is not None:
            if len(self.truths)-1 in self.subtract_truths and self.truths[-1] is not None:
                subval = self.truths[-1]
        self.histvert[-1].hist(data[:,-1]-subval, density=True,
                               orientation='horizontal', label=label,
                               **self.hist_kwargs)
        if self.truths is not None:
            if self.truths[-1] is not None:
                marker = None
                if 'marker' in self.truths_kwargs:  # remove any marker for line
                    marker = self.truths_kwargs.pop('marker')
                self.histvert[-1].axhline(self.truths[-1]-subval,
                                          **self.truths_kwargs)
                if marker is not None:
                    self.truths_kwargs['marker'] = marker

        # put legend in the upper right plot
        _, l1 = self.histvert[-1].get_legend_handles_labels()
        if self.legend_labels is not None:
            if self.hist_kwargs['histtype'] == 'stepfilled':
                lc = self.hist_kwargs['edgecolor']
            else:
                lc = self.hist_kwargs['color']
            self.legend_labels.append(Line2D([], [],
                                             linewidth=self.hist_kwargs['linewidth'],
                                             color=lc))  # create fake line for legend (to use line rather than a box)
        if data.shape[1] == 2:
            self.legendaxis.legend(self.legend_labels, l1, loc='best',
                                   fancybox=True, framealpha=0.4)
        else:
            self.legendaxis.legend(self.legend_labels, l1, loc='lower left')
        if self.labels is not None:
            self.histvert[-1].set_ylabel(self.labels[-1])
            self._axes[self.labels[-1]] = self.histvert[-1]

        if self.showpoints:
            # set default scatter plot kwargs
            if 'color' in self.hist_kwargs:
                c = self.hist_kwargs['color']
            elif 'fc' in self.hist_kwargs and self.hist_kwargs['histtype'] == 'stepfilled':
                c = [self.hist_kwargs['fc'][0:3]]
            else:
                c = 'b'

            these_scatter_kwargs = {'c': c, 'marker': 'o', 's': 20,
                                    'alpha': 0.05, 'edgecolors': 'none'}
            
            for key in self.scatter_kwargs.keys():
                these_scatter_kwargs[key] = self.scatter_kwargs[key]
            self.scatter_kwargs = these_scatter_kwargs
        
        if self.limits is not None:
            if len(self.limits) != self.ndims:
                raise("Error... number of dimensions is not the same as the "
                      "number of limits being set")
        
        if self.contourlimits is not None:
            if len(self.contourlimits) != self.ndims:
                raise("Error... number of dimensions is not the same as the "
                      "number of contour limits being set")
        
        if self.showcontours:
            # set default contour kwargs
            these_contour_kwargs = {'colors': 'k'}

            for key in self.contour_kwargs.keys():
                these_contour_kwargs[key] = self.contour_kwargs[key]
            self.contour_kwargs = these_contour_kwargs

        # the horizontal histograms and joint plots
        jointcount = 0
        rowcount = 0
        for i in range(self.ndims-1):
            # check if subtracting the true values from the distribution
            subval = 0.
            if self.subtract_truths is not None and self.truths is not None:
                if i in self.subtract_truths and self.truths[i] is not None:
                    subval = self.truths[i]

            self.histhori[i].hist(data[:,i]-subval, density=True, **self.hist_kwargs)

            # make sure axes ranges on vertical histograms match those on the equivalent horizontal histograms
            if i > 0:
                xmin, xmax = self.histhori[i].get_xlim()
                self.histvert[i-1].set_ylim([xmin, xmax])

            if self.labels is not None:
                self.histhori[i].set_xlabel(self.labels[i])
                self._axes[self.labels[i]] = self.histhori[i]
            if self.truths is not None:
                if self.truths[i] is not None:
                    marker = None
                    if 'marker' in self.truths_kwargs:  # remove any marker for line
                        marker = self.truths_kwargs.pop('marker')
                    self.histhori[i].axvline(self.truths[i]-subval, **self.truths_kwargs)
                    if marker is not None:
                        self.truths_kwargs['marker'] = marker

            for j in range(i+1):
                if self.labels is not None:
                    if j == 0:
                        self.histvert[rowcount].set_ylabel(self.labels[i+1])
                        rowcount += 1

                    self._axes[self.labels[j]+'vs'+self.labels[i+1]] = self.jointaxes[jointcount]

                # check if subtracting the true values from the distribution
                subvals = [0., 0.]
                if self.subtract_truths is not None and self.truths is not None:
                    if self.truths[j] is not None and j in self.subtract_truths:
                        subvals[0] = self.truths[j]
                    if self.truths[i+1] is not None and i+1 in self.subtract_truths:
                        subvals[1] = self.truths[i+1]

                # get joint axes indices
                self.jointaxes_indices.append((j, i+1))

                if self.showpoints:
                    if self.thinpoints < 1. or self.thinpoints > data.shape[0]:
                        raise("Error... Thinning factor is less than 1 or greater than the total number of data points")
                    if self.thinpermutation is None:  # select points randomly for thinning
                        nthinpoints = int(data.shape[0]/self.thinpoints)
                        permutepoints = np.random.permutation(np.arange(data.shape[0]))[:nthinpoints]
                        self.thinpermutation = permutepoints

                    # plot scatter
                    self.jointaxes[jointcount].scatter(data[self.thinpermutation,j]-subvals[0],
                                                       data[self.thinpermutation,i+1]-subvals[1],
                                                       **self.scatter_kwargs)

                if self.showcontours:
                    xlow = xhigh = ylow = yhigh = None  # default limits
                    if self.contourlimits is not None:
                        if len(self.contourlimits[j]) == 2:
                            xlow = self.contourlimits[j][0]
                            xhigh = self.contourlimits[j][1]
                        if len(self.contourlimits[i+1]) == 2:
                            ylow = self.contourlimits[i+1][0]
                            yhigh = self.contourlimits[i+1][1]

                    self.plot_bounded_2d_kde_contours(self.jointaxes[jointcount],
                                                      np.vstack((data[:,j]-subvals[0], data[:,i+1]-subvals[1])).T,
                                                      xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)

                if self.truths is not None:
                    if self.truths[j] is not None and self.truths[i+1] is not None:
                        markertmp = None
                        if 'marker' not in self.truths_kwargs:
                            self.truths_kwargs['marker'] = 'x'

                        self.jointaxes[jointcount].plot(self.truths[j]-subvals[0],
                                                        self.truths[i+1]-subvals[1],
                                                        **self.truths_kwargs)

                jointcount += 1

    def get_axis(self, param):
        """
        Return the axis for the given "param" (for joint axes "param" should
        be the required parameters separated by "vs")
        """
        if param in self._axes:
            return self._axes[param]
        else:
            print("Parameter '%s' not one of the axes.")
            return None

    def _format_axes(self):
        """
        Set some formatting of the axes
        """

        pl.draw()  # force labels to be drawn

        theselimits = None
        if self.limits is not None:
            theselimits = list(self.limits)  # local copy of the limits

        for i, ax in enumerate(self.histhori):
            # set limits
            if theselimits is not None:
                xmin, xmax = ax.get_xlim()  # get current limits
                if len(theselimits[self.histhori_indices[i]]) == 2:
                    xminnew, xmaxnew = theselimits[self.histhori_indices[i]]
                    if xminnew == None:
                        xminnew = xmin
                    if xmaxnew == None:
                        xmaxnew = xmax
                    ax.set_xlim([xminnew, xmaxnew])
                    theselimits[self.histhori_indices[i]] = [xminnew, xmaxnew]  # reset local copy of limits to these values (so vertical hists and joint axes have the same ranges)
                else:
                    theselimits[self.histhori_indices[i]] = [xmin, xmax]  # set the local copy of limits, (so vertical hists and joint axes have the same ranges)
            
            nbins = min([len(ax.get_xticklabels()), 5])  # make sure there are at least 4 tick marks (after removal of one) and a max of 7
            prune = None
            if i > 0:  # remove the lower tick label to avoid overlapping labels
                prune = 'lower'
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7,
                                                   min_n_ticks=nbins,
                                                   prune=prune))
            for l in ax.get_xticklabels():
                l.set_rotation(45)
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=self.use_math_text))
            self.format_exponents_in_label_single_ax(ax.xaxis)  # move exponents into label

        for i, ax in enumerate(self.histvert):
            # set limits
            if theselimits is not None:
                if len(theselimits[self.histvert_indices[i]]) == 2:
                    ymin, ymax = ax.get_ylim()  # get current limits
                    yminnew, ymaxnew = theselimits[self.histvert_indices[i]]
                    if yminnew == None:
                        yminnew = ymin
                    if ymaxnew == None:
                        ymaxnew = ymax
                    ax.set_ylim([yminnew, ymaxnew])

            # make sure there are at least 4 tick marks (after removal of one) and a max of 7
            nbins = min([len(ax.get_yticklabels()), 5])
            prune = None  # remove lower tick to avoid overlapping labels
            if i < len(self.histvert)-1:
                prune = 'lower'
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7,
                                                   min_n_ticks=nbins,
                                                   prune=prune))
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=self.use_math_text))
            self.format_exponents_in_label_single_ax(ax.yaxis)  # move exponents into label

        for i, ax in enumerate(self.jointaxes):
            # remove any offset text from shared axes caused by the scalar formatter for MathText
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

            if theselimits is not None:
                if len(theselimits[self.jointaxes_indices[i][0]]) == 2:
                    xmin, xmax = ax.get_xlim()  # get current limits
                    xminnew, xmaxnew = theselimits[self.jointaxes_indices[i][0]]
                    if xminnew == None:
                        xminnew = xmin
                    if xmaxnew == None:
                        xmaxnew = xmax
                    dx = 0.02*(xmaxnew-xminnew)  # add a little bit of space
                    ax.set_xlim([xminnew-dx, xmaxnew+dx])
                if len(theselimits[self.jointaxes_indices[i][1]]) == 2:
                    ymin, ymax = ax.get_ylim()  # get current limits
                    yminnew, ymaxnew = theselimits[self.jointaxes_indices[i][1]]
                    if yminnew == None:
                        yminnew = ymin
                    if ymaxnew == None:
                        ymaxnew = ymax
                    dy = 0.02*(ymaxnew-yminnew) # add a little bit of space
                    ax.set_ylim([yminnew-dy, ymaxnew+dy])

    def plot_bounded_2d_kde_contours(self, ax, pts, xlow=None, xhigh=None,
                                     ylow=None, yhigh=None, transform=None,
                                     gridsize=250, clip=None):
        """
        Function (based on that in `plotutils` by
        `Will Farr <https://github.com/farr>`_ and edited by
        `Ben Farr <https://github.com/bfarr>`_) for plotting contours from a
        bounded 2d KDE.
        """

        if transform is None:
            transform = lambda x: x

        # Determine the clipping
        if clip is None:
            clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
        elif np.ndim(clip) == 1:
            clip = [clip, clip]

        # Calculate the KDE
        Npts = pts.shape[0]
        kde_pts = transform(pts[:int(Npts/2), :])
        den_pts = transform(pts[int(Npts/2):, :])

        Nden = den_pts.shape[0]

        post_kde = Bounded_2d_kde(kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow,
                                  yhigh=yhigh)
        den = post_kde(den_pts)
        densort = np.sort(den)[::-1]

        zvalues = []
        for level in self.levels:
            ilevel = int(Nden*level + 0.5)
            if ilevel >= Nden:
                ilevel = Nden-1
            zvalues.append(densort[ilevel])

        # sort into ascending order (required in Matplotlib v 1.5.1)
        zvalues.sort()

        x = pts[:,0]
        y = pts[:,1]
        deltax = x.max() - x.min()
        deltay = y.max() - y.min()
        x_pts = np.linspace(x.min() - .1*deltax, x.max() + .1*deltax, gridsize)
        y_pts = np.linspace(y.min() - .1*deltay, y.max() + .1*deltay, gridsize)

        xx, yy = np.meshgrid(x_pts, y_pts)

        positions = np.column_stack([xx.ravel(), yy.ravel()])

        z = np.reshape(post_kde(transform(positions)), xx.shape)

        # Black (thin) contours with while outlines by default
        self.contour_kwargs['linewidths'] = self.contour_kwargs.get('linewidths', 1.)

        # Plot the contours (plot them seperately)
        for k, level in enumerate(self.levels):
            alpha = self.contour_kwargs.pop('alpha', 1.0)
            self.contour_kwargs['alpha'] = level  # set tranparency to the contour level
            cset = ax.contour(xx, yy, z, [zvalues[k]], **self.contour_kwargs)
            self.contour_kwargs['alpha'] = alpha

            # Add white outlines
            if self.contour_kwargs['colors'] == 'k':
                pl.setp(cset.collections, path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="w")])
            fmt = {}
            fmt[cset.levels[0]] = '{}%'.format(int(100*level))

            if self.show_level_labels:
                lw = self.contour_kwargs.pop('linewidths')
                alpha = self.contour_kwargs.pop('alpha')
                ax.clabel(cset, cset.levels, fmt=fmt, fontsize=11, **self.contour_kwargs)
                pl.setp(cset.labelTexts, color='k', path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="w")])
                self.contour_kwargs['linewidths'] = lw
                self.contour_kwargs['alpha'] = alpha

    def _check_alpha(self):
        """
        Use alpha transparency on (step filled) histogram patches, but not on edges
        (based on the answer `here <http://stackoverflow.com/a/28398471/1862861>`_)
        """
        if 'alpha' in self.hist_kwargs:
            alpha = self.hist_kwargs.pop('alpha')
            if 'color' in self.hist_kwargs:
                cl = self.hist_kwargs.pop('color')
            else:
                # default to blue if no color is given
                cl = 'blue'

            if not isinstance(cl, tuple):
                # import these to get RGB color codes for names colors
                from matplotlib import colors as cs

                if cl in cs.cnames:
                    rgbcolor = cs.hex2color(cs.cnames[cl])
                else:
                    print("histogram color '%s' not recognised. Defaulting to blue" % cl)
                    rgbcolor = cs.hex2color(cs.cnames['blue'])
                # add facecolor 'fc' to hist_kwargs
                ctup = rgbcolor + (alpha,)
            else:
                if len(cl) == 3:
                    ctup = cl + (alpha,)
                else:
                    ctup = cl

            # add tuple (r, g, b, alpha) facecolor 'fc' to hist_kwargs
            self.hist_kwargs['fc'] = ctup
            
    def update_label(self, old_label, exponent_text):
        """
        Method to transform given label into the new label (this function comes
        from `this patch <https://github.com/dfm/corner.py/pull/53/files>`_ to
        `corner.py <https://github.com/dfm/corner.py>`_
        by `Greg Ashton <https://github.com/ga7g08>`_) """
        if exponent_text == "":
            return old_label
        try:
            units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
        except ValueError:
            units = ""

        label = old_label.replace("[{}]".format(units), "")
        exponent_text = exponent_text.replace("\\times", "")
        if units == "":
            if label == "":
                s = r"[{}]".format(exponent_text)
            else:
                s = r"{} [{}]".format(label, exponent_text)
        else:
            if label == "":
                s = r"[{} {}]".format(exponent_text, units)
            else:
                s = r"{} [{} {}]".format(label, exponent_text, units)
        return s

    def format_exponents_in_label_single_ax(self, ax):
        """ Routine for a single axes instance (by Greg Ashton) """

        exponent_text = ax.get_offset_text().get_text()
        exponent_text = exponent_text.replace("\\mathdefault", "")
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(self.update_label(label, exponent_text))
            
    def savefig(self, filename):
        """
        Save the figure

        Parameters
        ----------
        filename : str, required
            The filename of the figure to save. The figure format is determined
            by the file extension.
        """
        self._format_axes()
        self._fig.savefig(filename)

    def show(self):
        """ Show the figure """
        self._format_axes()
        self._fig.show()

    @property
    def fig(self):
        """ Return the :class:`matplotlib.figure.Figure` """
        return self._fig

