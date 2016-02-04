from __future__ import print_function

__version__ = "0.0.1"
__author__ = "Matthew Pitkin (matthew.pitkin@glasgow.ac.uk)"
__copyright__ = "Copyright 2016 Matthew Pitkin, Ben Farr and Will Farr"

import numpy as np
import scipy.stats as ss

import matplotlib as mpl
from matplotlib import pyplot as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from matplotlib import patheffects as PathEffects

# A bounded KDE class (inherited from the SciPy Gaussian KDE class) created by Ben Farr @bfarr
class Bounded_2d_kde(ss.gaussian_kde):
    r"""Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

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
        """Return an estimate of the density evaluated at the given
        points."""
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
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, 2*self.ylow - y])

            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, 2*self.yhigh - y])

        if self.xhigh is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, 2*self.ylow - y])
            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, 2*self.yhigh - y])

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


class newcorner:
    """
    Define class to plot the new corner plot style
    """
    def __init__(self, data, bins=None, ratio=3, labels=None, truths=None, legend=None, showlims=False,
                 limlinestyle='dotted', showpoints=True, showcontours=False, hist_kwargs={},
                 scatter_kwargs={}, contour_kwargs={}, contour_levels=[0.5, 0.9], show_contour_levels=True,
                 use_math_text=True, limits=None, figsize=None, mplparams=None):
        # get number of dimensions in the data
        self.ndims = data.shape[1] # get number of dimensions in data
        self.ratio = ratio
        self.labels = labels
        self.truths = truths                 # true values for each parameter in data 
        if self.truths != None:
            if len(self.truths) != self.ndims: # must be same number of true values as parameters
                self.truths = None
        self.levels = contour_levels
        self.showpoints = showpoints
        self.showcontours = showcontours
        self.scatter_kwargs = scatter_kwargs
        self.contour_kwargs = contour_kwargs
        self.show_contour_levels = show_contour_levels
        self.legend_labels = []
        self.use_math_text = use_math_text
        self.limits = limits  # a list of tuples giving the lower and upper limits for each parameter - if some values aren't given then an empty tuple must be placed in the list for that value
        
        # default figure size (numbers "stolen" from those used in corner.py that are, to quote, "Some magic numbers for pretty axis layout."
        factor = 2.0           # size of one side of one panel
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
        whspace = 0.05         # w/hspace size
        K = self.ndims - 1. + (1./self.ratio) # different from corner.py to account for histogram ratio
        plotdim = factor * K + factor * (K - 1.) * whspace
        self.figsize = (lbdim + plotdim + trdim , lbdim + plotdim + trdim) # default figure size
        if figsize != None:
            if isinstance(figsize, tuple):
                if len(figsize) == 2:
                    self.figsize = figsize
        
        # set plot parameters
        if mplparams == None:
            # set default parameters
            self.mplparams = {
                'text.usetex': True, # use LaTeX for all text
                'axes.linewidth': 0.5, # set axes linewidths to 0.5
                'axes.grid': False, # add a grid
                'font.family': 'sans-serif',
                'font.sans-serif': 'Avant Garde, Helvetica, Computer Modern Sans serif',
                'font.size': 16,
                'legend.fontsize': 'medium',
                'legend.frameon': False,
                'axes.formatter.limits': (-3,4)}
        else:
            self.mplparams = mplparams

        mpl.rcParams.update(self.mplparams)
        
        # set default hist_kwargs
        self.hist_kwargs = {'bins': 20, 'histtype': 'stepfilled', 'color': 'lightslategrey', 'alpha': 0.4, 'edgecolor': 'lightslategray', 'linewidth': 1.5}
        for key in hist_kwargs.keys(): # set any values input
            self.hist_kwargs[key] = hist_kwargs[key]

        if bins != None:
            if isinstance(bins, int) and bins > 0:
                self.hist_kwargs['bins'] = bins

        # create figure
        self.fig = pl.figure(figsize=self.figsize)
        self.histhori = []
        self.histhori_indices = range(0,self.ndims-1) # indexes of parameters in horizontal histograms
        self.histvert = []
        self.histvert_indices = range(1,self.ndims) # indexes of parameters in vertical histograms
        self.jointaxes = []
        self.jointaxes_indices = []
        
        # create grid
        gridsize = self.ratio*(self.ndims-1) + 1
        gs = gridspec.GridSpec(gridsize, gridsize, wspace=0.1, hspace=0.1)

        # empty axes to hold any legend information
        self.legendaxis = self.fig.add_subplot(gs[0:ratio,((self.ndims-2)*ratio+1):(1+(self.ndims-1)*ratio)])
        for loc in ['top', 'right', 'left', 'bottom']:
            self.legendaxis.spines[loc].set_visible(False) # remove borders
        pl.setp(self.legendaxis.get_xticklabels(), visible=False) # remove xtick labels
        pl.setp(self.legendaxis.get_yticklabels(), visible=False) # remove ytick labels
        self.legendaxis.tick_params(bottom='off', top='off', left='off', right='off') # remove tick marks
        
        # create figure axes
        for i in range(self.ndims-1):
            # vertical histogram (and empty axes)
            axv = self.fig.add_subplot(gs[i*ratio:(i+1)*ratio,0])
            if showlims:
                for loc in ['top', 'bottom']:
                    axv.spines[loc].set_alpha(0.2)
                    axv.spines[loc].set_linestyle(limlinestyle)
            else:
                axv.spines['top'].set_visible(False)    # remove top border
                axv.spines['bottom'].set_visible(False) # remove bottom border
            axv.spines['right'].set_visible(False)  # remove right border
            axv.set_xticklabels([])
            axv.set_xticks([])
            axv.yaxis.set_ticks_position('left') # just show ticks on left
            self.histvert.append(axv)
            self.histvert_indices.append(i+1)
                
            # horizontal histograms
            axh = self.fig.add_subplot(gs[-1,(i*ratio+1):(1+(i+1)*ratio)])
            axh.spines['top'].set_visible(False)    # remove top border
            if showlims:
                for loc in ['left', 'right']:
                    axh.spines[loc].set_alpha(0.2)
                    axh.spines[loc].set_linestyle(limlinestyle)
            else:
                axh.spines['left'].set_visible(False)   # remove left border
                axh.spines['right'].set_visible(False)  # remove right border
            axh.set_yticklabels([])
            axh.set_yticks([])
            axh.xaxis.set_ticks_position('bottom') # just show ticks on left
            self.histhori.append(axh)
                
            # joint plots
            for j in range(i+1):
                axj = self.fig.add_subplot(gs[i*ratio:(i+1)*ratio,(j*ratio+1):(1+(j+1)*ratio)], sharey=self.histvert[i], sharex=self.histhori[j])
                if showlims:
                    for loc in ['top', 'right', 'left', 'bottom']:
                        axj.spines[loc].set_alpha(0.2) # show border, but with alpha = 0.2
                        axj.spines[loc].set_linestyle(limlinestyle)
                else:
                    for loc in ['top', 'right', 'left', 'bottom']:
                        axj.spines[loc].set_visible(False) # remove borders

                pl.setp(axj.get_xticklabels(), visible=False) # remove xtick labels
                pl.setp(axj.get_yticklabels(), visible=False) # remove ytick labels
                axj.tick_params(bottom='off', top='off', left='off', right='off') # remove tick marks 
                self.jointaxes.append(axj)
        
        # check for alpha of filled histogram plot
        if self.hist_kwargs['histtype'] == 'stepfilled':
            self._check_alpha()
        
        # create plots
        self._add_plots(data, label=legend)
        
    def add_data(self, data, hist_kwargs, legend=None, showpoints=True, showcontours=False, scatter_kwargs={},
                 contour_kwargs={}, contour_levels=[0.5, 0.9], limits=None, show_contour_levels=True):
        """
        Add another data set to the plots, hist_kwargs are required.
        """

        if data.shape[1] != self.ndims:
            raise("Error... number of dimensions not the same")

        self.hist_kwargs = hist_kwargs
        if 'bins' not in self.hist_kwargs:
            # set default number of bins to 20
            self.hist_kwargs['bins'] = 20
        self.scatter_kwargs = scatter_kwargs
        self.levels = contour_levels
        self.showpoints = showpoints
        self.showcontours = showcontours
        self.contour_kwargs = contour_kwargs
        self.show_contour_levels = show_contour_levels
        self.limits = limits

        self._add_plots(data, label=legend)

    def _add_plots(self, data, label=None):
        """
        Add histogram and joint plots to the figure using data
        
        Label is a legend label if required.
        """

        # the vertical histogram
        self.histvert[-1].hist(data[:,-1], normed=True, orientation='horizontal', label=label, **self.hist_kwargs)
        if self.truths != None:
            self.histvert[-1].axhline(self.truths[-1], color='k', ls='--', lw=1.5)

        # put legend in the upper right plot
        _, l1 = self.histvert[-1].get_legend_handles_labels()
        if self.legend_labels != None:
            if self.hist_kwargs['histtype'] == 'stepfilled':
                lc = self.hist_kwargs['edgecolor']
            else:
                lc = self.hist_kwargs['color']
            self.legend_labels.append(Line2D([], [], linewidth=self.hist_kwargs['linewidth'], color=lc)) # create fake line for legend (to use line rather than a box)
        self.legendaxis.legend(self.legend_labels, l1, loc='lower left')
        if self.labels != None:
            self.histvert[-1].set_ylabel(self.labels[-1])
        
        if self.showpoints:
            # set default scatter plot kwargs
            if 'color' in self.hist_kwargs:
                c = self.hist_kwargs['color']
            elif 'fc' in self.hist_kwargs and self.hist_kwargs['histtype'] == 'stepfilled':
                c = self.hist_kwargs['fc'][0:3]
            else:
                c = 'b'

            these_scatter_kwargs = {'c': c, 'marker': 'o', 's': 20, 'alpha': 0.05, 'edgecolors': 'none'}
            
            for key in self.scatter_kwargs.keys():
                these_scatter_kwargs[key] = self.scatter_kwargs[key]
            self.scatter_kwargs = these_scatter_kwargs
        
        if self.limits != None:
          if len(self.limits) != self.ndims:
            raise("Error... number of dimensions is not the same as the number of limits being set")
        
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
            self.histhori[i].hist(data[:,i], normed=True, **self.hist_kwargs)
            if self.labels != None:
                self.histhori[i].set_xlabel(self.labels[i])
            if self.truths != None:
                self.histhori[i].axvline(self.truths[i], color='k', ls='--', lw=1.5)

            for j in range(i+1):
                if self.labels != None:
                    if j == 0:
                        self.histvert[rowcount].set_ylabel(self.labels[i+1])
                        rowcount += 1

                # get joint axes indices
                self.jointaxes_indices.append((j, i+1))

                if self.showpoints:
                    self.jointaxes[jointcount].scatter(data[:,j], data[:,i+1], **self.scatter_kwargs) # plot scatter

                if self.showcontours:
                    #self.plot_2d_contours(self.jointaxes[jointcount], np.vstack((data[:,j], data[:,i+1])).T)
                    xlow = xhigh = ylow = yhigh = None # default limits
                    if self.limits != None:
                      if len(self.limits[j]) == 2:
                        xlow = self.limits[j][0]
                        xhigh = self.limits[j][1]
                      if len(self.limits[i+1]) == 2:
                        ylow = self.limits[i+1][0]
                        yhigh = self.limits[i+1][1]

                    self.plot_bounded_2d_kde_contours(self.jointaxes[jointcount], np.vstack((data[:,j], data[:,i+1])).T, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)

                if self.truths != None:
                    self.jointaxes[jointcount].plot(self.truths[j], self.truths[i+1], 'kx')

                jointcount += 1

        self._format_axes()

    def _format_axes(self):
        """
        Set some formatting of the axes
        """
        pl.draw() # force labels to be drawn
        
        # move exponents into label
        for i, ax in enumerate(self.histvert):
            #[l.set_rotation(45) for l in ax.get_yticklabels()]
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=self.use_math_text))
            self.format_exponents_in_label_single_ax(ax.yaxis)

            # set limits
            if self.limits != None:
                if len(self.limits[self.histvert_indices[i]]) == 2:
                    ymin, ymax = ax.get_ylim() # get current limits
                    yminnew, ymaxnew = self.limits[self.histvert_indices[i]]
                    if yminnew == None:
                        yminnew = ymin
                    if ymaxnew == None:
                        ymaxnew = ymax
                    dy = 0.025*(ymaxnew-yminnew) # add a little bit of space
                    ax.set_ylim([yminnew-dy, ymaxnew+dy])

        for i, ax in enumerate(self.histhori):
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=self.use_math_text))
            self.format_exponents_in_label_single_ax(ax.xaxis)

            # set limits
            if self.limits != None:
                if len(self.limits[self.histhori_indices[i]]) == 2:
                    xmin, xmax = ax.get_xlim() # get current limits
                    xminnew, xmaxnew = self.limits[self.histhori_indices[i]] 
                    if xminnew == None:
                        xminnew = xmin
                    if xmaxnew == None:
                        xmaxnew = xmax
                    dx = 0.025*(xmaxnew-xminnew) # add a little bit of space
                    ax.set_xlim([xminnew-dx, xmaxnew+dx])
        
        # remove any offset text from shared axes caused by the scalar formatter for MathText
        for i, ax in enumerate(self.jointaxes):
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)

            if self.limits != None:
                if len(self.limits[self.jointaxes_indices[i][0]]) == 2:
                    xmin, xmax = ax.get_xlim() # get current limits
                    xminnew, xmaxnew = self.limits[self.jointaxes_indices[i][0]]
                    if xminnew == None:
                        xminnew = xmin
                    if xmaxnew == None:
                        xmaxnew = xmax
                    dx = 0.02*(xmaxnew-xminnew) # add a little bit of space
                    ax.set_xlim([xminnew-dx, xmaxnew+dx])
                if len(self.limits[self.jointaxes_indices[i][1]]) == 2:
                    ymin, ymax = ax.get_ylim() # get current limits
                    yminnew, ymaxnew = self.limits[self.jointaxes_indices[i][1]]
                    if yminnew == None:
                        yminnew = ymin
                    if ymaxnew == None:
                        ymaxnew = ymax
                    dy = 0.02*(ymaxnew-yminnew) # add a little bit of space
                    ax.set_ylim([yminnew-dy, ymaxnew+dy])

    def plot_bounded_2d_kde_contours(self, ax, pts, xlow=None, xhigh=None, ylow=None, yhigh=None, transform=None, gridsize=250, clip=None):
        """Function (based on that by Will Farr (@farr) and Ben Farr (@bfarr)) for plotting contours from a bounded 2d KDE"""

        if transform is None:
            transform = lambda x: x
        
        # Determine the clipping
        if clip is None:
            clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
        elif np.ndim(clip) == 1:
            clip = [clip, clip]

        # Calculate the KDE
        Npts = pts.shape[0]
        kde_pts = transform(pts[:Npts/2,:])
        den_pts = transform(pts[Npts/2:,:])

        Nden = den_pts.shape[0]

        post_kde = Bounded_2d_kde(kde_pts, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)
        den = post_kde(den_pts)
        densort = np.sort(den)[::-1]

        zvalues=[]
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
            self.contour_kwargs['alpha'] = level # set tranparency to the contour level
            cset = ax.contour(xx, yy, z, [zvalues[k]], **self.contour_kwargs)
            self.contour_kwargs['alpha'] = alpha

            # Add white outlines
            if self.contour_kwargs['colors'] == 'k':
                pl.setp(cset.collections, path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="w")])
            fmt = {}
            #strs = ['{}%'.format(int(100*level)) for level in self.levels]
            fmt[cset.levels[0]] = '{}%'.format(int(100*level))
            #for l, s in zip(cset.levels, strs):
            #    fmt[l] = s
            
            if self.show_contour_levels:
                pl.clabel(cset, cset.levels, fmt=fmt, fontsize=11, **self.contour_kwargs)#, use_clabeltext=True)
                pl.setp(cset.labelTexts, color='k', path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="w")])

    def _check_alpha(self):
        # use answer from http://stackoverflow.com/a/28398471/1862861 to have alpha transparency on hist patches, but not on edges
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
        """ Method to transform given label into the new label (by Greg Ashton) """
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
        #s = s.replace("-", "\\textrm{-}")
        return s

    def format_exponents_in_label_single_ax(self, ax):
        """ Routine for a single axes instance (by Greg Ashton) """

        exponent_text = ax.get_offset_text().get_text()
        exponent_text = exponent_text.replace("\\mathdefault", "")
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(self.update_label(label, exponent_text))
            
    def savefig(self, filename):
        self.fig.savefig(filename)

    def get_fig(self):
        return self.fig

