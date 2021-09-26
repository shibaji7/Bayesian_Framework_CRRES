"""plotlib.py: Module is used to implement various plotting functions"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, Chakraborty"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, num2date
from matplotlib import patches
import matplotlib.patches as mpatches
from matplotlib.dates import date2num
import datetime as dt

import pandas as pd

class FrequencyTimePlot(object):
    """
    Create plots for spactral datasets.
    """
    def __init__(self, dates, WFR, num_subplots, fig_title="Daily Summary: {date}"):
        self.dates = dates
        self.WFR = WFR
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        fig_title = fig_title.format(date=self.dates[0].strftime("%Y-%m-%d"))
        self.fig = plt.figure(figsize=(8, 3*self.num_subplots), dpi=100) # Size for website
        plt.suptitle(fig_title, x=0.9, y=0.95, ha="right", fontweight="bold", fontsize=12)
        mpl.rcParams.update({"font.size": 10})
        return
        
    def addParamPlot(self, Z, title, vmax=1e-5, vmin=1e-9, steps=3, cmap = plt.cm.Spectral, xlabel="Time UT",
                     ylabel="Frequency, Hz", label=r"[$nT^2Hz^{-1}$]", ax=None, fig=None, add_colbar=True):
        if fig is None: fig = self.fig
        if ax is None: ax = self._add_axis()
        if vmax is None: vmax = Z.max()
        if vmin is None: vmin = Z.min()
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12})
        ax.set_ylabel(ylabel, fontdict={"size":12})
        ax.set_xlim([self.dates[0], self.dates[-1]])
        ax.set_ylim([self.WFR["frequencies"][0], self.WFR["frequencies"][-1]])
        X, Y = np.meshgrid(self.dates, self.WFR["frequencies"])
        ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, norm=norm)
        ax.set_yscale("log")
        if add_colbar: self._add_colorbar(fig, ax, norm, cmap, label=title+" "+label)
        ax.set_title(title, loc="left")
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")

    def close(self):
        self.fig.clf()
        plt.close()

    # Private helper functions

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.tick_params(axis="both", labelsize=12)
        return ax

    def _add_colorbar(self, fig, ax, norm, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param bounds:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        spacing="uniform",
                                        orientation="vertical")
        cb2.set_label(label)
        return

def get_gridded_parameters(q, xparam="x", yparam="y", zparam="z"):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    plotParamDF[xparam] = plotParamDF[xparam].tolist()
    plotParamDF[yparam] = np.round(plotParamDF[yparam].tolist(), 1)
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z
    
class RangeTimePlot(object):
    """
    Create plots for wave datasets.
    """
    
    def __init__(self, dates, num_subplots, fig_title="Summary: {date}"):
        self.dates = dates
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        fig_title = fig_title.format(date=self.dates[0].strftime("%Y.%m.%d") + "-" + self.dates[-1].strftime("%m.%d"))
        self.fig = plt.figure(figsize=(8, 3*self.num_subplots), dpi=150) # Size for website
        plt.suptitle(fig_title, x=0.9, y=0.95, ha="right", fontweight="bold", fontsize=12)
        mpl.rcParams.update({"font.size": 10})
        return
    
    def addParamPlot(self, x, y, z, title="", vmax=1e2, vmin=1e0, steps=3, cmap = plt.cm.Spectral_r, xlabel="Time UT",
                     ylabel="L", label=r"$B_{chorus}$[pT]", ax=None, fig=None, add_colbar=True, 
                     interpolate_params={"dt":"1T"}):
        if fig is None: fig = self.fig
        if ax is None: ax = self._add_axis()
        if vmax is None: vmax = Z.max()
        if vmin is None: vmin = Z.min()
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        cmap.set_bad("w", alpha=0.0)
        df = pd.DataFrame()
        df["x"], df["y"], df["z"] = x, y, z
        df = df.set_index("x").resample(interpolate_params["dt"]).max().reset_index()
        df["x"] = df.x.apply(lambda k: date2num(k))
        X, Y, Z = get_gridded_parameters(df)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter(r"$%d$"))
        ax.xaxis.set_minor_formatter(DateFormatter(r"$%H^{%M}$"))
        hours = mdates.HourLocator(byhour=[12])
        ax.xaxis.set_minor_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12})
        ax.set_ylabel(ylabel, fontdict={"size":12})
        ax.set_xlim([self.dates[0], self.dates[-1]+dt.timedelta(1)])
        ax.set_ylim(1.5, 6.5)
        ax.pcolormesh(X, Y, Z.T, lw=4., edgecolors="None", cmap=cmap, norm=norm)
        if add_colbar: self._add_colorbar(fig, ax, norm, cmap, label=title+" "+label)
        ax.set_title(title, loc="left")
        return
    
    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.tick_params(axis="both", labelsize=12)
        return ax

    def _add_colorbar(self, fig, ax, norm, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param bounds:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        spacing="uniform",
                                        orientation="vertical")
        cb2.set_label(label)
        return
    
    def close(self):
        self.fig.clf()
        plt.close()
        return
    
    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return