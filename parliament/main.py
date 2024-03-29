"""
main
~~~~

Main analysis code
"""
import argparse
from copy import copy
import numbers
from pathlib import Path
import re
import sys
import traceback
from warnings import warn

from bs4 import BeautifulSoup
import colorcet as cc
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.stats import median_abs_deviation
from scipy.stats.mstats import winsorize
from seaborn.categorical import _BoxPlotter
from seaborn.utils import remove_na
import seaborn as sns

from parliament.analyze import FileCalculations

tc_pat = re.compile(r'_(ar|bru|fuz|ikd|lrn|vic|wri)')


# taken from skimage
def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).
    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.
    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.
    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle
    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_windows
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])
    >>> A = np.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = np.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def _rolling_func(vals, win_size, func):
    strides = view_as_windows(vals, (win_size,), step=1)
    tmp = func(strides, axis=1)
    return np.append([np.nan]*(win_size-1), tmp)


def rolling_nan_median(vals, win_size):
    return _rolling_func(vals, win_size, np.nanmedian)


def rolling_nan_mean(vals, win_size):
    return _rolling_func(vals, win_size, np.nanmean)


def sequential_nan_median(vals, win_size):
    # create strides as columns. Then compute median for each stride. Is quite time efficient.
    strides = view_as_windows(vals, (win_size,), step=win_size)
    meds = np.nanmedian(strides, axis=1)
    # these next 2 lines fill in nans between sequence val and next sequence val
    tmp = np.expand_dims(meds[:-1], axis=0)
    tmp = np.pad(tmp.T, ((0, 0), (0, win_size-1)), constant_values=np.nan).ravel()
    tmp = np.append(tmp, [meds[-1]])
    # this line makes sure any remaining nans are filled.
    return np.pad(tmp, (win_size-1, len(vals)-(len(tmp)+win_size-1)), constant_values=np.nan)


class ResultsContainer(object):

    def __init__(self, experiment_name, window_n, no_algo_restrict):
        self.proc_results = []
        self.algos_used = []
        self.experiment_name = experiment_name
        self.raw_results = []
        self.no_algo_restrict = no_algo_restrict
        self.results_dir = Path(__file__).parent.joinpath('results', experiment_name)
        # you can always crank the dpi up for paper time
        self.dpi = 200
        self.boot_resamples = 100
        self.window_n = window_n
        self.full_analysis_done = False
        self.scatter_marker_symbols = [
            'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'P', 'X', 'D', 'd', 'H',
            '$\Join$', '$\clubsuit$', '$\spadesuit$', '$\heartsuit$', '$\$$',
            '$\dag$', '$\ddag$', '$\P$'
        ]
        self.algo_markers = {}
        self.algo_colors = {
            'al_rawas': cc.cm.glasbey(0),
            'al_rawas_ar': cc.cm.glasbey(21),
            'al_rawas_bru': cc.cm.glasbey(16),
            'al_rawas_fuz': cc.cm.glasbey(17),
            'al_rawas_ikd': cc.cm.glasbey(18),
            'al_rawas_lren': cc.cm.glasbey(19),
            'al_rawas_vic': cc.cm.glasbey(14),
            'al_rawas_wri': cc.cm.glasbey(20),
            'ft_insp_lstsq': cc.cm.glasbey(1),
            'howe_lstsq': cc.cm.glasbey(2),
            'iimipr': cc.cm.glasbey(3),
            'iipr': cc.cm.glasbey(4),
            'iipredator': cc.cm.glasbey(5),
            'kannangara': cc.cm.glasbey(6),
            'major': cc.cm.glasbey(7),
            'mipr': cc.cm.glasbey(8),
            'polynomial': cc.cm.glasbey(9),
            'predator': cc.cm.glasbey(10),
            'pt_exp_lstsq': cc.cm.glasbey(11),
            'pt_insp_lstsq': cc.cm.glasbey(12),
            'vicario_co': cc.cm.glasbey(13),
            'vicario_nieap': cc.cm.glasbey(14),
            'vicario_nieap_ar': cc.cm.glasbey(21),
            'vicario_nieap_bru': cc.cm.glasbey(16),
            'vicario_nieap_fuz': cc.cm.glasbey(17),
            'vicario_nieap_ikd': cc.cm.glasbey(18),
            'vicario_nieap_lren': cc.cm.glasbey(19),
            'vicario_nieap_vic': cc.cm.glasbey(14),
            'vicario_nieap_wri': cc.cm.glasbey(20),
        }
        self.pp_frames = {}
        self.bb_frames = {}
        self.label_abs_diff = 'Absolute Difference ($|C_{rs}^k-\hat{C}_{rs}^k|$)'
        self.label_diff = 'Difference ($C_{rs}^k-\hat{C}_{rs}^k$)'
        self.scatter_marker_symbols = {
            'al_rawas': 'o',
            'al_rawas_ar': 'o',
            'al_rawas_bru': 'o',
            'al_rawas_fuz': 'o',
            'al_rawas_ikd': 'o',
            'al_rawas_lren': 'o',
            'al_rawas_vic': 'o',
            'al_rawas_wri': 'o',
            'ft_insp_lstsq': 'v',
            'howe_lstsq': '^',
            'iimipr': '<',
            'iipr': '>',
            'iipredator': 's',
            'kannangara': 'p',
            'major': '*',
            'mipr': 'h',
            'polynomial': 'P',
            'predator': 'X',
            'pt_exp_lstsq': 'D',
            'pt_insp_lstsq': 'd',
            'vicario_co': 'H',
            'vicario_nieap': '$\Join$',
            'vicario_nieap_ar': '$\Join$',
            'vicario_nieap_bru': '$\Join$',
            'vicario_nieap_fuz': '$\Join$',
            'vicario_nieap_ikd': '$\Join$',
            'vicario_nieap_lren': '$\Join$',
            'vicario_nieap_vic': '$\Join$',
            'vicario_nieap_wri': '$\Join$',
        }

    def _compare_breath_level_masks(self, df1, df2, windowing1, windowing2, algos, mask1_name, mask2_name, figname, absolute, **kwargs):
        """
        Private method for comparing breath by breath masks.
        """
        figname = str(self.results_dir.joinpath(figname).resolve())
        algos_in_frame = set(self.proc_results.columns).intersection(self.algos_used) if algos is None else algos

        if windowing1 in ['smd', 'wmd']:
            diff_colname_suffix1 = '_{}_{}'.format(windowing1, self.window_n)
        else:
            diff_colname_suffix1 = '_diff'

        if windowing2 in ['smd', 'wmd']:
            diff_colname_suffix2 = '_{}_{}'.format(windowing2, self.window_n)
        else:
            diff_colname_suffix2 = '_diff'
        sorted_diff_cols1 = sorted(["{}{}".format(algo, diff_colname_suffix1) for algo in algos_in_frame])
        sorted_diff_cols2 = sorted(["{}{}".format(algo, diff_colname_suffix2) for algo in algos_in_frame])

        bb1 = df1[sorted_diff_cols1].melt()
        bb2 = df2[sorted_diff_cols2].melt()
        bb1['Mask'] = mask1_name
        bb2['Mask'] = mask2_name
        df = pd.concat([bb1, bb2])
        df = df.rename(columns={'variable': 'algo'})
        df = df.dropna()
        if absolute:
            df['value'] = df['value'].abs()
        df.algo = df.algo.str.replace('(_diff|_(wmd|smd)_\d+)', '')

        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (kwargs.get('fig_width', 3*8), kwargs.get('fig_height', 3*3))))

        # alphabetical order again
        sns.boxplot(x='algo', y='value', data=df, hue='Mask', ax=ax, notch=False, bootstrap=None, showfliers=False, palette=kwargs.get('palette', 'Set2'), linewidth=kwargs.get('box_lw', None), zorder=kwargs.get('bar_zorder', 2))
        xtick_names = plt.setp(ax, xticklabels=[FileCalculations.shorthand_name_mapping[algo._text] for algo in ax.get_xticklabels()])
        plt.setp(xtick_names, rotation=kwargs.get('rotation', 60), fontsize=kwargs.get('tick_fontsize', 14))
        plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('tick_fontsize', 14))
        ax.set_ylabel(self.label_diff, fontsize=kwargs.get('label_fontsize', 18))
        ax.set_xlabel('')
        ax.legend(fontsize=kwargs.get('legend_fontsize', 18), loc=kwargs.get('legend_loc', 'best'), title=kwargs.get('legend_title', 'Breath Type'), title_fontsize=kwargs.get('legend_fontsize', 18))
        title = '{} vs {}'.format(mask1_name, mask2_name)
        ax.set_title(title, fontsize=kwargs.get('title_fontsize', 18), pad=25)
        ax.grid(True, lw=kwargs.get('grid_lw', 1), alpha=kwargs.get('grid_alpha', None), axis='y')

        proc_frame = self.extract_medians_and_iqr(df1, windowing1)
        self._show_breath_by_breath_algo_table(proc_frame, mask1_name)

        proc_frame = self.extract_medians_and_iqr(df2, windowing2)
        self._show_breath_by_breath_algo_table(proc_frame, mask2_name)

        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], ls='--', zorder=kwargs.get('line_zorder', 0.9), c='red', lw=kwargs.get('lw', 3))
        ax.set_xlim(xlim)
        ylim = ax.get_ylim()
        ax.set_ylim(kwargs.get('ylim', ylim))
        plt.tight_layout()
        plt.savefig(figname, dpi=self.dpi)
        plt.show(fig)

    def _draw_seaborn_boxplot(self, data, ax, medians=None, palette=None, linewidth=None, **kwargs):
        """
        Basically an exact replica of what happens in seaborn except for support of
        usermedians
        """
        plotter = _BoxPlotter(x=None, y=None, hue=None, data=data, order=None, hue_order=None,
                              orient=None, color=None, palette=palette, saturation=.75, width=.8,
                              dodge=True, fliersize=5, linewidth=linewidth)
        kwargs.update(dict(whis=1.5, showfliers=False))
        vert = plotter.orient == "v"

        props = {}
        for obj in ["box", "whisker", "cap", "median", "flier"]:
            props[obj] = kwargs.pop(obj + "props", {})

        for i, group_data in enumerate(plotter.plot_data):

            # Draw a single box or a set of boxes
            # with a single level of grouping
            box_data = np.asarray(remove_na(group_data))

            if medians is not None:
                artist_dict = ax.boxplot(box_data,
                                         vert=vert,
                                         patch_artist=True,
                                         positions=[i],
                                         widths=plotter.width,
                                         usermedians=[medians[i]],
                                         **kwargs)
            else:
                artist_dict = ax.boxplot(box_data,
                                         vert=vert,
                                         patch_artist=True,
                                         positions=[i],
                                         widths=plotter.width,
                                         **kwargs)
            color = plotter.colors[i]
            plotter.restyle_boxplot(artist_dict, color, props)

    def _bootstrap(self, col_vals):
        """
        So the bootstrapping procedure here is that we bootstrap the
        vector N times, then take the median from each vector n_i. After
        that we take the median again to return final result. We perform
        this same process with the IQR as well.
        """
        M = len(col_vals)
        percentiles = (25, 75)

        if self.boot_resamples >= 1:
            boot_index = np.random.randint(M, size=(self.boot_resamples, M))
            boot_data = col_vals[boot_index]
        else:  # for testing circumstances. the repeat is to simulate a bootstrap
            boot_data = col_vals[np.expand_dims(np.arange(M), axis=0).repeat(2, axis=0)]

        estimate = np.nanmedian(boot_data, axis=1)
        # this is basically the same thing that scipy.stats.iqr does.
        # Also we need to take a median of the IQR here because we are performing bootstrapping.
        # axis=1 is used here because this is the axis with the actual data. axis=0 contains
        # the N bootstraps. if you took over axis=0 then you would just be taking median
        # across [breath 1 for N bootstraps, breath 2 for N bootstraps, .. etc]
        iqr = np.median(np.nanpercentile(boot_data, percentiles, axis=1), axis=1)
        return np.median(estimate), iqr[0], iqr[1]

    def _change_td_to_bold(self, soup, td):
        """
        Change td tag so that interior text is boldfaced.
        """
        val = td.string
        td.clear()
        td.insert(0, soup.new_tag('b'))
        td.b.string = val

    def _create_per_patient_comparison_frames(self, ad_std1, ad_std2, hue_colname, hue1, hue2, std_or_mad):
        ad_rows = []
        for algo, items in ad_std1.items():
            algo_name = FileCalculations.shorthand_name_mapping[algo]
            for val in items[0]:
                ad_rows.append([algo_name, val, hue1])

        for algo, items in ad_std2.items():
            algo_name = FileCalculations.shorthand_name_mapping[algo]
            for val in items[0]:
                ad_rows.append([algo_name, val, hue2])

        # create frame for std.
        std_rows = []
        for algo, items in ad_std1.items():
            algo_name = FileCalculations.shorthand_name_mapping[algo]
            for val in items[1]:
                std_rows.append([algo_name, val, hue1])

        for algo, items in ad_std2.items():
            algo_name = FileCalculations.shorthand_name_mapping[algo]
            for val in items[1]:
                std_rows.append([algo_name, val, hue2])

        if std_or_mad == 'std':
            ylabel = 'Standard Deviation'
        elif std_or_mad == 'mad':
            ylabel = 'Median Absolute Deviation'

        xlabel = 'Absolute Difference'
        ad_df = pd.DataFrame(ad_rows, columns=['Algorithm', xlabel, hue_colname])
        std_df = pd.DataFrame(std_rows, columns=['Algorithm', ylabel, hue_colname])
        return ad_df, std_df

    def _get_async_mask_name_for_compare_breath_level_masks(self, asynchrony_type):
        return {
            None: 'Asynchronous',
            'async_no_fam': 'Asynchronous (no mild FA)',
            'fa_no_fam': 'Moderate/Severe FA',
            'fa': 'FA',
            'fa_mod': 'Moderate FA',
            'fa_mild': 'Mild FA',
            'fa_sev': 'Severe FA',
            'dta': 'Double Trigger Asynchrony',
            'bsa': 'Breath Stacking Asynchrony',
            'dca': 'Delayed Cycling Asynchrony',
            'async_no_dca': 'Asynchronous, No DCA',
            'async_no_fa': 'Asynchronous, No FA',
        }[asynchrony_type]

    def _get_windowing_algo_diff_colnames(self, windowing):
        algos_in_frame = set(self.proc_results.columns).intersection(self.algos_used)
        if windowing in ['smd', 'wmd']:
            diff_colname_suffix = '_{}_{}'.format(windowing, self.window_n)
        else:
            diff_colname_suffix = '_diff'
        return sorted(["{}{}".format(algo, diff_colname_suffix) for algo in algos_in_frame])

    def _get_windowing_colnames(self, windowing, std_or_mad):

        if std_or_mad not in ['std', 'mad']:
            raise Exception('For std_or_mad you must either input "mad" or "std"')

        if windowing is None:
            ad_col = 'ad_pt'
            dev_col = '{}_pt'.format(std_or_mad)
        elif windowing == 'wmd':
            ad_col = 'ad_wmd'
            dev_col = '{}_wmd'.format(std_or_mad)
        elif windowing == 'smd':
            ad_col = 'ad_smd'
            dev_col = '{}_smd'.format(std_or_mad)
        else:
            raise ValueError('windowing variable must be None, "wmd", or "smd"')
        return ad_col, dev_col

    def _ad_std_scatter(self, ad_std, windowing, plt_title, figname, individual_patients, std_lim, std_or_mad, custom_xlabel=None, custom_ylabel=None, highlight_algos=None, **kwargs):
        """
        Perform scatter plot using with AD and std information for each algorithm.
        """
        algos_in_order = sorted(list(ad_std.keys()))
        algo_dict = {algo: {'m': self.algo_markers[algo], 'c': self.algo_colors[algo]} for algo in algos_in_order}
        fig, ax = plt.subplots(figsize=(3*6.5, 3*2.5))

        if individual_patients:
            for i, algo in enumerate(algos_in_order):
                ax.scatter(
                    x=ad_std[algo][0],
                    y=ad_std[algo][1],
                    marker=algo_dict[algo]['m'],
                    color=algo_dict[algo]['c'],
                    label=algo,
                    alpha=0.4,
                    s=100,
                    zorder=1
                )

        for i, algo in enumerate(algos_in_order):
            if highlight_algos and algo in highlight_algos:
                s = kwargs.get('marker_hl_size', 500)
                m_lw = kwargs.get('marker_hl_lw', 1)
                m_ec = kwargs.get('marker_hl_ec', 'black')
                m_alph = kwargs.get('marker_hl_alpha', .9)
            else:
                s = kwargs.get('main_marker_size', 350)
                m_lw = kwargs.get('main_marker_lw', 1)
                m_ec = 'black'
                m_alph = kwargs.get('main_marker_alpha', .9)
            ax.scatter(
                x=ad_std[algo][2],
                y=ad_std[algo][3],
                marker=algo_dict[algo]['m'],
                color=algo_dict[algo]['c'],
                label=algo if not individual_patients else None,
                alpha=m_alph,
                s=s,
                edgecolors=m_ec,
                zorder=len(algos_in_order)+2-i,
                linewidths=m_lw,
            )
        x = [ad_std[a][2] for a in algos_in_order]
        y = [ad_std[a][3] for a in algos_in_order]
        ax.tick_params(axis='x', labelsize=kwargs.get('tick_fontsize', 14))
        ax.tick_params(axis='y', labelsize=kwargs.get('tick_fontsize', 14))
        xlabel = 'Median Absolute Difference ($|C_{rs}^k-\hat{C}_{rs}^k|$)'
        if custom_ylabel:
            ylabel = custom_ylabel
            dev_field_name = 'Standard Deviation (std)'
        elif std_or_mad == 'std':
            ylabel = 'Standard Deviation ($\sigma$)'
            dev_field_name = 'Standard Deviation (std)'
        elif std_or_mad == 'mad':
            ylabel = 'MAD'
            dev_field_name = 'Median Absolute Deviation (MAD)'
        ax.set_ylabel(ylabel, fontsize=kwargs.get('label_fontsize', 22), labelpad=kwargs.get('ylabel_pad', 4.0))
        ax.set_xlabel(xlabel, fontsize=kwargs.get('label_fontsize', 22), labelpad=kwargs.get('xlabel_pad', 4.0))
        if std_lim is not None and len(x) > 1:
            ax.set_xlim(-.1, np.mean(x)+std_lim*np.std(x))
            ax.set_ylim(-.4, np.mean(y)+std_lim*np.std(y))

        # draw a black line across origin lines in dark black
        preset_xlim = ax.get_xlim()
        preset_ylim = ax.get_ylim()
        ax.plot(preset_xlim, [0, 0], color='black', zorder=0, lw=2)
        ax.plot([0, 0], preset_ylim, color='black', zorder=0, lw=2)
        ax.set_xlim(preset_xlim)
        ax.set_ylim(preset_ylim)
        ax.set_title(plt_title, fontsize=20, pad=30)
        ax.grid(True, lw=kwargs.get('grid_lw', 1), alpha=kwargs.get('grid_alpha', None))

        handles, labels = ax.get_legend_handles_labels()
        new_labels = [FileCalculations.shorthand_name_mapping[lab] for lab in labels]
        fig.legend(handles, new_labels, fontsize=kwargs.get('legend_fontsize', 17), loc=kwargs.get('legend_loc', 'center right'), framealpha=kwargs.get('legend_frame_alpha', .4), title=kwargs.get('legend_title', None), title_fontsize=kwargs.get('legend_fontsize', 17))

        # show table of scatter results
        table = PrettyTable()
        table.field_names = ['Algorithm', 'Shorthand Name', 'Absolute Difference', dev_field_name]
        medians = np.array([ad_std[algo][2] for algo in algos_in_order]).round(2)
        devs = np.array([ad_std[algo][3] for algo in algos_in_order]).round(2)
        for i, algo in enumerate(algos_in_order):
            table.add_row([FileCalculations.algo_name_mapping[algo], algo, medians[i], devs[i]])

        if np.isnan(medians).all():
            plt.close()
            return
        soup = BeautifulSoup(table.get_html_string())
        min_median = np.nanargmin(medians)
        min_dev = np.nanargmin(devs)
        # the +1 is because the header is embedded in a <tr> element
        min_med_elem = soup.find_all('tr')[min_median+1]
        min_dev_elem = soup.find_all('tr')[min_dev+1]

        self._change_td_to_bold(soup, min_med_elem.find_all('td')[2])
        self._change_td_to_bold(soup, min_dev_elem.find_all('td')[3])

        display(HTML('<h2>{}</h2>'.format(plt_title)))
        display(HTML(soup.prettify()))

        # show plot
        plt.tight_layout()
        figname = str(self.results_dir.joinpath(figname).resolve())
        fig.savefig(figname, dpi=self.dpi)
        plt.show(fig)

    def _perform_single_window_analysis(self, df, absolute, winsorizor, algos, robust, robust_and_reg, show_median):
        nrows = 4
        plot_data = [
            (0, 0, 'asynci_{}'.format(self.window_n)),
            (0, 1, 'asynci_no_fam_{}'.format(self.window_n)),
            (1, 0, 'dti_{}'.format(self.window_n)),
            (1, 1, 'bsi_{}'.format(self.window_n)),
            (2, 0, 'dci_{}'.format(self.window_n)),
            (2, 1, 'fai_{}'.format(self.window_n)),
            (3, 0, 'fai_no_fam_{}'.format(self.window_n)),
            (3, 1, 'dtw_wm_{}'.format(self.window_n)),
        ]
        algos = algos if algos != [] else self.algos_used
        # for now just run some of the least squares algos
        for algo in algos:
            # dims are in wxh
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(3*8, 3*nrows*3))
            wmd_colname = '{}_wmd_{}'.format(algo, self.window_n)
            for i, j, col in plot_data:
                scatter_kws = {'s': 2, 'alpha': .5, 'edgecolors': 'black', 'color': 'blue'}
                resp = self._regplot_wmd(df, algo, col, self.window_n, axes[i][j], winsorizor, scatter_kws, {'label': 'tmp', 'lw': 5}, absolute, robust, robust_and_reg, show_median)
                if not resp:
                    continue

                handles, labels = axes[i][j].get_legend_handles_labels()
                if not robust_and_reg:
                    x, y = axes[i][j].lines[0].get_data()
                    slope = round((y[-1] - y[0]) / (x[-1] - x[0]), 2)
                    new_labels = ['n: {} slope: {}'.format(self.window_n, slope)]
                else:
                    new_labels = labels
                if not robust_and_reg and len(labels) > 1:
                    new_labels += labels[1:]
                axes[i][j].legend(handles, new_labels, fontsize=16)

            plt.suptitle(FileCalculations.algo_name_mapping[algo] + ' n: {}'.format(self.window_n), fontsize=28)
            plt.tight_layout()
            plt.show(fig)

    def _regplot_wmd(self, df, algo, x_col, win_size, ax, winsorizor, scatter_kws, line_kws, absolute, robust, robust_and_reg, show_median, show_algo_color=True):
        """
        Perform regression plotting for WMD versus some index

        :param df: DataFrame of results we want to analyze
        """

        # At least for now just skip the plot if its not sensible to perform
        if (('fai' in x_col and algo in FileCalculations.algos_unavailable_for_vc) or \
            ('dci' in x_col and algo in FileCalculations.algos_unavailable_for_pc_prvc)) and not self.no_algo_restrict:
            ax.set_xlim((-1, 1))
            ax.annotate('N/A due to algo restrictions', (-.5, 0), fontsize=22)
            ax.set_ylim((-1, 1))
            ax.set_ylabel('')
            xlabel = x_col.replace('_', ' ').replace(str(win_size), '').strip()
            ax.set_xlabel(xlabel)
            return False

        abs_lmda = lambda x: np.abs(x) if absolute else x
        line_kws.setdefault('lw', 4)
        data = df.copy()
        wmd_colname = '{}_wmd_{}'.format(algo, win_size)
        data[wmd_colname] = abs_lmda(data[wmd_colname])
        data[wmd_colname] = winsorize(data[wmd_colname].values, limits=winsorizor)
        color = self.algo_colors[algo] if show_algo_color else None
        if not robust_and_reg:
            sns.regplot(
                x=x_col,
                y=wmd_colname,
                data=data,
                scatter_kws=scatter_kws,
                line_kws=line_kws,
                ax=ax,
                robust=robust,
                n_boot=self.boot_resamples,
                color=color,
            )
        else:
            line_kws['label'] = 'non-robust'
            sns.regplot(
                x=x_col,
                y=wmd_colname,
                data=data,
                scatter_kws=scatter_kws,
                line_kws=line_kws,
                ax=ax,
                robust=False,
                n_boot=self.boot_resamples,
                color=color,
            )
            # regplot on robust is taking up all the time. Why?
            line_kws['label'] = 'robust'
            sns.regplot(
                x=x_col,
                y=wmd_colname,
                data=data,
                scatter_kws=scatter_kws,
                line_kws=line_kws,
                ax=ax,
                robust=True,
                n_boot=self.boot_resamples,
            )

        if show_median and 'dtw' not in x_col:
            medians = []
            x = []
            x_groups = data[[x_col]].dropna().groupby(x_col)
            for num, df in x_groups:
                medians.append(data.loc[df.index, wmd_colname].median())
                x.append(num)
            ax.plot(x, medians, lw=line_kws['lw'], label='median', zorder=0)
        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], ls='--', zorder=0, c='red', lw=line_kws['lw'])
        ax.set_xlim(xlim)
        ax.set_ylabel('Difference (estimated v. true)')
        xlabel = x_col.replace('_', ' ').replace(str(win_size), '').strip()
        ax.set_xlabel(xlabel)
        return True

    def _set_new_frame(self, dict_, name, frame):
        if name not in dict_:
            dict_[name] = {self.window_n: frame}
        else:
            dict_[name][self.window_n] = frame

    def _set_new_frames(self, name, frame):
        self._set_new_frame(self.bb_frames, name, frame)
        tmp = self.analyze_per_patient_df(frame)
        self._set_new_frame(self.pp_frames, name, tmp)

    def _show_breath_by_breath_algo_table(self, df, title):
        """
        Show table of boxplot results for breath by breath analysis of algorithms.
        """
        table = PrettyTable()
        table.field_names = ['Algorithm', 'Shorthand Name', 'Median Diff', '25% IQR', '75% IQR', 'std']
        medians = df.medians.round(2).values
        iqr_low = df.iqr_low.round(2).values
        iqr_high = df.iqr_high.round(2).values
        stds = df.stds.round(2).values
        for i, algo in enumerate(df.algo.values):
            if not np.isnan(medians[i]):
                table.add_row([FileCalculations.algo_name_mapping[algo], algo, medians[i], iqr_low[i], iqr_high[i], stds[i]])
            else:
                table.add_row([FileCalculations.algo_name_mapping[algo], algo, '-', '-', '-', '-'])

        soup = BeautifulSoup(table.get_html_string())
        min_median = np.nanargmin(abs(medians))
        min_iqr_rel_to_0 = np.nanargmin(abs(np.array([iqr_low, iqr_high]).T).sum(axis=1))
        min_std = np.nanargmin(stds)
        # the +1 is because the header is embedded in a <tr> element
        min_med_elem = soup.find_all('tr')[min_median+1]
        min_iqr_elem = soup.find_all('tr')[min_iqr_rel_to_0+1]
        min_std_elem = soup.find_all('tr')[min_std+1]

        self._change_td_to_bold(soup, min_med_elem.find_all('td')[2])
        self._change_td_to_bold(soup, min_iqr_elem.find_all('td')[3])
        self._change_td_to_bold(soup, min_iqr_elem.find_all('td')[4])
        self._change_td_to_bold(soup, min_std_elem.find_all('td')[5])

        display(HTML('<h2>{}</h2>'.format(title)))
        display(HTML(soup.prettify()))

    def _validate_async_mask_name_by_mode(self, mode, asynchrony_type):
        if mode not in ['vc', 'pressure']:
            raise Exception('mode must be set to either "vc" or "pressure"')

        mode_prefix = 'vc' if mode == 'vc' else 'pc_prvc'
        allowed_async_types = ['bsa', 'dta', 'fa_no_fam', 'dca', "fa_mild", 'fa_mod', 'fa_sev', 'async_no_dca', 'async_no_fa', 'async_no_fam']
        if asynchrony_type is None:
            return '{}_async_only'.format(mode_prefix)
        elif asynchrony_type == 'async_no_fam':
            return '{}_async_only_no_fam'.format(mode_prefix)
        elif asynchrony_type in allowed_async_types:
            return '{}_{}_only'.format(mode_prefix, asynchrony_type)
        else:
            raise Exception('asynchrony type {} is not valid choose from {}'.format(asynchrony_type, ', '.join(allowed_async_types)))

    @classmethod
    def load_from_experiment_name(cls, experiment_name, n_minutes):
        """
        Load results container but even easier using the container.pkl obj
        """
        results_dir = Path(__file__).parent.joinpath('results', experiment_name)
        cls = pd.read_pickle(results_dir.joinpath('ResultsContainer_mins_{}.pkl'.format(n_minutes)))
        # a bit dirty, but some older analyses dont have these attr
        cls.scatter_marker_symbols = {
            'al_rawas': 'o',
            'al_rawas_ar': 'o',
            'al_rawas_bru': 'o',
            'al_rawas_fuz': 'o',
            'al_rawas_ikd': 'o',
            'al_rawas_lren': 'o',
            'al_rawas_vic': 'o',
            'al_rawas_wri': 'o',
            'ft_insp_lstsq': 'v',
            'howe_lstsq': '^',
            'iimipr': '<',
            'iipr': '>',
            'iipredator': 's',
            'kannangara': 'p',
            'major': '*',
            'mipr': 'h',
            'polynomial': 'P',
            'predator': 'X',
            'pt_exp_lstsq': 'D',
            'pt_insp_lstsq': 'd',
            'vicario_co': 'H',
            'vicario_nieap': '$\Join$',
            'vicario_nieap_ar': 'D',
            'vicario_nieap_bru': 'D',
            'vicario_nieap_fuz': 'D',
            'vicario_nieap_ikd': 'D',
            'vicario_nieap_lren': 'D',
            'vicario_nieap_vic': 'D',
            'vicario_nieap_wri': 'D',

        }
        cls.algo_colors = {
            'al_rawas': cc.cm.glasbey(0),
            'al_rawas_ar': cc.cm.glasbey(21),
            'al_rawas_bru': cc.cm.glasbey(16),
            'al_rawas_fuz': cc.cm.glasbey(17),
            'al_rawas_ikd': cc.cm.glasbey(18),
            'al_rawas_lren': cc.cm.glasbey(19),
            'al_rawas_vic': cc.cm.glasbey(14),
            'al_rawas_wri': cc.cm.glasbey(20),
            'ft_insp_lstsq': cc.cm.glasbey(1),
            'howe_lstsq': cc.cm.glasbey(2),
            'iimipr': cc.cm.glasbey(3),
            'iipr': cc.cm.glasbey(4),
            'iipredator': cc.cm.glasbey(5),
            'kannangara': cc.cm.glasbey(6),
            'major': cc.cm.glasbey(7),
            'mipr': cc.cm.glasbey(8),
            'polynomial': cc.cm.glasbey(9),
            'predator': cc.cm.glasbey(10),
            'pt_exp_lstsq': cc.cm.glasbey(11),
            'pt_insp_lstsq': cc.cm.glasbey(12),
            'vicario_co': cc.cm.glasbey(13),
            'vicario_nieap': cc.cm.glasbey(14),
            'vicario_nieap_ar': cc.cm.glasbey(21),
            'vicario_nieap_bru': cc.cm.glasbey(16),
            'vicario_nieap_fuz': cc.cm.glasbey(17),
            'vicario_nieap_ikd': cc.cm.glasbey(18),
            'vicario_nieap_lren': cc.cm.glasbey(19),
            'vicario_nieap_vic': cc.cm.glasbey(14),
            'vicario_nieap_wri': cc.cm.glasbey(20),
        }
        cls.label_abs_diff = 'Absolute Difference ($|C_{rs}^k-\hat{C}_{rs}^k|$)'
        cls.label_diff = 'Difference ($C_{rs}^k-\hat{C}_{rs}^k$)'

        return cls

    def add_results_df(self, patient, dataframe):
        """
        Add file results to our overall results storage and perform some preprocessing.
        """
        dataframe['patient_id'] = patient
        dataframe['abs_bs'] = pd.to_datetime(dataframe['abs_bs'], format='%Y-%m-%d %H-%M-%S.%f')
        # just add the is_valid_plat col for convenience
        dataframe['is_valid_plat'] = False
        dataframe.loc[~dataframe.gold_stnd_compliance.isna(), 'is_valid_plat'] = True
        dataframe['gold_orig'] = dataframe['gold_stnd_compliance']
        self.raw_results.append(dataframe)

    def analyze_per_patient_df(self, df):
        """
        Helper method for analyze_results
        """
        row_results = []
        for patient_id, frame in df.groupby('patient_id'):
            # find AD per patient, per algo
            algos_in_frame = set(df.columns).intersection(self.algos_used)
            for algo in algos_in_frame:
                row = [patient_id, algo]

                row.append(np.nanmedian(frame['{}_diff'.format(algo, self.window_n)].abs()))
                row.append(np.nanstd(frame[algo]))
                row.append(median_abs_deviation(frame[algo], nan_policy='omit'))

                row.append(np.nanmedian(frame['{}_wmd_{}'.format(algo, self.window_n)].abs()))
                row.append(np.nanstd(frame['{}_wmd_{}'.format(algo, self.window_n)]))
                row.append(median_abs_deviation(frame['{}_wmd_{}'.format(algo, self.window_n)], nan_policy='omit'))

                row.append(np.nanmedian(frame['{}_smd_{}'.format(algo, self.window_n)].abs()))
                row.append(np.nanstd(frame['{}_smd_{}'.format(algo, self.window_n)]))
                row.append(median_abs_deviation(frame['{}_smd_{}'.format(algo, self.window_n)], nan_policy='omit'))

                row_results.append(row)

        cols = [
            'patient_id', 'algo', 'ad_pt', 'std_pt', 'mad_pt', 'ad_wmd',
            'std_wmd', 'mad_wmd', 'ad_smd', 'std_smd', 'mad_smd'
        ]
        return pd.DataFrame(row_results, columns=cols)

    def analyze_results(self):
        """
        Analyze all results obtained.

        1. Analyze results on a breath by breath basis
        2. Analyze results on a patient by patient basis
            1/2a. Obtain AD (absolute difference) between algo and gold stnd compliance
        """
        if isinstance(self.proc_results, list):
            warn('Called analyze_results before any results were collated. Call collate_data first!')
            return

        self.calc_windows(self.proc_results)
        self.calc_async_index(self.proc_results)

        masks = self.get_masks()
        async_mask = masks['async_no_fam']

        # analyze all per patient breaths
        pp_all = self.analyze_per_patient_df(self.proc_results)
        self._set_new_frame(self.pp_frames, 'all', pp_all)

        # analyze non-asynchronous breaths
        frame = self.proc_results[~async_mask]
        self._set_new_frames('no_async', frame)

        # analyze all asynchronous breathing
        frame = self.proc_results[async_mask]
        self._set_new_frames('async', frame)

        # analyze only volume control breaths
        frame = self.proc_results[self.proc_results.ventmode == 'vc']
        self._set_new_frames('vc_only', frame)

        # analyze all artifact breathing
        frame = self.proc_results[self.proc_results.artifact != 0]
        self._set_new_frames('artifacts', frame)

        # analyze non-asynchronous breathing VC only
        frame = self.proc_results[(self.proc_results.ventmode == 'vc') & ~async_mask]
        self._set_new_frames('vc_no_async', frame)

        # analyze asynchronous breathing, VC only
        frame = self.proc_results[(self.proc_results.ventmode == 'vc') & async_mask]
        self._set_new_frames('vc_only_async', frame)

        # analyze all PC only
        frame = self.proc_results[(self.proc_results.ventmode == 'pc')]
        self._set_new_frames('pc_only', frame)

        # analyze non-asynchronous breathing PC only
        frame = self.proc_results[(self.proc_results.ventmode == 'pc') & ~async_mask]
        self._set_new_frames('pc_no_async', frame)

        # analyze asynchronous breathing PC only
        frame = self.proc_results[(self.proc_results.ventmode == 'pc') & async_mask]
        self._set_new_frames('pc_only_async', frame)

        # analyze all PRVC only
        frame = self.proc_results[(self.proc_results.ventmode == 'prvc')]
        self._set_new_frames('prvc_only', frame)

        # analyze non-asynchronous breathing PRVC only
        frame = self.proc_results[(self.proc_results.ventmode == 'prvc') & ~async_mask]
        self._set_new_frames('prvc_no_async', frame)

        # analyze asynchronous breathing PRVC only
        frame = self.proc_results[(self.proc_results.ventmode == 'prvc') & async_mask]
        self._set_new_frames('prvc_only_async', frame)

        # analyze only pressure breathing
        frame = self.proc_results[self.proc_results.ventmode.isin(['pc', 'prvc'])]
        self._set_new_frames('all_pressure_only', frame)

        # analyze non-asynchronous pressure related breathing. PC/PRVC
        frame = self.proc_results[(self.proc_results.ventmode.isin(['pc', 'prvc'])) & ~async_mask]
        self._set_new_frames('all_pressure_no_async', frame)

        # analyze asynchronous pressure related breathing. PC/PRVC
        frame = self.proc_results[(self.proc_results.ventmode.isin(['pc', 'prvc'])) & async_mask]
        self._set_new_frames('all_pressure_only_async', frame)

        # analyze breaths with no apparent efforting.
        frame = self.proc_results[masks['no_efforting']]
        self._set_new_frames('no_efforting', frame)

        # analyze breaths for early efforting
        frame = self.proc_results[masks['early_efforting']]
        self._set_new_frames('early_efforting', frame)

        # analyze breaths for inspiratory efforting
        frame = self.proc_results[masks['insp_efforting']]
        self._set_new_frames('insp_efforting', frame)

        # analyze breaths for late efforting
        frame = self.proc_results[masks['exp_efforting']]
        self._set_new_frames('exp_efforting', frame)

        # analyze breaths for all efforting
        frame = self.proc_results[masks['all_efforting']]
        self._set_new_frames('all_efforting', frame)

        # save a processed results container because this method takes the
        # longest time out of all the other methods to run
        self.full_analysis_done = True
        pd.to_pickle(self, self.results_dir.joinpath('ResultsContainer_mins_{}.pkl'.format(self.n_minutes)))

    def calc_windows(self, df):
        """
        Calculates the windowed stats of an algorithm for
        a set window size. Calcs windowed median deviation (WMD) and sequential
        median deviation (SMD). Also performs a bit of preprocessing onto the dataset

        Note: WM = window median
              SM = sequential median
        """
        for algo in self.algos_used:
            self.proc_results['{}_diff'.format(algo)] = self.proc_results['gold_stnd_compliance'] - self.proc_results[algo]

        for patiend_id, pt_df in df.groupby('patient_id'):
            for algo in self.algos_used:
                # found a bug in rolling where if there are any nans within the
                # step size, then the rolling window automatically returns nan.
                # so instead, we have to perform a custom rolling script
                df.loc[pt_df.index, '{}_wm_{}'.format(algo, self.window_n)] = rolling_nan_median(pt_df[algo].values, self.window_n)
            df.loc[pt_df.index, 'dtw_wm_{}'.format(self.window_n)] = rolling_nan_median(pt_df['dtw'].values, self.window_n)

        for algo in self.algos_used:
            wm_colname = '{}_wm_{}'.format(algo, self.window_n)
            wmd_colname = '{}_wmd_{}'.format(algo, self.window_n)
            diff_colname = '{}_diff_{}'.format(algo, self.window_n)
            df[wmd_colname] = df.gold_stnd_compliance - df[wm_colname]
            # make sure algo calcs are null if not available for specific mode
            if algo in FileCalculations.algos_unavailable_for_vc and not self.no_algo_restrict:
                df.loc[df.ventmode == 'vc', [wm_colname, wmd_colname, diff_colname]] = np.nan
            elif algo in FileCalculations.algos_unavailable_for_pc_prvc and not self.no_algo_restrict:
                df.loc[df.ventmode != 'vc', [wm_colname, wmd_colname, diff_colname]] = np.nan

        for patiend_id, pt_df in df.groupby('patient_id'):
            for algo in self.algos_used:
                tmp = sequential_nan_median(pt_df[algo].values, self.window_n)
                df.loc[pt_df.index, '{}_sm_{}'.format(algo, self.window_n)] = tmp

        for algo in self.algos_used:
            sm_colname = '{}_sm_{}'.format(algo, self.window_n)
            smd_colname = '{}_smd_{}'.format(algo, self.window_n)
            df[smd_colname] = df.gold_stnd_compliance - df[sm_colname]
            if algo in FileCalculations.algos_unavailable_for_vc and not self.no_algo_restrict:
                df.loc[df.ventmode == 'vc', [sm_colname, smd_colname]] = np.nan
            elif algo in FileCalculations.algos_unavailable_for_pc_prvc and not self.no_algo_restrict:
                df.loc[df.ventmode != 'vc', [sm_colname, smd_colname]] = np.nan
        # clear any breaths where there is no supported ventmode
        df = df[~(df.ventmode == '')]

    def calc_async_index(self, df):
        """
        Perform asynchrony index calculations on a dataset. The following calcs
        will be done

        * asynci
        * asynci_no_fam
        * bsi
        * dti
        * dci
        * fai
        * fai_no_fam
        * insp_effi

        We do index instead of frequency because you'll be able to compare
        across window size changes that way.
        """
        # make sure that dta is properly formatted
        df.loc[df.dta == 2, 'dta'] = 1
        # small posthoc fix for insp efforting
        df.loc[df.insp_efforting.isna(), 'insp_efforting'] = 0
        # make changes so we can handle flow async different cases
        df['fa_mild'] = 0
        df['fa_mod'] = 0
        df['fa_sev'] = 0
        df.loc[df.fa == 1, 'fa_mild'] = 1
        df.loc[df.fa == 2, 'fa_mod'] = 1
        df.loc[df.fa == 3, 'fa_sev'] = 1

        index_to_async_mapping = [
            ('asynci', ['bsa', 'dta', 'fa_mild', 'fa_mod', 'fa_sev', 'static_dca', 'dyn_dca']),
            ('asynci_no_fam', ['bsa', 'dta', 'fa_mod', 'fa_sev', 'static_dca', 'dyn_dca']),
            ('bsi', ['bsa']),
            ('dci', ['static_dca', 'dyn_dca']),
            ('dti', ['dta']),
            ('fai', ['fa_mild', 'fa_mod', 'fa_sev']),
            ('fai_no_fam', ['fa_mod', 'fa_sev']),
            ('insp_effi', ['insp_efforting']),
        ]
        for patiend_id, pt_df in df.groupby('patient_id'):
            for index_col, async_cols in index_to_async_mapping:
                final_index_col = '{}_{}'.format(index_col, self.window_n)
                df.loc[pt_df.index, final_index_col] = rolling_nan_mean(pt_df[async_cols].any(axis=1).astype(int).values, self.window_n)

    def collate_data(self, algos_used, n_minutes=30):
        """
        Now that (presumably) all patient results have been tabulated we can finally determine
        what our gold standard compliances are for specific time points. Filter

        :param algos_used: algorithms used in the calculations
        :param n_minutes: n minutes to find a plateau pressure within range of breath start.
        """
        self.algos_used = algos_used
        self.algo_markers = {algo: self.scatter_marker_symbols[algo] for i, algo in enumerate(self.algos_used)}
        self.algo_colors = {algo: cc.cm.glasbey(i) for i, algo in enumerate(self.algos_used)}
        self.n_minutes = n_minutes
        proc_results = pd.concat(self.raw_results)
        proc_results.index = range(len(proc_results))
        for patient, frame in proc_results.groupby('patient_id'):
            valid_plats = frame[frame.is_valid_plat == True]
            # some rows will have multiple plats overlapping with them. we can average the plat
            # get a compliance, plateau pressure, and driving pressure
            for i, row in frame.iterrows():
                plats_in_range = valid_plats[
                    (valid_plats.abs_bs - pd.Timedelta(minutes=n_minutes) <= row.abs_bs) &
                    (row.abs_bs <= valid_plats.abs_bs + pd.Timedelta(minutes=n_minutes))
                ]
                proc_results.loc[i, 'gold_stnd_compliance'] = plats_in_range.gold_stnd_compliance.mean()
                proc_results.loc[i, 'p_plat'] = (proc_results.loc[i, 'tvi']/proc_results.loc[i, 'gold_stnd_compliance']) + proc_results.loc[i, 'peep']
                proc_results.loc[i, 'p_driving'] = proc_results.loc[i, 'p_plat'] - proc_results.loc[i, 'peep']

        self.proc_results = proc_results
        # make sure newest metadata is available for frame
        self.update_for_new_metadata(save=False)

        # filter out breaths with no ventmode
        self.proc_results.loc[(self.proc_results.ventmode.isna()) | (self.proc_results.ventmode == '')]
        # if algo restrict then make sure that breaths are properly filtered by ventmode
        if not self.no_algo_restrict:
            for algo in FileCalculations.algos_unavailable_for_vc:
                self.proc_results.loc[self.proc_results.ventmode == 'vc', algo] = np.nan
            for algo in FileCalculations.algos_unavailable_for_pc_prvc:
                self.proc_results.loc[self.proc_results.ventmode.isin(['pc', 'prvc']), algo] = np.nan

        # filter outliers by patient
        for algo in algos_used:
            for patient_id, df in self.proc_results.groupby('patient_id'):
                inf_idxs = df[(df[algo] == np.inf) | (df[algo] == -np.inf)].index
                df.loc[inf_idxs, algo] = np.nan
                self.proc_results.loc[inf_idxs, algo] = np.nan
                # I've found that mean can blow up in the presence of outliers.
                # So instead use the median
                algo_median = df[algo].median(skipna=True)
                algo_mad = median_abs_deviation(df[algo].values, nan_policy='omit')
                # remove anything with < 0  compliance
                self.proc_results.loc[df[df[algo] < 0].index, algo] = np.nan
                # multiply the algo mad by 30 because std can be so ridiculous that it
                # is literally non-physiological 30x removal will basically remove
                # everything within the range that is non-physiologic. If we do not
                # do this filtering then std deviation explodes and makes it difficult
                # to interpret our results.
                #
                # There is an argument to be made that this is important to keep
                # because it will show true algorithm behavior. However, my
                # disagreement of this is that when results become non-physiologic,
                # meaning compliance that is impossible, then those results should just
                # automatically be removed in practice. Then keeping them in science
                # will do no real good to inform the actual implementation science.
                self.proc_results.loc[df[df[algo].abs() >= (algo_median + 30*algo_mad)].index, algo] = np.nan
        self.full_analysis_done = False

    def compare_breath_level_async_types_by_ventmode_bar(self, mode, std_or_mad, algos=None, asynchrony_type=None, absolute=True, **kwargs):
        """
        Compare asynchrony types by ventmode on breath level.

        :param mode: ventilation mode "vc"/"pressure"
        :param std_or_mad: use standard deviation ('std') or MAD ('mad')
        :param algos: list of algos to sample
        :param asynchrony_type: analyze by specific asynchrony type options: "bsa", "dta", "fa_no_fam", "fa_mild", "fa_mod", "fa_sev", "dca"
        :param absolute: return absolute values for WMD calcs
        """
        if algos is None:
            algos = self.algos_used

        if absolute:
            abs_func = lambda x: x.abs()
            abs_label = '|'
        else:
            abs_func = lambda x: x
            abs_label = ''
        async_mask_name = self._validate_async_mask_name_by_mode(mode, asynchrony_type)
        async_data_mask = self.get_masks()[async_mask_name]
        async_data = self.proc_results.loc[async_data_mask]

        norm_data_mask = self.get_masks()['{}_no_async'.format('vc' if mode == 'vc' else 'pc_prvc')]
        norm_data = self.proc_results.loc[norm_data_mask]

        async_mask_name = self._get_async_mask_name_for_compare_breath_level_masks(asynchrony_type)
        if std_or_mad == 'std':
            dev_colname = 'Standard Deviation'
            dev_estim = np.nanstd
        elif std_or_mad == 'mad':
            dev_colname = 'MAD'
            dev_estim = lambda x: median_abs_deviation(x, nan_policy='omit')

        rows = []
        for algo in algos:
            algo_name = FileCalculations.shorthand_name_mapping[algo]
            algo_col = '{}_diff'.format(algo)
            tmp_df_async = abs_func(async_data[[algo_col]]).rename(columns={algo_col: 'Median Absolute Difference'})
            tmp_df_async['Algorithm'] = algo_name
            tmp_df_async['Breath Type'] = async_mask_name
            tmp_df_norm = abs_func(norm_data[[algo_col]]).rename(columns={algo_col: 'Median Absolute Difference'})
            tmp_df_norm['Algorithm'] = algo_name
            tmp_df_norm['Breath Type'] = 'Normal'
            rows.extend([tmp_df_norm, tmp_df_async])
        df = pd.concat(rows)
        df = df.dropna()

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3*8, 3*kwargs.get('figsize_height_mult', 5)))
        sns.barplot(x='Algorithm', y='Median Absolute Difference', hue='Breath Type', data=df, ax=axes[0], estimator=np.nanmedian, palette=kwargs.get('ax0_palette', 'Set1'), linewidth=kwargs.get('bar_lw', 0.5), ci=None, edgecolor=kwargs.get('bar_ec', 'black'))
        sns.barplot(x='Algorithm', y='Median Absolute Difference', hue='Breath Type', data=df, ax=axes[1], estimator=dev_estim, palette=kwargs.get('ax1_palette', 'Set2'), linewidth=kwargs.get('bar_lw', 1), ci=None, edgecolor=kwargs.get('bar_ec', 'black'))
        axes[0].xaxis.set_major_locator(plt.NullLocator())
        axes[0].set_ylabel('Median $'+abs_label+'C_{rs}^k-\hat{C}_{rs}^k'+abs_label+'$', fontsize=kwargs.get('label_fontsize', 16), labelpad=kwargs.get('ylabel_pad', 4.0))
        axes[0].set_ylim(kwargs.get('ax0_ylim', axes[0].get_ylim()))
        #axes[1].get_legend().remove()
        axes[1].set_ylabel(dev_colname, fontsize=kwargs.get('label_fontsize', 16))
        axes[1].set_ylim(kwargs.get('ax1_ylim', axes[1].get_ylim()))
        for ax in axes:
            xtick_names = plt.setp(ax, xticklabels=[algo._text for algo in ax.get_xticklabels()])
            plt.setp(xtick_names, rotation=kwargs.get('rotation', 60), fontsize=kwargs.get('tick_fontsize', 18))
            plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('tick_fontsize', 14))
            ax.grid(True, lw=kwargs.get('grid_lw', 1), alpha=kwargs.get('grid_alpha', None), axis='y')
            ax.set_xlabel(None)
            ax.legend(fontsize=kwargs.get('legend_fontsize', 18), loc=kwargs.get('legend_loc', 'best'), title=kwargs.get('legend_title', 'Breath Type'), title_fontsize=kwargs.get('legend_fontsize', 18))

        figname = str(self.results_dir.joinpath('compare-breath-level-async-v-no-async-bar-{}-{}-abs-{}-mins-{}.png'.format(mode, asynchrony_type, absolute, self.n_minutes)).resolve())
        plt.tight_layout()
        plt.savefig(figname, dpi=self.dpi)

    def compare_patient_level_masks_bar(self, mask1_name, mask2_name, windowing, std_or_mad, label_mask1=None, label_mask2=None, algos=None):
        """
        Compare results of different masks to each other on patient by patient basis.
        Plot results out with bar chart.

        :param mask1_name: mask name based on masks obtained from `get_masks`
        :param mask2_name: mask name based on masks obtained from `get_masks`
        :param windowing: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        :param std_or_mad: use standard deviation ('std') or MAD ('mad')
        :param label_mask1: custom label in legend for mask1
        :param label_mask2: custom label in legend for mask2
        :param algos: specific algos to analyze. If not set, defaults to all.
        """
        if (label_mask1 and not label_mask2) or (label_mask2 and not label_mask1):
            raise Exception('if you have a custom label for one mask, you must provide a custom label for the other')

        masks = self.get_masks()
        mask1 = masks[mask1_name]
        mask2 = masks[mask2_name]

        pp1 = self.analyze_per_patient_df(self.proc_results[mask1])
        pp2 = self.analyze_per_patient_df(self.proc_results[mask2])
        if algos is not None:
            pp1 = pp1[pp1.algo.isin(algos)]
            pp2 = pp2[pp2.algo.isin(algos)]

        ad_std1 = self.preprocess_ad_std_in_df(pp1, windowing, std_or_mad)
        ad_std2 = self.preprocess_ad_std_in_df(pp2, windowing, std_or_mad)

        ad_df, std_df = self._create_per_patient_comparison_frames(ad_std1, ad_std2, 'Breath Types', mask1_name, mask2_name, std_or_mad)

        if std_or_mad == 'std':
            ylabel = 'Standard Deviation'
        elif std_or_mad == 'mad':
            ylabel = 'Median Absolute Deviation'

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3*8, 3*3))
        sns.barplot(x='Algorithm', y='Absolute Difference', hue='Breath Types', data=ad_df, ax=axes[0])
        sns.barplot(x='Algorithm', y=ylabel, hue='Breath Types', data=std_df, ax=axes[1])

        for ax in axes:
            xtick_names = plt.setp(ax, xticklabels=[algo._text for algo in ax.get_xticklabels()])
            plt.setp(xtick_names, rotation=90)
            if label_mask1:
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [label_mask1, label_mask2]
                ax.legend(handles, new_labels, fontsize=20, framealpha=0.4)
        figname = str(self.results_dir.joinpath('compare-patient-level-masks-bar-{}-{}-windowing-{}-mins-{}.png'.format(mask1_name, mask2_name, windowing, self.n_minutes)).resolve())
        plt.tight_layout()
        plt.savefig(figname, dpi=self.dpi)

    def compare_patient_level_masks_scatter(self, mask1_name, mask2_name, windowing, std_or_mad, individual_patients=False, std_lim=None):
        """
        Compare results of different masks to each other on patient by patient basis.
        Plot results out with scatter plots as usual.

        :param mask1_name: mask name based on masks obtained from `get_masks`
        :param mask2_name: mask name based on masks obtained from `get_masks`
        :param windowing: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        :param std_or_mad: use standard deviation ('std') or MAD ('mad')
        :param individual_patients: show individual_patients scatter points
        :param std_lim: limit graphs by standard deviation within certain
                        factor. Normally is set to None (no limit). But can be
                        set to any floating value > 0.
        """
        masks = self.get_masks()
        mask1 = masks[mask1_name]
        mask2 = masks[mask2_name]

        pp1 = self.analyze_per_patient_df(self.proc_results[mask1])
        pp2 = self.analyze_per_patient_df(self.proc_results[mask2])

        ad_std1 = self.preprocess_ad_std_in_df(pp1, windowing, std_or_mad)
        ad_std2 = self.preprocess_ad_std_in_df(pp2, windowing, std_or_mad)

        ad_std = copy(ad_std1)
        for algo in pp1.algo.unique():
            for i in range(4):
                ad_std[algo][i] = ad_std2[algo][i] - ad_std1[algo][i]

        plt_title = '{} vs {}'.format(mask1_name, mask2_name)
        figname = '{}_vs_{}-scatter-windowing-{}-mins-{}.png'.format(mask1_name, mask2_name, windowing, self.n_minutes)
        xlabel = '({} larger)\u21C7\u21C7   |   \u21C9\u21C9({} larger)\n\nAbsolute Difference of (Compliance - Algo)'.format(mask1_name, mask2_name)
        self._ad_std_scatter(ad_std, windowing, plt_title, figname, individual_patients, std_lim, std_or_mad, custom_xlabel=xlabel)

    def compare_breath_level_masks(self, mask1_name, mask2_name, windowing, algos=None, absolute=False, **kwargs):
        """
        Compare results of different masks to each other on breath by breath results.

        :param mask1_name: mask name based on masks obtained from `get_masks`
        :param mask2_name: mask name based on masks obtained from `get_masks`
        :param windowing: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        :param algos: list of algos to display
        """
        figname = 'compare-breath-masks-{}-{}-windowing-{}-mins-{}.png'.format(mask1_name, mask2_name, windowing, self.n_minutes)
        masks = self.get_masks()
        mask1 = masks[mask1_name]
        mask2 = masks[mask2_name]
        self._compare_breath_level_masks(
            self.proc_results[mask1],
            self.proc_results[mask2],
            windowing,
            windowing,
            algos,
            mask1_name.replace('_', ' '),
            mask2_name.replace('_', ' '),
            figname,
            absolute,
            **kwargs,
        )

    def compare_breath_level_async_v_no_async_monte_carlo(self, n_resamples, mode, algos=None, asynchrony_type=None, absolute=False, **kwargs):
        """
        Compare breath-level algo performance for asynchronies v no asynchronies by
        resampling asynchronous and non-asynchronous breathing n number times. Performs
        operation by mode ('vc'/'pressure').

        Not actually statistically meaningful, however it can give us a good idea of how our
        algos are performing and where to focus on future efforts

        :param n_resamples: number of times to resample non-async/async data. If None then do not resample
        :param mode: ventilation mode "vc"/"pressure"
        :param algos: list of algos to sample
        :param asynchrony_type: analyze by specific asynchrony type options: "bsa", "dta", "fa_no_fam", "fa_mild", "fa_mod", "fa_sev", "dca", "async_no_fam"
        """
        async_mask_name = self._validate_async_mask_name_by_mode(mode, asynchrony_type)
        async_data_mask = self.get_masks()[async_mask_name]
        async_data = self.proc_results.loc[async_data_mask]
        if n_resamples is not None:
            async_idxs = np.random.choice(async_data.index, size=n_resamples)
            async_data = async_data.loc[async_idxs]

        norm_data_mask = {
            'vc': self.get_masks()['vc_no_async'],
            'pressure': self.get_masks()['pc_prvc_no_async'],
        }[mode]
        norm_data = self.proc_results.loc[norm_data_mask]
        if n_resamples is not None:
            norm_data_idxs = np.random.choice(norm_data.index, size=n_resamples)
            norm_data = norm_data.loc[norm_data_idxs]

        async_mask_name = self._get_async_mask_name_for_compare_breath_level_masks(asynchrony_type)
        self._compare_breath_level_masks(
            norm_data,
            async_data,
            None,
            None,
            algos,
            'Normal',
            async_mask_name,
            'breath_by_breath_async_v_no_async_monte_carlo_{}_mode_{}_async_type_{}_resamps.png'.format(n_resamples, mode, asynchrony_type),
            absolute,
            **kwargs,
        )

    def compare_breath_level_on_ventmode_monte_carlo(self, n_resamples, algos=None, absolute=False, **kwargs):
        """
        Compare algos based on mode. Compares resampled results from mode agnostic
        algos if there are algo restrictions. If no algo restrictions, then
        compare all possible algos.

        Not actually statistically meaningful, however it can give us a good idea of how our
        algos are performing and where to focus on future efforts

        :param n_resamples: number of times to resample non-async/async data. If None then do not resample
        :param algos: list of algos to sample
        """
        if not self.no_algo_restrict and algos is None:
            algos = set(self.algos_used).difference(FileCalculations.algos_unavailable_for_pc_prvc).difference(FileCalculations.algos_unavailable_for_vc)

        vc_data = self.proc_results.loc[self.get_masks()['vc_only']]
        if n_resamples is not None:
            vc_idxs = np.random.choice(vc_data.index, size=n_resamples)
            vc_data = vc_data.loc[vc_idxs]

        pressure_data = self.proc_results.loc[self.get_masks()['pc_prvc']]
        if n_resamples is not None:
            pressure_idxs = np.random.choice(pressure_data.index, size=n_resamples)
            pressure_data = pressure_data.loc[pressure_idxs]

        self._compare_breath_level_masks(
            vc_data,
            pressure_data,
            None,
            None,
            algos,
            'Volume Control',
            'Pressure Control',
            'breath_by_breath_compare_algos_by_mode_monte_carlo_{}_resamps.png'.format(n_resamples),
            absolute,
            **kwargs,
        )

    def compare_plat_minutes_bar_per_breath(self, windowing, win_size=20, n_minutes=[5, 10, 15, 30], absolute=True):
        """
        Compare different plateau minute ranges in a bar chart.

        :param windowing: wmd OR smd
        :param win_size: window size. some number > 0
        :param n_minutes: list of different minute ranges to compare to. individual numbers should be
                          between 1 and 30.
        :param absolute: return absolute values for WMD calcs
        """
        if windowing not in ['wmd', 'smd']:
            raise ValueError('Must specify a valid window strategy. choices: wmd OR smd')
        min_containers = [(mins, self.reset_plat_minutes(mins)) for mins in n_minutes]

        super_frame = None
        fig, ax = plt.subplots(figsize=(3*8, 4*3))
        for mins, cont in min_containers:
            cols = ['{}_{}_{}'.format(algo, windowing, win_size) for algo in self.algos_used]
            cont.set_new_window_n(win_size)
            if absolute:
                frame = cont.proc_results[cols].abs()
            else:
                frame = cont.proc_results[cols]

            rename_cols = {col: col.replace('_{}_{}'.format(windowing, win_size), '')  for col in cols}
            frame = frame.rename(columns=rename_cols)
            frame = frame.melt()
            frame['Minutes'] = mins
            if super_frame is None:
                super_frame = frame
            else:
                super_frame = super_frame.append(frame)

        sns.barplot(x='variable', y='value', data=super_frame, hue='Minutes')
        plt.ylabel('Median{}Difference ({})'.format(' Absolute ' if absolute else '', '$|C_{rs}^k-\hat{C}_{rs}^k|$' if absolute else ''))
        plt.xlabel('Algorithm')
        xtick_names = plt.setp(ax, xticklabels=[FileCalculations.shorthand_name_mapping[algo] for algo in sorted(self.algos_used)])
        plt.setp(xtick_names, rotation=90)
        plt.legend(loc='upper right', framealpha=.7)
        plt.tight_layout()
        fig.savefig(self.results_dir.joinpath('minute_analysis_bar-windowing-{}_winsize-{}.png'.format(windowing, win_size)).resolve(), dpi=self.dpi)
        plt.show(fig)

    def compare_window_strategies_bar_per_breath(self, windowing1, windowing2):
        """
        Compare results of different window types to each other on patient breath by breath basis.
        Plot results out with bar charts.

        :param windowing1: window type to use first ('wmd', 'smd', or None)
        :param windowing2: window type to use second ('wmd', 'smd', or None)
        """
        sorted_diff_cols1 = self._get_windowing_algo_diff_colnames(windowing1)
        sorted_diff_cols2 = self._get_windowing_algo_diff_colnames(windowing2)
        bb1 = self.proc_results[sorted_diff_cols1].abs()
        bb2 = self.proc_results[sorted_diff_cols2].abs()
        window_names = {None: 'No windowing', 'wmd': 'WMD', 'smd': 'SMD'}
        win1 = window_names[windowing1]
        win2 = window_names[windowing2]
        # rename cols
        bb1 = bb1.rename(columns={col: re.sub('(_diff|_(wmd|smd)_\d+)', '', col) for i, col in enumerate(bb1.columns)}).melt()
        bb2 = bb2.rename(columns={col: re.sub('(_diff|_(wmd|smd)_\d+)', '', col) for i, col in enumerate(bb2.columns)}).melt()
        bb1['Windowing'] = win1
        bb2['Windowing'] = win2
        df = pd.concat([bb1, bb2])
        df = df.rename(columns={'variable': 'algo'})
        fig, ax = plt.subplots(figsize=(3*8, 3*3))

        # alphabetical order again
        algos_in_order = sorted(self.algos_used)
        sns.barplot(x='algo', y='value', data=df, hue='Windowing', ax=ax, n_boot=self.boot_resamples, palette='Set2')
        xtick_names = plt.setp(ax, xticklabels=[FileCalculations.shorthand_name_mapping[i] for i in algos_in_order])
        plt.setp(xtick_names, rotation=90, fontsize=14)
        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], ls='--', zorder=0, c='red')
        ax.set_ylabel('Difference between Compliance and Algo', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.legend(fontsize=16)
        title = '{} vs {}'.format(win1, win2)
        ax.set_title(title, fontsize=20)

        # XXX also for this function we're using mean. so median and IQR is a bit misleading.
        # This is true elsewhere for some of the barplots as well. Need to remove this calc
        # from here and just replace with mean and confidence.
        proc_frame = self.extract_medians_and_iqr(self.proc_results, windowing1, absolute=True)
        self._show_breath_by_breath_algo_table(proc_frame, win1)

        proc_frame = self.extract_medians_and_iqr(self.proc_results, windowing2, absolute=True)
        self._show_breath_by_breath_algo_table(proc_frame, win2)
        figname = str(self.results_dir.joinpath('compare-window-strats-bar-per-breath-{}-{}-mins-{}.png'.format(win1, win2, self.n_minutes)).resolve())
        plt.tight_layout()
        plt.savefig(figname, dpi=self.dpi)
        plt.show(fig)

    def compare_window_strategies_box_per_breath(self, windowing1, windowing2, mask_name=None, algos=None, absolute=False, **kwargs):
        """
        Compare per-breath window strategies via boxplot. Optionally will accept a mask for more
        fine-grained analysis.

        :param windowing1: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        :param windowing2: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        :param mask_name: mask name based on masks obtained from `get_masks`
        :param algos: list of algos to display
        :param absolute: (bool) use absolute value for results or no.
        """
        if mask_name is not None:
            mask = self.get_masks()[mask_name]
        else:
            mask = self.get_masks()['all']

        window_name1 = {None: 'No windowing', 'wmd': 'Rolling Median', 'smd': 'Sequential Median'}[windowing1]
        window_name2 = {None: 'No windowing', 'wmd': 'Rolling Median', 'smd': 'Sequential Median'}[windowing2]

        self._compare_breath_level_masks(
            self.proc_results[mask],
            self.proc_results[mask],
            windowing1,
            windowing2,
            algos,
            window_name1,
            window_name2,
            'breath_by_breath_compare_window_strats_{}_{}_{}_resamps.png'.format(windowing1, windowing2, mask_name),
            absolute,
            **kwargs,
        )

    def compare_window_strategies_bar_per_patient(self, windowing1, windowing2, std_or_mad, label_mask1=None, label_mask2=None, **kwargs):
        """
        Compare results of different window types to each other on patient by patient basis.
        Plot results out with bar charts. Confidence intervals here tend to be quite
        wide because our current n is 18. Future work can improve upon this number.

        :param windowing1: window type to use first ('wmd', 'smd', or None)
        :param windowing2: window type to use second ('wmd', 'smd', or None)
        :param std_or_mad: use standard deviation ('std') or MAD ('mad')
        :param label_mask1: custom label in legend for mask1
        :param label_mask2: custom label in legend for mask2
        """
        if (label_mask1 and not label_mask2) or (label_mask2 and not label_mask1):
            raise Exception('if you have a custom label for one mask, you must provide a custom label for the other')
        pp = self.analyze_per_patient_df(self.proc_results)

        ad_std1 = self.preprocess_ad_std_in_df(pp, windowing1, std_or_mad)
        ad_std2 = self.preprocess_ad_std_in_df(pp, windowing2, std_or_mad)

        window_names = {None: 'No windowing', 'wmd': 'WMD', 'smd': 'SMD'}
        win1 = window_names[windowing1]
        win2 = window_names[windowing2]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3*8, 3*3))
        # create frame for absolute diff
        ad_df, std_df = self._create_per_patient_comparison_frames(ad_std1, ad_std2, 'Window Strategy', win1, win2, std_or_mad)

        if std_or_mad == 'std':
            dev = 'Standard Deviation'
        elif std_or_mad == 'mad':
            dev = 'Median Absolute Deviation'

        sns.barplot(x='Algorithm', y='Absolute Difference', hue='Window Strategy', data=ad_df, ax=axes[0], palette=kwargs.get('ax0_palette', 'Set2'), n_boot=self.boot_resamples, capsize=kwargs.get('capsize', None), ci=kwargs.get('ci', 95), linewidth=kwargs.get('bar_lw', None), edgecolor=kwargs.get('bar_ec', 'black'))
        sns.barplot(x='Algorithm', y=dev, hue='Window Strategy', data=std_df, ax=axes[1], palette=kwargs.get('ax1_palette', 'Set3'), n_boot=self.boot_resamples, capsize=kwargs.get('capsize', None), ci=kwargs.get('ci', 95), linewidth=kwargs.get('bar_lw', None), edgecolor=kwargs.get('bar_ec', 'black'))

        for ax in axes:
            xtick_names = plt.setp(ax, xticklabels=[algo._text for algo in ax.get_xticklabels()])
            plt.setp(xtick_names, rotation=kwargs.get('rotation', 70))
            if label_mask1:
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [label_mask1, label_mask2]
                ax.legend(handles, new_labels, fontsize=kwargs.get('legend_fontsize', 18), loc=kwargs.get('legend_loc', 'best'), title=kwargs.get('legend_title', 'Window Type'), title_fontsize=kwargs.get('legend_fontsize', 18))
                ax.set_xlabel('')
                ax.grid(True, lw=kwargs.get('grid_lw', 1), alpha=kwargs.get('grid_alpha', None), axis='y')
                plt.setp(xtick_names, rotation=kwargs.get('rotation', 60), fontsize=kwargs.get('tick_fontsize', 14))
                plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('tick_fontsize', 14))

        axes[0].set_ylabel('Median $|C_{rs}^k-\hat{C}_{rs}^k|$', fontsize=kwargs.get('label_fontsize', 18))
        axes[1].set_ylabel(dev, fontsize=kwargs.get('label_fontsize', 18))

        figname = str(self.results_dir.joinpath('compare-window-strats-bar-{}-{}-{}-mins-{}.png'.format(win1, win2, std_or_mad, self.n_minutes)).resolve())
        plt.tight_layout()
        plt.savefig(figname, dpi=self.dpi)

    def compare_window_lengths_bar_per_patient(self, windowing, std_or_mad, windows=[5, 10, 20, 50, 100, 200], **kwargs):
        """
        Compare results of different window types to each other on patient by patient basis.
        Plot results out with bar charts. Confidence intervals here tend to be quite
        wide because our current n is 18. Future work can improve upon this number.

        :param windowing: window type to use ('wmd', 'smd')
        :param std_or_mad: use standard deviation ('std') or MAD ('mad')
        :param windows: window lengths to compare
        """
        def show_ad_std_plot(data, ycolname, legend, **kwargs):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(kwargs.get('fig_width', 3*8), kwargs.get('fig_height', 3*3)))
            sns.barplot(x='Algorithm', y=ycolname, data=data, ax=ax, hue='Window Size', n_boot=self.boot_resamples)
            plt.setp(ax.get_xticklabels(), rotation=kwargs.get('rotation', 60), fontsize=kwargs.get('tick_fontsize', 14))
            plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('tick_fontsize', 14))
            if ycolname == 'Absolute Difference':

                ax.set_ylabel('Median $|C_{rs}^k-\hat{C}_{rs}^k|$', fontsize=kwargs.get('label_fontsize', 18), labelpad=kwargs.get('ylabel_pad', 4.0))
            else:
                ax.set_ylabel(ylabel, fontsize=kwargs.get('label_fontsize', 18), labelpad=kwargs.get('ylabel_pad', 4.0))
            ax.set_xlabel('')
            if not legend:
                ax.get_legend().remove()
            else:
                ax.legend(loc=kwargs.get('legend_loc', 'upper left'), title='Window Size', framealpha=kwargs.get('legend_alpha', 0.4), fontsize=kwargs.get('legend_fontsize', 22), title_fontsize=kwargs.get('legend_fontsize', 22))

            ax.grid(True, lw=kwargs.get('grid_lw', 1), alpha=kwargs.get('grid_alpha', None), axis='y')
            plt.tight_layout()
            fig.savefig(self.results_dir.joinpath('compare_per_patient_win_lens_{}_mins_{}.png'.format(ycolname.replace(' ', '_'), self.n_minutes)).resolve(), dpi=self.dpi)
            plt.show(fig)
            plt.close()

        sns.set_style('whitegrid')

        pp = self.analyze_per_patient_df(self.proc_results)
        ad_stds = {'None': self.preprocess_ad_std_in_df(pp, None, std_or_mad)}
        for win_size in windows:
            self.set_new_window_n(win_size)
            pp = self.analyze_per_patient_df(self.proc_results)
            ad_stds[win_size] = self.preprocess_ad_std_in_df(pp, windowing, std_or_mad)

        window_names = {'wmd': 'WMD', 'smd': 'SMD'}
        win = window_names[windowing]

        ad_rows = []
        std_rows = []
        for size in ad_stds.keys():
            ad_std = ad_stds[size]
            for algo, items in ad_std.items():
                algo_name = FileCalculations.shorthand_name_mapping[algo]
                for val in items[0]:
                    ad_rows.append([algo_name, val, size])

            # create frame for std.
            for algo, items in ad_std.items():
                algo_name = FileCalculations.shorthand_name_mapping[algo]
                for val in items[1]:
                    std_rows.append([algo_name, val, size])

        if std_or_mad == 'std':
            ylabel = 'Standard Deviation'
        elif std_or_mad == 'mad':
            ylabel = 'MAD'
        ad_df = pd.DataFrame(ad_rows, columns=['Algorithm', 'Absolute Difference', 'Window Size'])
        std_df = pd.DataFrame(std_rows, columns=['Algorithm', ylabel, 'Window Size'])

        show_ad_std_plot(ad_df, 'Absolute Difference', False, **kwargs)
        show_ad_std_plot(std_df, ylabel, True, **kwargs)

    def compare_window_strategies_scatter_per_patient(self, windowing1, windowing2, std_or_mad, individual_patients=False, std_lim=None):
        """
        Compare results of different window types to each other on patient by patient basis.
        Plot results out with scatter plots as usual.

        :param windowing1: window type to use first ('wmd', 'smd', or None)
        :param windowing2: window type to use second ('wmd', 'smd', or None)
        :param std_or_mad: use standard deviation ('std') or MAD ('mad')
        :param individual_patients: show individual_patients scatter points
        :param std_lim: limit graphs by standard deviation within certain
                        factor. Normally is set to None (no limit). But can be
                        set to any floating value > 0.
        """
        pp = self.analyze_per_patient_df(self.proc_results)

        ad_std1 = self.preprocess_ad_std_in_df(pp, windowing1, std_or_mad)
        ad_std2 = self.preprocess_ad_std_in_df(pp, windowing2, std_or_mad)

        ad_std = copy(ad_std1)
        for algo in ad_std1.keys():
            for i in range(4):
                ad_std[algo][i] = ad_std2[algo][i] - ad_std1[algo][i]

        name_mapping = {'smd': 'SMD', 'wmd': 'WMD', None: 'No Windowing'}
        plt_title = '{} vs {}'.format(name_mapping[windowing1], name_mapping[windowing2])
        figname = '{}_vs_{}-scatter-mins-{}.png'.format(windowing1, windowing2, self.n_minutes)
        xlabel = '({} larger)\u21C7\u21C7   |   \u21C9\u21C9({} larger)\n\nAbsolute Difference of (Compliance - Algo)'.format(name_mapping[windowing1], name_mapping[windowing2])
        self._ad_std_scatter(ad_std, 'window_compr', plt_title, figname, individual_patients, std_lim, std_or_mad, custom_xlabel=xlabel)

    def extract_medians_and_iqr(self, df, windowing, absolute=False):
        """
        Extract median/IQR/std vals from a DataFrame for algorithm diffs. Is used in breath
        by breath plotting and table outputs

        :param df: should be a breath by breath DataFrame. Can have a mask applied if needed
        :param windowing: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        """
        if windowing in ['smd', 'wmd']:
            diff_colname_suffix = '_{}_{}'.format(windowing, self.window_n)
        else:
            diff_colname_suffix = '_diff'
        algos_in_frame = set(df.columns).intersection(self.algos_used)
        sorted_diff_cols = sorted([algo+diff_colname_suffix for algo in algos_in_frame])
        if not absolute:
            frame = df[sorted_diff_cols]
        else:
            frame = df[sorted_diff_cols].abs()
        rows = []
        for col in frame.columns:
            rows.append([col.replace(diff_colname_suffix, '')] + list(self._bootstrap(frame[col].values)))
        proc = pd.DataFrame(rows, columns=['algo', 'medians', 'iqr_low', 'iqr_high'])
        stds = frame.std(skipna=True, ddof=0).values
        proc['stds'] = stds
        return proc

    def extract_descriptive_statistics(self):
        """
        Gather descriptive statistics for dataset using the processed results dataframe
        """
        # find if data is cvc or not
        is_cvc = False
        for patient in self.proc_results.patient.unique():
            if 'cvc' in patient:
                is_cvc = True
                break
        pt_or_exp = 'patient' if not is_cvc else 'experiment'

        data = [
            # general patient stats
            'n {}s'.format(pt_or_exp),
            # ventmode n
            'n breaths',
            # asynchronous breath counts
            'async breaths',
            # deeper dive into asynchronies
            'dta breaths',
            'total bsa breaths',
            'total fa breaths',
            'total fa mild breaths',
            'total fa moderate breaths',
            'total fa severe breaths',
            'total static/dynamic dca breaths',
            'total static dca breaths',
            'total dynamic dca breaths',
            'rass -1',
            'rass -2',
            'rass -3',
            'rass -4',
            'rass -5',
            'rass n/a',
        ]
        n_patients = len(self.proc_results.patient_id.unique())
        masks = self.get_masks()
        async_mask = masks['async']
        vc_df = self.proc_results[self.proc_results.ventmode=='vc']
        pc_df = self.proc_results[self.proc_results.ventmode=='pc']
        prvc_df = self.proc_results[self.proc_results.ventmode=='prvc']
        vc_pts = len(vc_df.patient_id.unique())
        pc_pts = len(pc_df.patient_id.unique())
        prvc_pts = len(prvc_df.patient_id.unique())
        n_vc_async = len(self.proc_results[(self.proc_results.ventmode=='vc') & async_mask])
        n_pc_async = len(self.proc_results[(self.proc_results.ventmode=='pc') & async_mask])
        n_prvc_async = len(self.proc_results[(self.proc_results.ventmode=='prvc') & async_mask])
        total_async = len(self.proc_results[async_mask])
        vals = [
            [vc_pts, pc_pts, prvc_pts],
            # XXX show percentage counts as well so you can include it into the paper.
            # general breath counts
            [
                len(self.proc_results[self.proc_results.ventmode=='vc']),
                len(self.proc_results[self.proc_results.ventmode=='pc']),
                len(self.proc_results[self.proc_results.ventmode=='prvc'])
            ],
            # async breath counts
            [
                n_vc_async,
                n_pc_async,
                n_prvc_async,
            ],
            # deeper dive into asynchronies
            [
                len(vc_df[vc_df.dta > 0]),
                len(pc_df[pc_df.dta > 0]),
                len(prvc_df[prvc_df.dta > 0]),
            ],
            [
                len(vc_df[vc_df.bsa > 0]),
                len(pc_df[pc_df.bsa > 0]),
                len(prvc_df[prvc_df.bsa > 0]),
            ],
            [
                len(vc_df[vc_df.fa > 0]),
                len(pc_df[pc_df.fa > 0]),
                len(prvc_df[prvc_df.fa > 0]),
            ],
            [
                len(vc_df[vc_df.fa == 1]),
                len(pc_df[pc_df.fa == 1]),
                len(prvc_df[prvc_df.fa == 1]),
            ],
            [
                len(vc_df[vc_df.fa == 2]),
                len(pc_df[pc_df.fa == 2]),
                len(prvc_df[prvc_df.fa == 2]),
            ],
            [
                len(vc_df[vc_df.fa == 3]),
                len(pc_df[pc_df.fa == 3]),
                len(prvc_df[prvc_df.fa == 3]),
            ],
            [
                len(vc_df[(vc_df.dyn_dca > 0) | (vc_df.static_dca > 0)]),
                len(pc_df[(pc_df.dyn_dca > 0) | (pc_df.static_dca > 0)]),
                len(prvc_df[(prvc_df.dyn_dca > 0) | (prvc_df.static_dca > 0)]),
            ],
            [
                len(vc_df[(vc_df.static_dca > 0)]),
                len(pc_df[(pc_df.static_dca > 0)]),
                len(prvc_df[(prvc_df.static_dca > 0)]),
            ],
            [
                len(vc_df[(vc_df.dyn_dca > 0)]),
                len(pc_df[(pc_df.dyn_dca > 0)]),
                len(prvc_df[(prvc_df.dyn_dca > 0)]),
            ],
            [
                len(vc_df[(vc_df.rass == '-1')]),
                len(pc_df[(pc_df.rass == '-1')]),
                len(prvc_df[(prvc_df.rass == '-1')]),
            ],
            [
                len(vc_df[(vc_df.rass == '-2')]),
                len(pc_df[(pc_df.rass == '-2')]),
                len(prvc_df[(prvc_df.rass == '-2')]),
            ],
            [
                len(vc_df[(vc_df.rass == '-3')]),
                len(pc_df[(pc_df.rass == '-3')]),
                len(prvc_df[(prvc_df.rass == '-3')]),
            ],
            [
                len(vc_df[(vc_df.rass == '-4')]),
                len(pc_df[(pc_df.rass == '-4')]),
                len(prvc_df[(prvc_df.rass == '-4')]),
            ],
            [
                len(vc_df[(vc_df.rass == '-5')]),
                len(pc_df[(pc_df.rass == '-5')]),
                len(prvc_df[(prvc_df.rass == '-5')]),
            ],
            [
                len(vc_df[(vc_df.rass == 'other')]),
                len(pc_df[(pc_df.rass == 'other')]),
                len(prvc_df[(prvc_df.rass == 'other')]),
            ],
        ]
        table = PrettyTable()
        table.field_names = ['', 'vc', 'pc', 'prvc']
        for stat, val in zip(data, vals):
            table.add_row([stat]+val)
        prefix = 'CVC' if is_cvc else 'Patient'
        display(HTML('<h2>{}</h2>'.format(prefix + " Data Descriptive Statistics")))
        display(HTML(table.get_html_string()))

    def get_masks(self):
        return {
            'all': [True] * len(self.proc_results),
            'all_efforting': (
                (self.proc_results.early_efforting != 0) |
                (self.proc_results.insp_efforting != 0) |
                (self.proc_results.exp_efforting != 0)
            ),
            'async': (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa != 0) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            ),
            # FA mild can look very close to normal breathing
            'async_no_fam': (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa > 1) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            ),
            'async_and_efforting': (
                (self.proc_results.early_efforting != 0) |
                (self.proc_results.insp_efforting != 0) |
                (self.proc_results.exp_efforting != 0) |
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa != 0) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            ),
            # FA mild can look very close to normal breathing
            'async_no_fam_and_efforting': (
                (self.proc_results.early_efforting != 0) |
                (self.proc_results.insp_efforting != 0) |
                (self.proc_results.exp_efforting != 0) |
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                # FA mild can look very close to normal breathing
                (self.proc_results.fa > 1) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            ),
            'bsa': (self.proc_results.bsa != 0),
            'dca': (
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            ),
            'dta': (self.proc_results.dta != 0),
            'early_efforting': (self.proc_results.early_efforting != 0),
            'exp_efforting': (self.proc_results.exp_efforting != 0),
            'fa': (self.proc_results.fa != 0),
            'fa_mod_sev': (self.proc_results.fa > 1),
            'fa_sev': (self.proc_results.fa > 2),
            'insp_efforting': (self.proc_results.insp_efforting != 0),
            'no_async': (
                (self.proc_results.dta == 0) &
                (self.proc_results.bsa == 0) &
                (self.proc_results.fa == 0) &
                (self.proc_results.static_dca == 0) &
                (self.proc_results.dyn_dca == 0) &
                (self.proc_results.artifact == 0)
            ),
            'no_async_no_efforting': (
                (self.proc_results.dta == 0) &
                (self.proc_results.bsa == 0) &
                (self.proc_results.fa == 0) &
                (self.proc_results.static_dca == 0) &
                (self.proc_results.dyn_dca == 0) &
                (self.proc_results.early_efforting == 0) &
                (self.proc_results.insp_efforting == 0) &
                (self.proc_results.exp_efforting == 0) &
                (self.proc_results.artifact == 0)
            ),
            'no_efforting': (
                (self.proc_results.early_efforting == 0) &
                (self.proc_results.insp_efforting == 0) &
                (self.proc_results.exp_efforting == 0)
            ),
            'pc_only': (self.proc_results.ventmode == 'pc' ),
            'pc_async_only_no_fam': ((self.proc_results.ventmode == 'pc') & (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa > 1) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            )),
            'pc_no_async': ((self.proc_results.ventmode == 'pc') & (
                (self.proc_results.dta == 0) &
                (self.proc_results.bsa == 0) &
                (self.proc_results.fa == 0) &
                (self.proc_results.static_dca == 0) &
                (self.proc_results.dyn_dca == 0) &
                (self.proc_results.artifact == 0)
            )),
            'pc_prvc': self.proc_results.ventmode.isin(['pc', 'prvc']),
            'pc_prvc_async_only': ((self.proc_results.ventmode.isin(['pc', 'prvc'])) & (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa != 0) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            )),
            'pc_prvc_async_only_no_fam': ((self.proc_results.ventmode.isin(['pc', 'prvc'])) & (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa > 1) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            )),
            'pc_prvc_no_async': ((self.proc_results.ventmode.isin(['pc', 'prvc'])) & (
                (self.proc_results.dta == 0) &
                (self.proc_results.bsa == 0) &
                (self.proc_results.fa == 0) &
                (self.proc_results.static_dca == 0) &
                (self.proc_results.dyn_dca == 0) &
                (self.proc_results.artifact == 0)
            )),
            'pc_prvc_bsa_only': ((self.proc_results.ventmode.isin(['pc', 'prvc'])) & (
                (self.proc_results.bsa != 0)
            )),
            'pc_prvc_dca_only': ((self.proc_results.ventmode.isin(['pc', 'prvc'])) & (
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            )),
            'pc_prvc_dta_only': ((self.proc_results.ventmode.isin(['pc', 'prvc'])) & (
                (self.proc_results.dta != 0)
            )),
            'pc_prvc_async_no_dca_only': ((self.proc_results.ventmode.isin(['pc', 'prvc'])) & (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) &
                (self.proc_results.static_dca == 0) &
                (self.proc_results.dyn_dca == 0)
            )),
            'prvc_only': (self.proc_results.ventmode == 'prvc'),
            'vc_only': (self.proc_results.ventmode == 'vc'),
            'vc_bsa_only': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.bsa != 0)
            )),
            'vc_dta_only': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.dta != 0)
            )),
            'vc_fa_no_fam_only': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.fa > 1)
            )),
            'vc_fa_mild_only': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.fa == 1)
            )),
            'vc_fa_mod_only': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.fa == 2)
            )),
            'vc_fa_sev_only': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.fa == 3)
            )),
            'vc_async_only': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa != 0) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            )),
            'vc_async_only_no_fam': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa > 1) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            )),
            'vc_async_no_fa_only': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) &
                (self.proc_results.fa == 0)
            )),
            'vc_no_async': ((self.proc_results.ventmode == 'vc') & (
                (self.proc_results.dta == 0) &
                (self.proc_results.bsa == 0) &
                (self.proc_results.fa == 0) &
                (self.proc_results.static_dca == 0) &
                (self.proc_results.dyn_dca == 0) &
                (self.proc_results.artifact == 0)
            )),
        }

    def preprocess_ad_std_in_df(self, df, windowing, std_or_mad):
        """
        In patient processed frames, extract descriptive statistics on the patient-level
        such as mean diffs and MAD or std deviation.
        """
        algos = sorted(list(df.algo.unique()))
        ad_std = {algo: [[], [], None, None] for algo in algos}
        ad_col, dev_col = self._get_windowing_colnames(windowing, std_or_mad)

        for algo in algos:
            nan_mask = df[df.algo==algo][ad_col].isna()
            if nan_mask.values.all():
                del ad_std[algo]
                continue
            non_nan = df[df.algo==algo][~nan_mask]
            ad_std[algo][0] = non_nan[ad_col]
            ad_std[algo][1] = non_nan[dev_col]
            ad_std[algo][2] = np.mean(ad_std[algo][0])
            ad_std[algo][3] = np.mean(ad_std[algo][1])

        return ad_std

    def plot_algo_scatter(self, df, windowing, plt_title, figname, individual_patients, std_lim, std_or_mad, algos=None, highlight_algos=None, **kwargs):
        """
        Perform scatterplot for all available algos based on an input per-patient dataframe.

        X-axis is AD and Y-axis is std.
        """
        if algos is not None:
            df = df[df.algo.isin(algos)]
        ad_std = self.preprocess_ad_std_in_df(df, windowing, std_or_mad)
        self._ad_std_scatter(ad_std, windowing, plt_title, figname, individual_patients, std_lim, std_or_mad, highlight_algos=highlight_algos, **kwargs)

    def plot_algo_ad_std_boxplots(self, df, windowing, figname_prefix):
        fig, ax = plt.subplots(figsize=(3*8, 3*3))
        # you can use bootstrap too if you want, but for now I'm not going to
        ad_col, std_col = self._get_windowing_colnames(windowing)
        algo_ordering = sorted(list(df.algo.unique()))

        sns.boxplot(x='algo', y=ad_col, data=df, order=algo_ordering, notch=False, showfliers=False)
        ax.set_ylabel('Difference of (Compliance - Algo)', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.set_ylim(-.4, 26)
        plt.tight_layout()
        fig.savefig(self.results_dir.joinpath('{}_ad_windowing_{}_boxplot_result-mins-{}.png'.format(windowing, figname_prefix, self.n_minutes)).resolve(), dpi=self.dpi)

        fig, ax = plt.subplots(figsize=(3*8, 3*3))
        sns.boxplot(x='algo', y=std_col, data=df, order=algo_ordering, notch=False, showfliers=False)
        ax.set_ylabel('Standard Deviation ($\sigma$) of Algo', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.set_ylim(-.4, 31)
        plt.tight_layout()
        fig.savefig(self.results_dir.joinpath('{}_std_windowing_{}_boxplot_result-mins-{}.png'.format(windowing, figname_prefix, self.n_minutes)).resolve(), dpi=self.dpi)

    def show_individual_breath_by_breath_frame_results(self, df, figname, windowing, **kwargs):
        fig, ax = plt.subplots(figsize=(3*8, 3*3))
        algos_in_frame = set(df.columns).intersection(self.algos_used)
        if windowing is None:
            sorted_diff_cols = sorted(["{}_diff".format(algo) for algo in algos_in_frame])
        else:
            sorted_diff_cols = sorted(["{}_{}_{}".format(algo, windowing, self.window_n) for algo in algos_in_frame])
        proc_frame = self.extract_medians_and_iqr(df, windowing)

        # alphabetical order again
        algos_in_order = sorted(list(proc_frame.algo.unique()))
        self._draw_seaborn_boxplot(
            df[sorted_diff_cols],
            ax,
            proc_frame.medians.values,
            notch=False,
            bootstrap=None,
            palette=[self.algo_colors[algo] for algo in algos_in_order],
            linewidth=kwargs.get('box_lw', None),
        )
        xtick_names = plt.setp(ax, xticklabels=[FileCalculations.shorthand_name_mapping[algo] for algo in sorted(self.algos_used)])
        plt.setp(xtick_names, rotation=kwargs.get('rotation', 60), fontsize=kwargs.get('tick_fontsize', 18))
        plt.setp(ax.get_yticklabels(), fontsize=kwargs.get('tick_fontsize', 18))
        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], ls='--', zorder=0.9, c='red', lw=kwargs.get('lw', 4))
        ax.set_xlim(xlim)
        ax.set_xlabel('')
        ax.set_ylabel(self.label_diff, fontsize=kwargs.get('label_fontsize', 16))
        title = figname.replace('.png', '').replace('_', ' ').replace('-', ': ')
        ax.set_title(title, fontsize=20, pad=25.0)
        ax.grid(False)
        ax.grid(True, lw=kwargs.get('grid_lw', 1), alpha=kwargs.get('grid_alpha', None), axis='y')
        plt.tight_layout()
        fig.savefig(self.results_dir.joinpath(figname).resolve(), dpi=self.dpi)

        # show table of boxplot results
        try:
            self._show_breath_by_breath_algo_table(proc_frame, title)
        except ValueError:  # likely due to an all-nan slice resulting from bootstrap.
            pass
        plt.show(fig)

    def perform_algo_based_multi_window_analysis_regression(self, absolute=True, windows=[5, 10, 20, 50, 100, 200, 400, 800], winsorizor=(0, 0.05), algos=[]):
        """
        Perform multi-window analysis but centered on how algorithms differ by window
        instead of how windows differ by algorithm. Basically we're doing a sensitivity
        analysis by algorithm. We take a set window and then vary algorithms over it
        to see which algo performs best for a certain window size

        :param absolute: return absolute values for WMD calcs
        :param windows: list of window sizes to use
        :param winsorizor: (<low>, 1-<high>) percentiles to choose
        :param algos: restrict to using only specific algorithms. must be a list of algos
        """
        nrows = 4
        algos_to_use = self.algos_used if not algos else algos
        plot_data = [
            (0, 0, 'asynci_{}'),
            (0, 1, 'asynci_no_fam_{}'),
            (1, 0, 'dti_{}'),
            (1, 1, 'bsi_{}'),
            (2, 0, 'dci_{}'),
            (2, 1, 'fai_{}'),
            (3, 0, 'fai_no_fam_{}'),
            (3, 1, 'dtw_wm_{}'),
        ]
        for win_size in windows:
            self.set_new_window_n(win_size)

        for size in windows:
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(3*8, 3*nrows*3))

            for i, j, col in plot_data:
                win_col = col.format(size)

                for algo in algos_to_use:
                    if (('fai' in col and algo in FileCalculations.algos_unavailable_for_vc) or \
                        ('dci' in col and algo in FileCalculations.algos_unavailable_for_pc_prvc)) and not self.no_algo_restrict:
                        continue

                    self._regplot_wmd(self.proc_results, algo, win_col, size, axes[i][j], winsorizor, {'s': 0, 'alpha': .0}, {'label': algo, 'lw': 3}, absolute, False, False, False)

                if len(axes[i][j].lines) != 0:
                    axes[i][j].legend(fontsize=14)
                    y_min = sys.maxsize
                    y_max = -sys.maxsize
                    for line in axes[i][j].lines:
                        x, y = line.get_data()
                        if min(y) < y_min:
                            y_min = min(y)
                        if max(y) > y_max:
                            y_max = max(y)
                    min_ = y_min-5 if not absolute else -1
                    axes[i][j].set_ylim((min_, y_max+5))

            plt.suptitle('Window Size {}'.format(size), fontsize=28)
            plt.tight_layout()
            fig.savefig(self.results_dir.joinpath('algo_based_multi_window_analysis_regression_size_{}_mins_{}.png'.format(size, self.n_minutes)).resolve(), dpi=self.dpi)
            plt.show(fig)

    def perform_multi_window_analysis_regression(self, absolute=True, windows=[5, 10, 20, 50, 100, 200, 400, 800], winsorizor=(0, 0.05), robust=False):
        """
        Show insights from analyzing multiple different window sizes for different
        algorithms.

        :param absolute: return absolute values for WMD calcs
        :param windows: list of window sizes to use
        :param winsorizor: (<low>, 1-<high>) percentiles to choose
        :param robust: (bool) is robust regression or not
        """
        # for now just run some of the least squares algos
        nrows = 4
        plot_data = [
            (0, 0, 'asynci_{}'),
            (0, 1, 'asynci_no_fam_{}'),
            (1, 0, 'dti_{}'),
            (1, 1, 'bsi_{}'),
            (2, 0, 'dci_{}'),
            (2, 1, 'fai_{}'),
            (3, 0, 'fai_no_fam_{}'),
            (3, 1, 'dtw_wm_{}'),
        ]
        for win_size in windows:
            self.set_new_window_n(win_size)

        for algo in self.algos_used:
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(3*8, 3*nrows*3))
            for i, j, col in plot_data:

                for size in windows:
                    win_col = col.format(size)
                    self._regplot_wmd(self.proc_results, algo, win_col, size, axes[i][j], winsorizor, {'s': 0, 'alpha': .0}, {'label': size, 'lw': 3}, absolute, robust, False, False, show_algo_color=False)

                if len(axes[i][j].lines) != 0:
                    axes[i][j].legend()
                    y_min = sys.maxsize
                    y_max = -sys.maxsize
                    for line in axes[i][j].lines:
                        x, y = line.get_data()
                        if min(y) < y_min:
                            y_min = min(y)
                        if max(y) > y_max:
                            y_max = max(y)
                    min_ = y_min-5 if not absolute else -1
                    axes[i][j].set_ylim((min_, y_max+5))

            plt.suptitle(FileCalculations.algo_name_mapping[algo] + ' n: {}'.format(self.window_n), fontsize=28, y=.9)
            plt.tight_layout()
            fig.savefig(self.results_dir.joinpath('{}_multi_window_analysis_regression_mins_{}.png'.format(algo, self.n_minutes)).resolve(), dpi=self.dpi)
            plt.show(fig)

    def perform_multi_window_analysis_bar(self, absolute=True, windows=[5, 10, 20, 50, 100, 200, 400, 800], windowing='smd'):
        """
        Show insights from analyzing multiple different window sizes for different
        algorithms.

        :param absolute: return absolute values for WMD calcs
        :param windows: list of window sizes to use
        :param windowing: 'wmd' for WMD, and 'smd' for SMD
        """
        if windowing not in ['wmd', 'smd']:
            raise ValueError('Must specify a valid window strategy. choices: wmd OR smd')
        for win_size in windows:
            self.set_new_window_n(win_size)

        super_frame = None
        fig, ax = plt.subplots(figsize=(3*8, 4*3))
        windows = ['No Windowing'] + windows
        for size in windows:
            if size == 'No Windowing':
                cols = ['{}_diff'.format(algo) for algo in self.algos_used]
            else:
                cols = ['{}_{}_{}'.format(algo, windowing, size) for algo in self.algos_used]
            if absolute:
                frame = self.proc_results[cols].abs()
            else:
                frame = self.proc_results[cols]
            if size == 'No Windowing':
                rename_cols = {col: col.replace('_diff', '') for col in cols}
            else:
                rename_cols = {col: col.replace('_{}_{}'.format(windowing, size), '')  for col in cols}
            frame = frame.rename(columns=rename_cols)
            frame = frame.melt()
            frame['Window Size'] = size
            if super_frame is None:
                super_frame = frame
            else:
                super_frame = super_frame.append(frame)

        sns.barplot(x='variable', y='value', data=super_frame, hue='Window Size')
        plt.ylabel('{}Difference'.format('Absolute ' if absolute else ''))
        plt.xlabel('Algorithm')
        xtick_names = plt.setp(ax, xticklabels=[FileCalculations.shorthand_name_mapping[algo] for algo in sorted(self.algos_used)])
        plt.setp(xtick_names, rotation=90)
        plt.legend(loc='upper right', framealpha=.7)
        plt.tight_layout()
        fig.savefig(self.results_dir.joinpath('multi_window_analysis_bar-windowing-{}-mins-{}-windows_{}.png'.format(windowing, self.n_minutes, '_'.join([str(w) for w in windows]))).resolve(), dpi=self.dpi)
        plt.show(fig)

    def perform_single_window_by_patients_and_breaths(self, patient_breath_map, absolute=True, winsorizor=(0, 0.05), algos=[], robust=False, robust_and_reg=False, show_median=False):
        """
        :param patient_breath_map: mapping of {patient: [bn_low, bn_high]} to use
        :param absolute: return absolute values for WMD calcs
        :param winsorizor: (<low>, 1-<high>) percentiles to choose
        :param algos: list of algos to use
        :param robust: (bool) is robust regression or not
        :param robust_and_reg: (bool) show robust and non-robust regression
        :param show_median: (bool) show median per x
        """
        idxs = None
        for patient, bns in patient_breath_map.items():
            tmp = self.proc_results[(self.proc_results.patient == patient) & (self.proc_results.rel_bn >= bns[0]) & (self.proc_results.rel_bn <= bns[1])].index
            if isinstance(idxs, type(None)):
                idxs = tmp
            else:
                idxs = idxs.append(tmp)
        df = self.proc_results.loc[idxs]
        self._perform_single_window_analysis(df, absolute, winsorizor, algos, robust, robust_and_reg, show_median)

    def perform_single_window_analysis(self, absolute=True, winsorizor=(0, 0.05), algos=[], robust=False, robust_and_reg=False, show_median=False):
        """
        Show insights form analyzing windowed calculations

        Lineplots: For each algorithm display regression lineplot showing the
                   window's performance across varying scenarios.

        :param absolute: return absolute values for WMD calcs
        :param winsorizor: (<low>, 1-<high>) percentiles to choose
        :param algos: list of algos to use
        :param robust: (bool) is robust regression or not
        :param robust_and_reg: (bool) show robust and non-robust regression
        :param show_median: (bool) show median per x
        """
        self._perform_single_window_analysis(self.proc_results, absolute, winsorizor, algos, robust, robust_and_reg, show_median)

    def plot_breath_by_breath_results(self, only_patient=None, exclude_cols=[], windowing=None, **kwargs):
        only_patient_wrapper = lambda df, pt: df[df.patient == pt] if pt is not None else df
        exclude_algos_wrapper = lambda df, cols: df.drop(cols, axis=1) if exclude_cols else df
        windowing_mod = lambda x, y: x.replace('.png', '-windowing-{}-mins-{}.png'.format(y, self.n_minutes))
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.proc_results, only_patient), exclude_cols),
            windowing_mod('breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['no_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('no_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['vc_only'][self.window_n], only_patient), exclude_cols),
            windowing_mod('vc_only_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['vc_no_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('vc_no_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['vc_only_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('vc_only_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['pc_only'][self.window_n], only_patient), exclude_cols),
            windowing_mod('pc_only_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['pc_no_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('pc_no_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['pc_only_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('pc_only_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['prvc_only'][self.window_n], only_patient), exclude_cols),
            windowing_mod('prvc_only_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['prvc_no_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('prvc_no_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['prvc_only_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('prvc_only_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['no_efforting'][self.window_n], only_patient), exclude_cols),
            windowing_mod('no_efforting_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['early_efforting'][self.window_n], only_patient), exclude_cols),
            windowing_mod('early_efforting_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['insp_efforting'][self.window_n], only_patient), exclude_cols),
            windowing_mod('insp_efforting_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['exp_efforting'][self.window_n], only_patient), exclude_cols),
            windowing_mod('exp_efforting_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['all_efforting'][self.window_n], only_patient), exclude_cols),
            windowing_mod('all_efforting_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['all_pressure_only'][self.window_n], only_patient), exclude_cols),
            windowing_mod('pc_prvc_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['all_pressure_no_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('pc_prvc_no_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['all_pressure_only_async'][self.window_n], only_patient), exclude_cols),
            windowing_mod('pc_prvc_only_async_breath_by_breath_results.png', windowing),
            windowing,
            **kwargs,
        )

    def plot_per_patient_results(self, windowing, std_or_mad, individual_patients=False, show_boxplots=True, std_lim=None, **kwargs):
        """
        Plot patient by patient results

        :param windowing: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        :param std_or_mad: use standard deviation ('std') or MAD ('mad')
        :param individual_patients: show individual_patients scatter points
        :param show_boxplots: show boxplots after scatter plots
        :param std_lim: limit graphs by standard deviation within certain
                        factor. Normally is set to None (no limit). But can be
                        set to any floating value > 0.
        """
        windowing_mod = lambda x, y: x.replace('.png', '-windowing-{}-mins-{}.png'.format(y, self.n_minutes))
        # Patient by Patient. All breathing
        self.plot_algo_scatter(
            self.pp_frames['all'][self.window_n],
            windowing,
            'Patient by patient results. No filters',
            windowing_mod('patient_by_patient_result.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )
        if show_boxplots:
            self.plot_algo_ad_std_boxplots(self.pp_frames['all'][self.window_n], windowing, 'patient_by_patient')

        # Patient by patient. no asynchronies
        self.plot_algo_scatter(
            self.pp_frames['no_async'][self.window_n],
            windowing,
            'Patient by patient results. No Asynchronies',
            windowing_mod('patient_by_patient_no_async_result.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )
        if show_boxplots:
            self.plot_algo_ad_std_boxplots(self.pp_frames['no_async'][self.window_n], windowing, 'patient_by_patient_no_async')

        # VC only patient by patient.
        self.plot_algo_scatter(
            self.pp_frames['vc_only'][self.window_n],
            windowing,
            'Patient by patient results. VC only',
            windowing_mod('patient_by_patient_vc_only.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )
        if show_boxplots:
            self.plot_algo_ad_std_boxplots(self.pp_frames['vc_only'][self.window_n], windowing, 'vc_only_pbp')

        # VC only, non-asynchronies
        self.plot_algo_scatter(
            self.pp_frames['vc_no_async'][self.window_n],
            windowing,
            'Patient by patient results. VC, No Asynchronies',
            windowing_mod('patient_by_patient_vc_no_asynchronies.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # PC only patient by patient.
        self.plot_algo_scatter(
            self.pp_frames['pc_only'][self.window_n],
            windowing,
            'Patient by patient results. PC only',
            windowing_mod('patient_by_patient_pc_only.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )
        if show_boxplots:
            self.plot_algo_ad_std_boxplots(self.pp_frames['pc_only'][self.window_n], windowing, 'pc_only_pbp')

        # PC only, non-asynchronies
        self.plot_algo_scatter(
            self.pp_frames['pc_no_async'][self.window_n],
            windowing,
            'Patient by patient results. PC, No Asynchronies',
            windowing_mod('patient_by_patient_pc_no_asynchronies.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # PRVC only patient by patient.
        self.plot_algo_scatter(
            self.pp_frames['prvc_only'][self.window_n],
            windowing,
            'Patient by patient results. PRVC only',
            windowing_mod('patient_by_patient_prvc_only.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )
        if show_boxplots:
            self.plot_algo_ad_std_boxplots(self.pp_frames['prvc_only'][self.window_n], windowing, 'prvc_only_pbp')

        # PRVC only, non-asynchronies
        self.plot_algo_scatter(
            self.pp_frames['prvc_no_async'][self.window_n],
            windowing,
            'Patient by patient results. PRVC, No Asynchronies',
            windowing_mod('patient_by_patient_prvc_no_asynchronies.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # mode-agnostic algorithms only
        self.plot_algo_scatter(
            self.pp_frames['all'][self.window_n],
            windowing,
            'Patient by patient results. Mode Agnostic Algos',
            windowing_mod('patient_by_patient_mode_agnostic_algos.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            algos=(set(self.algos_used).difference(FileCalculations.algos_unavailable_for_pc_prvc)).difference(FileCalculations.algos_unavailable_for_vc),
            **kwargs,
        )

        # no efforting only
        self.plot_algo_scatter(
            self.pp_frames['no_efforting'][self.window_n],
            windowing,
            'Patient by patient results. No Apparent Efforting',
            windowing_mod('patient_by_patient_no_efforting.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # early efforting only
        self.plot_algo_scatter(
            self.pp_frames['early_efforting'][self.window_n],
            windowing,
            'Patient by patient results. Early Efforting',
            windowing_mod('patient_by_patient_early_efforting.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # insp efforting only
        self.plot_algo_scatter(
            self.pp_frames['insp_efforting'][self.window_n],
            windowing,
            'Patient by patient results. Inspiratory Efforting',
            windowing_mod('patient_by_patient_insp_efforting.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # exp efforting only
        self.plot_algo_scatter(
            self.pp_frames['exp_efforting'][self.window_n],
            windowing,
            'Patient by patient results. Expiratory Efforting',
            windowing_mod('patient_by_patient_exp_efforting.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # all efforting only
        self.plot_algo_scatter(
            self.pp_frames['all_efforting'][self.window_n],
            windowing,
            'Patient by patient results. All Efforting',
            windowing_mod('patient_by_patient_all_efforting.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # PC/PRVC only
        self.plot_algo_scatter(
            self.pp_frames['all_pressure_only'][self.window_n],
            windowing,
            'Patient by patient results. PC/PRVC only',
            windowing_mod('patient_by_patient_pc_prvc_only.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        if show_boxplots:
            self.plot_algo_ad_std_boxplots(self.pp_frames['all_pressure_only'][self.window_n], windowing, 'pressure_only_pbp')

        # PC/PRVC only no async
        self.plot_algo_scatter(
            self.pp_frames['all_pressure_no_async'][self.window_n],
            windowing,
            'Patient by patient results. PC/PRVC only. No Asynchrony',
            windowing_mod('patient_by_patient_pc_prvc_no_async.png', windowing),
            individual_patients,
            std_lim,
            std_or_mad,
            **kwargs,
        )

        # I go back and forth in between questioning whether this belongs here or in per_patient
        if show_boxplots:
            for algo in self.algos_used:
                fig, ax = plt.subplots(figsize=(3*8, 3*2))
                sns.boxplot(x='patient_id', y="{}_diff".format(algo), data=self.proc_results, showfliers=False)
                ax.set_ylabel('Difference between Compliance and Algo')
                ax.set_xlabel('Patient', fontsize=16)
                ax.set_title('{} plot by patient'.format(algo), fontsize=20)
                # want to keep a constant y perspective to compare algos
                #ax.set_ylim(-0.07, 0.07)
                plt.tight_layout()
                fig.savefig(self.results_dir.joinpath(windowing_mod('{}_breath_by_breath_patient_result.png'.format(algo), windowing)).resolve(), dpi=self.dpi)

    def reset_plat_minutes(self, n_minutes):
        try:
            return pd.read_pickle(str(self.results_dir.joinpath('ResultsContainer_mins_{}.pkl'.format(n_minutes))))
        except OSError:
            self.collate_data(self.algos_used, n_minutes)
            self.analyze_results()
            return self

    def save_results(self):
        if not self.results_dir.parent.exists():
            self.results_dir.parent.mkdir()
        if not self.results_dir.exists():
            self.results_dir.mkdir()
        pd.to_pickle(self, str(self.results_dir.joinpath('ResultsContainer_mins_{}.pkl'.format(self.n_minutes))))

    def set_new_window_n(self, window_n):
        """
        Set a new number of samples for windows.

        :param window_n: new length of WMD window
        """
        self.window_n = window_n
        # make sure to only analyze results if we haven't previously done the
        # analysis for this window size before. Just use asynci as our canary
        if not 'asynci_{}'.format(window_n) in self.proc_results.columns:
            self.analyze_results()

    def update_for_new_metadata(self, save=True):
        """
        update existing cohort analysis for new metadata that has
        come in. Generally this function is run because of a new
        metadata field that comes in like RASS. However it can also
        be due to an update in a field as well such as ventmode annotations
        """
        extra_cols = [
            'rass', 'ventmode', 'dta', 'bsa', 'artifact', 'fa', 'dtw',
            'dyn_dca', 'dyn_dca_timing', 'static_dca', 'fa_loc',
        ]
        for patient in self.proc_results.patient.unique():
            # get patient files
            processed_dir = Path(__file__).parent.joinpath('../dataset/processed_data/{}'.format(patient))
            extra_fs = processed_dir.glob('*.extra.pkl')
            extra_data = pd.concat([pd.read_pickle(str(f)) for f in extra_fs])
            # now update the processed data with the extra data
            #
            # There is a possibility that there will be breaths in extra data that arent in the
            # analysis because of filtering. you need to make sure the code can handle that.
            pt_slice = self.proc_results[self.proc_results.patient == patient]
            merged = pt_slice[['rel_bn', 'abs_bs']].merge(extra_data, how='left', on=['rel_bn', 'abs_bs'])
            if len(extra_data) != len(pt_slice):
                print('uneq patient slices on merge. fill this case out')
                import IPython; IPython.embed()
                raise Exception('unequal patient slices, this case needs to be covered')

            # this can happen because left index order is preserved
            merged.index = pt_slice.index
            for col in extra_cols:
                self.proc_results.loc[merged.index, col] = merged[col]

        if save:
            # save frame
            self.save_results()

    def visualize_patients(self, patients, algos, extra_mask=None, ts_xlim=None, ts_ylim=None, windowing=None, **kwargs):
        if algos == 'all':
            algos = self.algos_used
        algo_cols = algos
        diff_cols = ['{}_diff'.format(c) for c in algo_cols]
        wmd_cols = ['{}_wmd_{}'.format(c, self.window_n) for c in algo_cols]
        smd_cols = ['{}_smd_{}'.format(c, self.window_n) for c in algo_cols]
        final_cols = algo_cols + diff_cols + wmd_cols + smd_cols + ['rel_bn', 'gold_stnd_compliance', 'patient_id']

        if not isinstance(extra_mask, type(None)):
            patient_df = self.proc_results[(self.proc_results.patient.isin(patients)) & extra_mask][final_cols]
        else:
            patient_df = self.proc_results[self.proc_results.patient.isin(patients)][final_cols]

        patient_df[algo_cols].plot(figsize=(3*8, 4*3), colormap=cc.cm.glasbey, fontsize=16)
        plt.legend(fontsize=16)
        plt.title(', '.join(patients), fontsize=20)
        if ts_xlim is not None:
            plt.xlim(ts_xlim)
        if ts_ylim is not None:
            plt.ylim(ts_ylim)
        plt.plot(patient_df.gold_stnd_compliance, label='gt')
        plt.xlabel('DataFrame index')
        plt.tight_layout()
        plt.savefig(self.results_dir.joinpath('{}-individual-breath-time-series-plot-mins-{}.png'.format('-'.join(patients), self.n_minutes)).resolve(), dpi=self.dpi)
        plt.show()

        pp_custom = self.analyze_per_patient_df(patient_df)
        self.plot_algo_scatter(
            pp_custom,
            windowing,
            'Patients {}. Custom plot'.format(', '.join(patients)),
            '{}_custom_scatter_plot-windowing-{}-mins-{}.png'.format('-'.join(patients), windowing, self.n_minutes),
            False,
            None,
            **kwargs,
        )

        # breath by breath results
        self.show_individual_breath_by_breath_frame_results(
            patient_df,
            '{}_custom_breath_by_breath-windowing-{}-mins-{}.png'.format('-'.join(patients), windowing, self.n_minutes),
            windowing,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('--only-patient', help='only run results for specific patient', nargs='*')
    parser.add_argument('--algos', nargs="*", default='all')
    parser.add_argument('--tc-algos', choices=['al_rawas', 'brunner', 'fuzzy', 'ikeda', 'lourens', 'wiri', 'all'], nargs='*', default='all')
    parser.add_argument('-ltc', '--lourens-tc-choice', type=int, default=50)
    parser.add_argument('-dp', '--data-path', default=str(Path(__file__).parent.joinpath('../dataset/processed_data')))
    parser.add_argument('--cvc-only', action='store_true', help='only analyze cvc data')
    parser.add_argument('--no-cvc', action='store_true', help='dont analyze cvc data')
    parser.add_argument('--no-algo-restrictions', action='store_true', help='Do not restrict algorithms to their designed modes. Run in all modes possible')

    args = parser.parse_args()

    all_patient_dirs = Path(args.data_path).glob('*')
    results = ResultsContainer(args.experiment_name, 20, args.no_algo_restrictions)

    for dir_ in sorted(list(all_patient_dirs)):
        if args.cvc_only and 'cvc' not in str(dir_):
            continue

        if args.no_cvc and 'cvc' in str(dir_):
            continue

        patient_id = dir_.name
        for file in dir_.glob('*.raw.npy'):
            patient_id = file.parent.name
            if args.only_patient and patient_id not in args.only_patient:
                continue
            extra = pd.read_pickle(str(file).replace('raw.npy', 'extra.pkl'))
            if 'cvc' in str(file):
                compliance_f = dir_.joinpath('compliance.txt')
                recorded_compliance = int(open(compliance_f).read().strip())
            else:
                recorded_compliance = None
            calcs = FileCalculations(patient_id, str(file), args.algos, 9, extra, tc_algos=args.tc_algos, lourens_tc_choice=args.lourens_tc_choice, recorded_compliance=recorded_compliance, no_algo_restrict=args.no_algo_restrictions)
            try:
                calcs.analyze_file()
            except Exception as err:
                print('Failed on file: {}'.format(str(file)))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)
                print(err)
                return
            results.add_results_df(patient_id, calcs.results)
    results.collate_data(calcs.algos_used, n_minutes=30)
    results.save_results()


if __name__ == '__main__':
    main()
