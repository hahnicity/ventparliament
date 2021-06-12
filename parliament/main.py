"""
main
~~~~

Main analysis code
"""
import argparse
from copy import copy
from itertools import cycle
from pathlib import Path
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
from scipy.stats import median_absolute_deviation
from scipy.stats.mstats import winsorize
from seaborn.categorical import _BoxPlotter
from seaborn.utils import remove_na
import seaborn as sns
from skimage.util.shape import view_as_windows

from parliament.analyze import FileCalculations


def _rolling_func(vals, win_size, func):
    strides = view_as_windows(vals, (win_size,), step=1)
    tmp = func(strides, axis=1)
    return np.append([np.nan]*(win_size-1), tmp)


def rolling_nan_median(vals, win_size):
    return _rolling_func(vals, win_size, np.nanmedian)


def rolling_nan_mean(vals, win_size):
    return _rolling_func(vals, win_size, np.nanmean)


def sequential_nan_median(vals, win_size):
    strides = view_as_windows(vals, (win_size,), step=win_size)
    meds = np.nanmedian(strides, axis=1)
    # these next 2 lines fill in nans between sequence val and next sequence val
    tmp = np.expand_dims(meds[:-1], axis=0)
    tmp = np.pad(tmp.T, ((0, 0), (0, win_size-1)), constant_values=np.nan).ravel()
    tmp = np.append(tmp, [meds[-1]])
    # this line makes sure any remaining nans are filled.
    return np.pad(tmp, (win_size-1, len(vals)-(len(tmp)+win_size-1)), constant_values=np.nan)


class ResultsContainer(object):

    def __init__(self, experiment_name, window_n):
        self.proc_results = []
        self.algos_used = []
        self.experiment_name = experiment_name
        self.raw_results = []
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
        self.pp_frames = {}
        self.bb_frames = {}

    def _draw_seaborn_boxplot_with_bootstrap(self, data, ax, medians=None, **kwargs):
        """
        Basically an exact replica of what happens in seaborn except for support of
        usermedians
        """
        plotter = _BoxPlotter(x=None, y=None, hue=None, data=data, order=None, hue_order=None,
                              orient=None, color=None, palette=None, saturation=.75, width=.8,
                              dodge=True, fliersize=5, linewidth=None)
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

        bs_index = np.random.randint(M, size=(self.boot_resamples, M))
        bsData = col_vals[bs_index]
        estimate = np.nanmedian(bsData, axis=1, overwrite_input=True)

        # this is basically the same thing that scipy.stats.iqr does
        iqr = np.median(np.nanpercentile(bsData, percentiles, axis=1), axis=1)
        return np.median(estimate), iqr

    def _change_td_to_bold(self, soup, td):
        """
        Change td tag so that interior text is boldfaced.
        """
        val = td.string
        td.clear()
        td.insert(0, soup.new_tag('b'))
        td.b.string = val

    def _get_windowing_colnames(self, windowing):
        if windowing is None:
            mad_col = 'mad_pt'
            std_col = 'std_pt'
        elif windowing == 'wmd':
            mad_col = 'mad_wmd'
            std_col = 'std_wmd'
        elif windowing == 'smd':
            mad_col = 'mad_smd'
            std_col = 'std_smd'
        else:
            raise ValueError('windowing variable must be None, "wmd", or "smd"')
        return mad_col, std_col

    def _mad_std_scatter(self, mad_std, windowing, plt_title, figname, algos_in_order, individual_patients, std_lim):
        """
        Perform scatter plot using with MAD and std information for each algorithm.
        """
        markers = cycle(self.scatter_marker_symbols)
        colors = [cc.cm.glasbey(i) for i in range(len(algos_in_order))]
        algo_dict = {algo: {'m': next(markers), 'c': colors[i]} for i, algo in enumerate(algos_in_order)}
        fig, ax = plt.subplots(figsize=(3*6.5, 3*2.5))

        if individual_patients:
            for i, algo in enumerate(algos_in_order):
                ax.scatter(
                    x=mad_std[algo][0],
                    y=mad_std[algo][1],
                    marker=algo_dict[algo]['m'],
                    color=algo_dict[algo]['c'],
                    label=algo,
                    alpha=0.4,
                    s=100,
                    zorder=1
                )

        for i, algo in enumerate(algos_in_order):
            ax.scatter(
                x=mad_std[algo][2],
                y=mad_std[algo][3],
                marker=algo_dict[algo]['m'],
                color=algo_dict[algo]['c'],
                label=algo if not individual_patients else None,
                alpha=.9,
                s=350,
                edgecolors='black',
                zorder=len(algos_in_order)+2-i,
                linewidths=1,
            )
        x = [mad_std[a][2] for a in algos_in_order]
        y = [mad_std[a][3] for a in algos_in_order]
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylabel('Standard Deviation ($\sigma$) of Algo', fontsize=16)
        ax.set_xlabel('MAD (Median Absolute Deviation) of (Compliance - Algo)', fontsize=16)
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

        fig.legend(fontsize=16, loc='center right')
        ax.set_title(plt_title, fontsize=20)
        figname = str(self.results_dir.joinpath(figname).resolve()).replace('.png', '-windowing-{}.png'.format(windowing))
        fig.savefig(figname, dpi=self.dpi)

        # show table of boxplot results
        table = PrettyTable()
        table.field_names = ['Algorithm', 'Shorthand Name', 'MAD (Median Absolute Deviation)', 'Standard Deviation (std)']
        medians = np.array([mad_std[algo][2] for algo in algos_in_order])
        stds = np.array([mad_std[algo][3] for algo in algos_in_order])
        medians = medians.round(2)
        stds = stds.round(2)
        for i, algo in enumerate(algos_in_order):
            table.add_row([FileCalculations.algo_name_mapping[algo], algo, medians[i], stds[i]])

        soup = BeautifulSoup(table.get_html_string())
        min_median = np.nanargmin(medians)
        min_std = np.nanargmin(stds)
        # the +1 is because the header is embedded in a <tr> element
        min_med_elem = soup.find_all('tr')[min_median+1]
        min_std_elem = soup.find_all('tr')[min_std+1]

        self._change_td_to_bold(soup, min_med_elem.find_all('td')[2])
        self._change_td_to_bold(soup, min_std_elem.find_all('td')[3])

        display(HTML('<h2>{}</h2>'.format(plt_title)))
        display(HTML(soup.prettify()))
        plt.show(fig)
        return mad_std, algos_in_order

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

            plt.suptitle(FileCalculations.algo_name_mapping[algo] + ' n: {}'.format(self.window_n), fontsize=28, y=.9)
            plt.show(fig)

    def _regplot_wmd(self, df, algo, x_col, win_size, ax, winsorizor, scatter_kws, line_kws, absolute, robust, robust_and_reg, show_median):
        """
        Perform regression plotting for WMD versus some index

        :param df: DataFrame of results we want to analyze
        """

        # At least for now just skip the plot if its not sensible to perform
        if ('fai' in x_col and algo in FileCalculations.algos_unavailable_for_vc) or \
            ('dci' in x_col and algo in FileCalculations.algos_unavailable_for_pc_prvc):
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

        # XXX can implement smoother to enable dtw viz. but until then, median just
        # looks like a jagged mess. You can also do scatter for DTW median too.
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

    def _show_breath_by_breath_algo_table(self, algos_in_order, medians, iqr, stds, title):
        """
        Show table of boxplot results for breath by breath analysis of algorithms.
        """
        table = PrettyTable()
        table.field_names = ['Algorithm', 'Shorthand Name', 'Median Diff', '25% IQR', '75% IQR', 'std']
        medians = medians.round(2)
        iqr = iqr.round(2)
        stds = stds.round(2)
        for i, algo in enumerate(algos_in_order):
            if not np.isnan(medians[i]):
                table.add_row([FileCalculations.algo_name_mapping[algo], algo, medians[i], iqr[i, 0], iqr[i, 1], stds[i]])
            else:
                table.add_row([FileCalculations.algo_name_mapping[algo], algo, '-', '-', '-', '-'])

        soup = BeautifulSoup(table.get_html_string())
        min_median = np.nanargmin(abs(medians))
        # XXX in the future we should revisit this before publication
        min_iqr_rel_to_0 = np.nanargmin(abs(iqr).sum(axis=1))
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

    @classmethod
    def load_from_experiment_name(cls, experiment_name):
        """
        Load results container but even easier using the container.pkl obj
        """
        results_dir = Path(__file__).parent.joinpath('results', experiment_name)
        cls = pd.read_pickle(results_dir.joinpath('ResultsContainer.pkl'))
        # a bit dirty, but some older analyses dont have this attr
        try:
            cls.scatter_marker_symbols
        except AttributeError:
            cls.scatter_marker_symbols = [
                'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'P', 'X', 'D', 'd', 'H',
                '$\Join$', '$\clubsuit$', '$\spadesuit$', '$\heartsuit$', '$\$$',
                '$\dag$', '$\ddag$', '$\P$'
            ]
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
            # find MAD per patient, per algo
            algos_in_frame = set(df.columns).intersection(self.algos_used)
            for algo in algos_in_frame:
                row = [patient_id, algo]
                # pandas divides std by n-1, and numpy divides by n. We really
                # should be dividing by n here, so thats why we use np.nanstd instead
                # of using DataFrame.std(skipna=True). Also we weren't using skipna
                # originally, which was a big error
                row.append(np.nanmedian(frame['{}_diff'.format(algo, self.window_n)].abs()))
                row.append(np.nanstd(frame[algo]))
                row.append(np.nanmedian(frame['{}_wmd_{}'.format(algo, self.window_n)].abs()))
                row.append(np.nanstd(frame['{}_wmd_{}'.format(algo, self.window_n)]))
                row.append(np.nanmedian(frame['{}_smd_{}'.format(algo, self.window_n)].abs()))
                row.append(np.nanstd(frame['{}_smd_{}'.format(algo, self.window_n)]))
                row_results.append(row)
        cols = ['patient_id', 'algo', 'mad_pt', 'std_pt', 'mad_wmd', 'std_wmd', 'mad_smd', 'std_smd']
        return pd.DataFrame(row_results, columns=cols)

    def analyze_results(self):
        """
        Analyze all results obtained.

        1. Analyze results on a breath by breath basis
        2. Analyze results on a patient by patient basis
            1/2a. Obtain MAD (median absolute dev) between algo and gold stnd compliance
        """
        if isinstance(self.proc_results, list):
            warn('Called analyze_results before any results were collated. Call collate_data first!')
            return

        self.calc_windows(self.proc_results)
        self.calc_async_index(self.proc_results)

        masks = self.get_masks()
        async_mask = masks['async']

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
        pd.to_pickle(self, self.results_dir.joinpath('ResultsContainer.pkl'))

    def calc_windows(self, df):
        """
        Calculates the windowed stats of an algorithm for
        a set window size. Calcs windowed median deviation (WMD) and sequential
        median deviation (SMD).

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
            if algo in FileCalculations.algos_unavailable_for_vc:
                df.loc[df.ventmode == 'vc', [wm_colname, wmd_colname, diff_colname]] = np.nan
            elif algo in FileCalculations.algos_unavailable_for_pc_prvc:
                df.loc[df.ventmode != 'vc', [wm_colname, wmd_colname, diff_colname]] = np.nan

        for patiend_id, pt_df in df.groupby('patient_id'):
            for algo in self.algos_used:
                tmp = sequential_nan_median(pt_df[algo].values, self.window_n)
                df.loc[pt_df.index, '{}_sm_{}'.format(algo, self.window_n)] = tmp

        for algo in self.algos_used:
            sm_colname = '{}_sm_{}'.format(algo, self.window_n)
            smd_colname = '{}_smd_{}'.format(algo, self.window_n)
            df[smd_colname] = df.gold_stnd_compliance - df[sm_colname]
            if algo in FileCalculations.algos_unavailable_for_vc:
                df.loc[df.ventmode == 'vc', [sm_colname, smd_colname]] = np.nan
            elif algo in FileCalculations.algos_unavailable_for_pc_prvc:
                df.loc[df.ventmode != 'vc', [sm_colname, smd_colname]] = np.nan

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

    def collate_data(self, algos_used):
        """
        Now that (presumably) all patient results have been tabulated we can finally determine
        what our gold standard compliances are for specific time points. Filter
        """
        self.algos_used = algos_used
        proc_results = pd.concat(self.raw_results)
        proc_results.index = range(len(proc_results))
        for patient, frame in proc_results.groupby('patient_id'):
            valid_plats = frame[frame.is_valid_plat == True]
            # some rows will have multiple plats overlapping with them. we can average the plat
            # get a compliance, plateau pressure, and driving pressure
            for i, row in frame.iterrows():
                plats_in_range = valid_plats[(valid_plats.abs_bs - pd.Timedelta(hours=0.5) < row.abs_bs) & (row.abs_bs < valid_plats.abs_bs + pd.Timedelta(hours=0.5))]
                proc_results.loc[i, 'gold_stnd_compliance'] = plats_in_range.gold_stnd_compliance.mean()
                proc_results.loc[i, 'p_plat'] = (proc_results.loc[i, 'tvi']/proc_results.loc[i, 'gold_stnd_compliance']) + proc_results.loc[i, 'peep']
                proc_results.loc[i, 'p_driving'] = proc_results.loc[i, 'p_plat'] - proc_results.loc[i, 'peep']

        self.proc_results = proc_results
        # filter outliers by patient
        for algo in algos_used:
            for patient_id, df in self.proc_results.groupby('patient_id'):
                inf_idxs = df[(df[algo] == np.inf) | (df[algo] == -np.inf)].index
                df.loc[inf_idxs, algo] = np.nan
                self.proc_results.loc[inf_idxs, algo] = np.nan
                # I've found that mean can blow up in the presence of outliers.
                # So instead use the median
                algo_median = df[algo].median(skipna=True)
                algo_mad = median_absolute_deviation(df[algo].values, nan_policy='omit')
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

    def compare_patient_level_masks(self, mask1_name, mask2_name, windowing, individual_patients=False, std_lim=None):
        """
        Compare results of different masks to each other on patient by patient basis.
        Plot results out with scatter plots as usual.

        :param mask1_name: mask name based on masks obtained from `get_masks`
        :param mask2_name: mask name based on masks obtained from `get_masks`
        :param windowing: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        """
        masks = self.get_masks()
        mask1 = masks[mask1_name]
        mask2 = masks[mask2_name]

        pp1 = self.analyze_per_patient_df(self.proc_results[mask1])
        pp2 = self.analyze_per_patient_df(self.proc_results[mask2])
        algos_in_order = sorted(list(pp1.algo.unique()))

        mad_std1, _ = self.preprocess_mad_std_in_df(pp1, windowing)
        mad_std2, _ = self.preprocess_mad_std_in_df(pp2, windowing)

        mad_std = copy(mad_std1)
        for algo in algos_in_order:
            for i in range(4):
                mad_std[algo][i] = mad_std2[algo][i] - mad_std1[algo][i]

        plt_title = '{} vs {}'.format(mask1_name, mask2_name)
        figname = '{}_vs_{}.png'.format(mask1_name, mask2_name)
        return self._mad_std_scatter(mad_std, windowing, plt_title, figname, algos_in_order, individual_patients, std_lim)

    def compare_breath_level_masks(self, mask1_name, mask2_name, figname='custom_breath_by_breath.png'):
        """
        Compare results of different masks to each other on breath by breath results.

        :param mask1_name: mask name based on masks obtained from `get_masks`
        :param mask2_name: mask name based on masks obtained from `get_masks`
        :param figname: figure name to save
        """
        algos_in_frame = set(self.proc_results.columns).intersection(self.algos_used)
        diff_cols = ["{}_diff".format(algo) for algo in algos_in_frame]
        sorted_diff_cols = sorted(diff_cols)

        masks = self.get_masks()
        mask1 = masks[mask1_name]
        mask2 = masks[mask2_name]
        orig_bb1 = self.proc_results[mask1]
        orig_bb2 = self.proc_results[mask2]
        bb1 = orig_bb1[sorted_diff_cols].melt()
        bb2 = orig_bb2[sorted_diff_cols].melt()
        bb1['Mask'] = mask1_name.replace('_', ' ')
        bb2['Mask'] = mask2_name.replace('_', ' ')
        df = pd.concat([bb1, bb2])
        df = df.rename(columns={'variable': 'algo'})

        fig, ax = plt.subplots(figsize=(3*8, 3*3))
        # XXX add WMD option

        # alphabetical order again
        algos_in_order = sorted([algo.replace('_diff', '') for algo in sorted_diff_cols])
        #colors = [cc.cm.glasbey(i) for i in range(len(algos_in_order))]
        sns.boxplot(x='algo', y='value', data=df, hue='Mask', ax=ax, notch=False, bootstrap=self.boot_resamples, showfliers=False, palette='Set2')
        xtick_names = plt.setp(ax, xticklabels=algos_in_order)
        plt.setp(xtick_names, rotation=60, fontsize=14)
        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], ls='--', zorder=0, c='red')
        ax.set_ylabel('Difference between Compliance and Algo', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.legend(fontsize=16)
        # want to keep a constant y perspective to compare algos
        #ax.set_ylim(-0.025, 0.025)
        title = figname.replace('.png', '').replace('_', ' ')
        ax.set_title(title, fontsize=20)
        fig.savefig(self.results_dir.joinpath(figname).resolve(), dpi=self.dpi)

        medians, iqr = self.extract_medians_and_iqr(orig_bb1[sorted_diff_cols])
        stds = orig_bb1[sorted_diff_cols].std().values
        self._show_breath_by_breath_algo_table(algos_in_order, medians, iqr, stds, mask1_name)

        medians, iqr = self.extract_medians_and_iqr(orig_bb2[sorted_diff_cols])
        stds = orig_bb2[sorted_diff_cols].std().values
        self._show_breath_by_breath_algo_table(algos_in_order, medians, iqr, stds, mask2_name)
        plt.show(fig)

    def extract_medians_and_iqr(self, df):
        medians = []
        iqrs = []
        for col in df.columns:
            med, iqr = self._bootstrap(df[col].values)
            iqrs.append(iqr)
            medians.append(med)
        return np.array(medians), np.array(iqrs)

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
            '% Female',
            'Age (IQR)',
            'median RASS (IQR)',
            '% paralyzed',
            # ventmode n
            'n vc {}s'.format(pt_or_exp),
            'n pc {}s'.format(pt_or_exp),
            'n prvc {}s'.format(pt_or_exp),
            # general breath counts
            'total breaths',
            'total vc breaths'.format(pt_or_exp),
            'total pc breaths'.format(pt_or_exp),
            'total prvc breaths',
            'mean breaths per {}'.format(pt_or_exp),
            'mean vc per {}'.format(pt_or_exp),
            'mean pc per {}'.format(pt_or_exp),
            'mean prvc per {}'.format(pt_or_exp),
            # asynchronous breath counts
            'total vc async breaths',
            'total pc async breaths',
            'total prvc async breaths',
            'mean vc async per {}'.format(pt_or_exp),
            'mean pc async per {}'.format(pt_or_exp),
            'mean prvc async per {}'.format(pt_or_exp),
            # deeper dive into asynchronies
            'total dta breaths',
            'total bsa breaths',
            'total fa breaths',
            'total static/dynamic dca breaths',
            'total static dca breaths',
            'total dynamic dca breaths',
            # proportions of each async out of total async
            'proportion dta of async',
            'proportion bsa of async',
            'proportion fa of async',
            'proportion static/dynamic dca of async',
        ]
        n_patients = len(self.proc_results.patient_id.unique())
        masks = self.get_masks()
        async_mask = masks['async']
        vc_pts = len(self.proc_results[self.proc_results.ventmode=='vc'].patient_id.unique())
        pc_pts = len(self.proc_results[self.proc_results.ventmode=='pc'].patient_id.unique())
        prvc_pts = len(self.proc_results[self.proc_results.ventmode=='prvc'].patient_id.unique())
        n_vc_async = len(self.proc_results[(self.proc_results.ventmode=='vc') & async_mask])
        n_pc_async = len(self.proc_results[(self.proc_results.ventmode=='pc') & async_mask])
        n_prvc_async = len(self.proc_results[(self.proc_results.ventmode=='prvc') & async_mask])
        total_async = len(self.proc_results[async_mask])
        vals = [
            # general stats
            n_patients,
            'TODO',  # need sex data. can get this later tho
            'TODO',  # another thing that we dont need now
            'TODO',  # get this from cohort.csv
            'TODO',  # will need to get this from EHR. But probably not necessary yet.
            # ventmode n
            vc_pts,
            pc_pts,
            prvc_pts,
            # general breath counts
            len(self.proc_results),
            len(self.proc_results[self.proc_results.ventmode=='vc']),
            len(self.proc_results[self.proc_results.ventmode=='pc']),
            len(self.proc_results[self.proc_results.ventmode=='prvc']),
            len(self.proc_results)/n_patients,
            len(self.proc_results[self.proc_results.ventmode=='vc']) / vc_pts if vc_pts != 0 else np.nan,
            len(self.proc_results[self.proc_results.ventmode=='pc']) / pc_pts if pc_pts != 0 else np.nan,
            len(self.proc_results[self.proc_results.ventmode=='prvc']) / prvc_pts if prvc_pts != 0 else np.nan,
            # async breath counts
            n_vc_async,
            n_pc_async,
            n_prvc_async,
            round(n_vc_async / vc_pts, 2) if vc_pts != 0 else np.nan,
            round(n_pc_async / pc_pts, 2) if pc_pts != 0 else np.nan,
            n_prvc_async / prvc_pts if prvc_pts != 0 else np.nan,
            # deeper dive into asynchronies
            len(self.proc_results[self.proc_results.dta > 0]),
            len(self.proc_results[self.proc_results.bsa > 0]),
            len(self.proc_results[self.proc_results.fa > 0]),
            len(self.proc_results[(self.proc_results.dyn_dca > 0) | (self.proc_results.static_dca > 0)]),
            len(self.proc_results[(self.proc_results.static_dca > 0)]),
            len(self.proc_results[(self.proc_results.dyn_dca > 0)]),
            # proportions
            round(len(self.proc_results[self.proc_results.dta > 0])/total_async, 2),
            round(len(self.proc_results[self.proc_results.bsa > 0])/total_async, 2),
            round(len(self.proc_results[self.proc_results.fa > 0])/total_async, 2),
            round(len(self.proc_results[(self.proc_results.dyn_dca > 0) | (self.proc_results.static_dca > 0)])/total_async, 2),
        ]
        table = PrettyTable()
        table.field_names = ['stat', 'val']
        for stat, val in zip(data, vals):
            table.add_row([stat, val])
        prefix = 'CVC' if is_cvc else 'Patient'
        display(HTML('<h2>{}</h2>'.format(prefix + " Data Descriptive Statistics")))
        display(HTML(table.get_html_string()))

    def get_masks(self):
        return {
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
            'pc_only': (self.proc_results.ventmode == 'pc'),
            'pc_prvc': self.proc_results.ventmode.isin(['pc', 'prvc']),
            'prvc_only': (self.proc_results.ventmode == 'prvc'),
            'vc_only': (self.proc_results.ventmode == 'vc'),
        }

    def preprocess_mad_std_in_df(self, df, windowing):
        # this function is really not necessary. But its here, and it does save on
        # some complexity of computing things in the graphing function itself, like
        # performing some nan filtering and doing dataset slicing
        algos = sorted(list(df.algo.unique()))
        mad_std = {algo: [[], [], None, None] for algo in algos}
        mad_col, std_col = self._get_windowing_colnames(windowing)

        for algo in algos:
            nan_mask = df[df.algo==algo][mad_col].isna()
            if nan_mask.values.all():
                del mad_std[algo]
                continue
            non_nan = df[df.algo==algo][~nan_mask]
            mad_std[algo][0] = non_nan[mad_col]
            mad_std[algo][1] = non_nan[std_col]
            mad_std[algo][2] = np.mean(mad_std[algo][0])
            mad_std[algo][3] = np.mean(mad_std[algo][1])

        return mad_std, sorted(list(mad_std.keys()))

    def plot_algo_scatter(self, df, windowing, plt_title, figname, individual_patients, std_lim):
        """
        Perform scatterplot for all available algos based on an input per-patient dataframe.

        X-axis is MAD and Y-axis is std.
        """
        mad_std, algos_in_order = self.preprocess_mad_std_in_df(df, windowing)
        return self._mad_std_scatter(mad_std, windowing, plt_title, figname, algos_in_order, individual_patients, std_lim)

    def plot_algo_mad_std_boxplots(self, df, algo_ordering, windowing, figname_prefix):
        fig, ax = plt.subplots(figsize=(3*8, 3*3))
        # you can use bootstrap too if you want, but for now I'm not going to
        mad_col, std_col = self._get_windowing_colnames(windowing)

        sns.boxplot(x='algo', y=mad_col, data=df, order=algo_ordering, notch=True, showfliers=False)
        ax.set_ylabel('MAD (Median Absolute Deviation) of (Compliance - Algo)', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.set_ylim(-.4, 26)
        fig.savefig(self.results_dir.joinpath('{}_mad_windowing_{}_boxplot_result.png'.format(windowing, figname_prefix)).resolve(), dpi=self.dpi)

        fig, ax = plt.subplots(figsize=(3*8, 3*3))
        sns.boxplot(x='algo', y=std_col, data=df, order=algo_ordering, notch=True, showfliers=False)
        ax.set_ylabel('Standard Deviation ($\sigma$) of Algo', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.set_ylim(-.4, 31)
        fig.savefig(self.results_dir.joinpath('{}_std_windowing_{}_boxplot_result.png'.format(windowing, figname_prefix)).resolve(), dpi=self.dpi)

    def show_individual_breath_by_breath_frame_results(self, df, figname):
        fig, ax = plt.subplots(figsize=(3*8, 3*3))
        # XXX add WMD option
        algos_in_frame = set(df.columns).intersection(self.algos_used)
        diff_cols = ["{}_diff".format(algo) for algo in algos_in_frame]
        sorted_diff_cols = sorted(diff_cols)
        medians, iqr = self.extract_medians_and_iqr(df[sorted_diff_cols])
        stds = df[sorted_diff_cols].std().values.round(5)

        # alphabetical order again
        algos_in_order = sorted([algo.replace('_diff', '') for algo in sorted_diff_cols])
        self._draw_seaborn_boxplot_with_bootstrap(df[sorted_diff_cols], ax, medians, notch=False, bootstrap=self.boot_resamples)
        xtick_names = plt.setp(ax, xticklabels=algos_in_order)
        plt.setp(xtick_names, rotation=60, fontsize=14)
        xlim = ax.get_xlim()
        ax.plot(xlim, [0, 0], ls='--', zorder=0, c='red')
        ax.set_ylabel('Difference between Compliance and Algo', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        # want to keep a constant y perspective to compare algos
        #ax.set_ylim(-0.025, 0.025)
        title = figname.replace('.png', '').replace('_', ' ')
        ax.set_title(title, fontsize=20)
        fig.savefig(self.results_dir.joinpath(figname).resolve(), dpi=self.dpi)

        # show table of boxplot results
        self._show_breath_by_breath_algo_table(algos_in_order, medians, iqr, stds, title)
        plt.show(fig)

    def perform_algo_based_multi_window_analysis(self, absolute=True, windows=[5, 10, 20, 50, 100, 200, 400, 800], winsorizor=(0, 0.05), algos=[]):
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
                    if ('fai' in col and algo in FileCalculations.algos_unavailable_for_vc) or \
                        ('dci' in col and algo in FileCalculations.algos_unavailable_for_pc_prvc):
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

            plt.suptitle('Window Size {}'.format(size), fontsize=28, y=.9)
            plt.show(fig)

    def perform_multi_window_analysis(self, absolute=True, windows=[5, 10, 20, 50, 100, 200, 400, 800], winsorizor=(0, 0.05), robust=False):
        """
        Show insights from analyzing multiple different window sizes for different
        algorithms.

        :param absolute: return absolute values for WMD calcs
        :param windows: list of window sizes to use
        ;param winsorizor: (<low>, 1-<high>) percentiles to choose
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
                    self._regplot_wmd(self.proc_results, algo, win_col, size, axes[i][j], winsorizor, {'s': 0, 'alpha': .0}, {'label': size, 'lw': 3}, absolute, robust, False, False)

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

    def plot_breath_by_breath_results(self, only_patient=None, exclude_cols=[]):
        only_patient_wrapper = lambda df, pt: df[df.patient == pt] if pt is not None else df
        exclude_algos_wrapper = lambda df, cols: df.drop(cols, axis=1) if exclude_cols else df
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.proc_results, only_patient), exclude_cols),
            'breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['no_async'][self.window_n], only_patient), exclude_cols),
            'no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['vc_only'][self.window_n], only_patient), exclude_cols),
            'vc_only_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['vc_no_async'][self.window_n], only_patient), exclude_cols),
            'vc_no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['vc_only_async'][self.window_n], only_patient), exclude_cols),
            'vc_only_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['pc_only'][self.window_n], only_patient), exclude_cols),
            'pc_only_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['pc_no_async'][self.window_n], only_patient), exclude_cols),
            'pc_no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['pc_only_async'][self.window_n], only_patient), exclude_cols),
            'pc_only_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['prvc_only'][self.window_n], only_patient), exclude_cols),
            'prvc_only_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['prvc_no_async'][self.window_n], only_patient), exclude_cols),
            'prvc_no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['prvc_only_async'][self.window_n], only_patient), exclude_cols),
            'prvc_only_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['no_efforting'][self.window_n], only_patient), exclude_cols),
            'no_efforting_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['early_efforting'][self.window_n], only_patient), exclude_cols),
            'early_efforting_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['insp_efforting'][self.window_n], only_patient), exclude_cols),
            'insp_efforting_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['exp_efforting'][self.window_n], only_patient), exclude_cols),
            'exp_efforting_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['all_efforting'][self.window_n], only_patient), exclude_cols),
            'all_efforting_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['all_pressure_only'][self.window_n], only_patient), exclude_cols),
            'pc_prvc_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['all_pressure_no_async'][self.window_n], only_patient), exclude_cols),
            'pc_prvc_no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_frames['all_pressure_only_async'][self.window_n], only_patient), exclude_cols),
            'pc_prvc_only_async_breath_by_breath_results.png'
        )

    def plot_per_patient_results(self, windowing, individual_patients=False, show_boxplots=True, std_lim=None):
        """
        Plot patient by patient results

        :param windowing: None for no windowing, 'wmd' for WMD, and 'smd' for SMD
        :param individual_patients: show individual_patients scatter points
        :param show_boxplots: show boxplots after scatter plots
        :param std_lim: limit graphs by standard deviation within certain
                        factor. Normally is set to None (no limit). But can be
                        set to any floating value > 0.
        """
        # Patient by Patient. All breathing
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['all'][self.window_n],
            windowing,
            'Patient by patient results. No filters',
            'patient_by_patient_result.png',
            individual_patients,
            std_lim
        )
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_frames['all'][self.window_n], algos_in_order, windowing, 'patient_by_patient')

        # Patient by patient. no asynchronies
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['no_async'][self.window_n],
            windowing,
            'Patient by patient results. No Asynchronies',
            'patient_by_patient_no_async_result.png',
            individual_patients,
            std_lim
        )
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_frames['no_async'][self.window_n], algos_in_order, windowing, 'patient_by_patient_no_async')

        # Asynchronies only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['async'][self.window_n],
            windowing,
            'Patient by patient results. Asynchronies only',
            'patient_by_patient_asynchronies_only.png',
            individual_patients,
            std_lim
        )
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_frames['async'][self.window_n], algos_in_order, windowing, 'asynchronies_only_pbp')

        # VC only patient by patient.
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['vc_only'][self.window_n],
            windowing,
            'Patient by patient results. VC only',
            'patient_by_patient_vc_only.png',
            individual_patients,
            std_lim
        )
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_frames['vc_only'][self.window_n], algos_in_order, windowing, 'vc_only_pbp')

        # VC only, non-asynchronies
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['vc_no_async'][self.window_n],
            windowing,
            'Patient by patient results. VC, No Asynchronies',
            'patient_by_patient_vc_no_asynchronies.png',
            individual_patients,
            std_lim
        )

        # VC only, asynchronies only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['vc_only_async'][self.window_n],
            windowing,
            'Patient by patient results. VC Asynchronies only',
            'patient_by_patient_vc_asynchronies_only.png',
            individual_patients,
            std_lim
        )

        # PC only patient by patient.
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['pc_only'][self.window_n],
            windowing,
            'Patient by patient results. PC only',
            'patient_by_patient_pc_only.png',
            individual_patients,
            std_lim
        )
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_frames['pc_only'][self.window_n], algos_in_order, windowing, 'pc_only_pbp')

        # PC only, non-asynchronies
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['pc_no_async'][self.window_n],
            windowing,
            'Patient by patient results. PC, No Asynchronies',
            'patient_by_patient_pc_no_asynchronies.png',
            individual_patients,
            std_lim
        )

        # PC only, asynchronies only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['pc_only_async'][self.window_n],
            windowing,
            'Patient by patient results. PC Asynchronies only',
            'patient_by_patient_pc_asynchronies_only.png',
            individual_patients,
            std_lim
        )

        # PRVC only patient by patient.
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['prvc_only'][self.window_n],
            windowing,
            'Patient by patient results. PRVC only',
            'patient_by_patient_prvc_only.png',
            individual_patients,
            std_lim
        )
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_frames['prvc_only'][self.window_n], algos_in_order, windowing, 'prvc_only_pbp')

        # PRVC only, non-asynchronies
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['prvc_no_async'][self.window_n],
            windowing,
            'Patient by patient results. PRVC, No Asynchronies',
            'patient_by_patient_prvc_no_asynchronies.png',
            individual_patients,
            std_lim
        )

        # PRVC only, asynchronies only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['prvc_only_async'][self.window_n],
            windowing,
            'Patient by patient results. PRVC Asynchronies only',
            'patient_by_patient_prvc_asynchronies_only.png',
            individual_patients,
            std_lim
        )

        # no efforting only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['no_efforting'][self.window_n],
            windowing,
            'Patient by patient results. No Apparent Efforting',
            'patient_by_patient_no_efforting.png',
            individual_patients,
            std_lim
        )

        # early efforting only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['early_efforting'][self.window_n],
            windowing,
            'Patient by patient results. Early Efforting',
            'patient_by_patient_early_efforting.png',
            individual_patients,
            std_lim
        )

        # insp efforting only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['insp_efforting'][self.window_n],
            windowing,
            'Patient by patient results. Inspiratory Efforting',
            'patient_by_patient_insp_efforting.png',
            individual_patients,
            std_lim
        )

        # exp efforting only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['exp_efforting'][self.window_n],
            windowing,
            'Patient by patient results. Expiratory Efforting',
            'patient_by_patient_exp_efforting.png',
            individual_patients,
            std_lim
        )

        # all efforting only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['all_efforting'][self.window_n],
            windowing,
            'Patient by patient results. All Efforting',
            'patient_by_patient_all_efforting.png',
            individual_patients,
            std_lim
        )

        # PC/PRVC only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['all_pressure_only'][self.window_n],
            windowing,
            'Patient by patient results. PC/PRVC only',
            'patient_by_patient_pc_prvc_only.png',
            individual_patients,
            std_lim
        )

        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_frames['all_pressure_only'][self.window_n], algos_in_order, windowing, 'pressure_only_pbp')

        # Artifacts only
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['artifacts'][self.window_n],
            windowing,
            'Patient by patient results. Artifacts only',
            'patient_by_patient_artifacts_only.png',
            individual_patients,
            std_lim
        )

        # group asynchronies/artifacts by mode
        mad_std, algos_in_order = self.plot_algo_scatter(
            self.pp_frames['all_pressure_only_async'][self.window_n],
            windowing,
            'Patient by patient results. Pressure Mode Asynchronies only',
            'patient_by_patient_pressure_mode_asynchronies_only.png',
            individual_patients,
            std_lim
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
                fig.savefig(self.results_dir.joinpath('{}_breath_by_breath_patient_result.png'.format(algo)).resolve(), dpi=self.dpi)

    def save_results(self):
        if not self.results_dir.parent.exists():
            self.results_dir.parent.mkdir()
        if not self.results_dir.exists():
            self.results_dir.mkdir()
        pd.to_pickle(self, str(self.results_dir.joinpath('ResultsContainer.pkl')))

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

    def visualize_patient(self, patient, algos, extra_mask=None, ts_xlim=None, ts_ylim=None):
        if algos == 'all':
            algos = self.algos_used
        algo_cols = algos
        diff_cols = ['{}_diff'.format(c) for c in algo_cols]
        wmd_cols = ['{}_wmd_{}'.format(c, self.window_n) for c in algo_cols]
        smd_cols = ['{}_smd_{}'.format(c, self.window_n) for c in algo_cols]
        final_cols = algo_cols + diff_cols + wmd_cols + smd_cols + ['rel_bn', 'gold_stnd_compliance', 'patient_id']

        if not isinstance(extra_mask, type(None)):
            patient_df = self.proc_results[(self.proc_results.patient == patient) & extra_mask][final_cols]
        else:
            patient_df = self.proc_results[self.proc_results.patient == patient][final_cols]

        patient_df[algo_cols].plot(figsize=(3*8, 4*3), colormap=cc.cm.glasbey, fontsize=16)
        plt.legend(fontsize=16)
        plt.title(patient, fontsize=20)
        if ts_xlim is not None:
            plt.xlim(ts_xlim)
        if ts_ylim is not None:
            plt.ylim(ts_ylim)
        plt.plot(patient_df.gold_stnd_compliance, label='gt')
        plt.xlabel('DataFrame index')
        plt.show()

        pp_custom = self.analyze_per_patient_df(patient_df)
        mad_std, algos_in_order = self.plot_algo_scatter(
            pp_custom,
            None,
            'Patient {}. Custom plot'.format(patient),
            '{}_custom_plot.png'.format(patient),
            False,
            None,
        )

        # breath by breath results
        self.show_individual_breath_by_breath_frame_results(
            patient_df, 'pc_prvc_only_async_breath_by_breath_results.png'
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

    args = parser.parse_args()

    all_patient_dirs = Path(args.data_path).glob('*')
    algo = 'predator'
    baseline = 'insp_least_squares'
    results = ResultsContainer(args.experiment_name, 20)

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
            calcs = FileCalculations(patient_id, str(file), args.algos, 9, extra, tc_algos=args.tc_algos, lourens_tc_choice=args.lourens_tc_choice, recorded_compliance=recorded_compliance)
            try:
                calcs.analyze_file()
            except Exception as err:
                print('Failed on file: {}'.format(str(file)))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)
                print(err)
                return
            results.add_results_df(patient_id, calcs.results)
    results.collate_data(calcs.algos_used)
    results.save_results()


if __name__ == '__main__':
    main()
