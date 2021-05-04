"""
main
~~~~
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
from seaborn.categorical import _BoxPlotter
from seaborn.utils import remove_na
import seaborn as sns

from parliament.analyze import FileCalculations


class ResultsContainer(object):

    def __init__(self, experiment_name, wmd_n):
        self.proc_results = []
        self.algos_used = []
        self.experiment_name = experiment_name
        self.raw_results = []
        self.results_dir = Path(__file__).parent.joinpath('results', experiment_name)
        # you can always crank the dpi up for paper time
        self.dpi = 200
        self.boot_resamples = 100
        self.wmd_n = wmd_n
        self.full_analysis_done = False

    def _draw_seaborn_boxplot_with_bootstrap(self, data, ax, medians, **kwargs):
        """
        Basically an exact replica of what happens in seaborn except for support of conf intervals
        and usermedians
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

            artist_dict = ax.boxplot(box_data,
                                     vert=vert,
                                     patch_artist=True,
                                     positions=[i],
                                     widths=plotter.width,
                                     usermedians=[medians[i]],
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

    @classmethod
    def load_from_experiment_name(cls, experiment_name):
        """
        Load results container but even easier using the container.pkl obj
        """
        results_dir = Path(__file__).parent.joinpath('results', experiment_name)
        return pd.read_pickle(results_dir.joinpath('ResultsContainer.pkl'))

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
            for algo in self.algos_used:
                row = [patient_id, algo]
                row.append(median_absolute_deviation(frame['{}_diff'.format(algo)], nan_policy='omit'))
                row.append(frame[algo].std())
                row.append(median_absolute_deviation(frame['{}_wmd'.format(algo)], nan_policy='omit'))
                row.append(frame['{}_wmd'.format(algo)].std())
                row_results.append(row)
        cols = ['patient_id', 'algo', 'mad_pt', 'std_pt', 'mad_wmd', 'std_wmd']
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

        for algo in self.algos_used:
            self.proc_results['{}_diff'.format(algo)] = self.proc_results['gold_stnd_compliance'] - self.proc_results[algo]
        self.calc_wmd(self.proc_results)

        async_mask = ((self.proc_results.dta != 0) | (self.proc_results.bsa != 0) | (self.proc_results.fa != 0) | (self.proc_results.static_dca != 0) | (self.proc_results.dyn_dca != 0))

        # analyze all per patient breaths
        self.pp_all = self.analyze_per_patient_df(self.proc_results)

        # analyze non-asynchronous breaths
        self.bb_no_async = self.proc_results[~async_mask]
        self.pp_no_async = self.analyze_per_patient_df(self.bb_no_async)

        # analyze all asynchronous breathing
        self.bb_async_results = self.proc_results[async_mask]
        self.pp_async_results = self.analyze_per_patient_df(self.bb_async_results)

        # analyze only volume control breaths
        self.bb_vc_only = self.proc_results[self.proc_results.ventmode == 'vc']
        self.pp_vc_only = self.analyze_per_patient_df(self.bb_vc_only)

        # analyze all artifact breathing
        self.bb_artifacts = self.proc_results[self.proc_results.artifact != 0]
        self.pp_artifacts = self.analyze_per_patient_df(self.bb_artifacts)

        # analyze non-asynchronous breathing VC only
        self.bb_vc_no_async = self.proc_results[(self.proc_results.ventmode == 'vc') & ~async_mask]
        self.pp_vc_no_async = self.analyze_per_patient_df(self.bb_vc_no_async)

        # analyze asynchronous breathing, VC only
        self.bb_vc_only_async = self.proc_results[(self.proc_results.ventmode == 'vc') & async_mask]
        self.pp_vc_only_async = self.analyze_per_patient_df(self.bb_vc_only_async)

        # analyze all PC only
        self.bb_pc_only = self.proc_results[(self.proc_results.ventmode == 'pc')]
        self.pp_pc_only = self.analyze_per_patient_df(self.bb_pc_only)

        # analyze non-asynchronous breathing PC only
        self.bb_pc_no_async = self.proc_results[(self.proc_results.ventmode == 'pc') & ~async_mask]
        self.pp_pc_no_async = self.analyze_per_patient_df(self.bb_pc_no_async)

        # analyze asynchronous breathing PC only
        self.bb_pc_only_async = self.proc_results[(self.proc_results.ventmode == 'pc') & async_mask]
        self.pp_pc_only_async = self.analyze_per_patient_df(self.bb_pc_only_async)

        # analyze all PRVC only
        self.bb_prvc_only = self.proc_results[(self.proc_results.ventmode == 'prvc')]
        self.pp_prvc_only = self.analyze_per_patient_df(self.bb_prvc_only)

        # analyze non-asynchronous breathing PRVC only
        self.bb_prvc_no_async = self.proc_results[(self.proc_results.ventmode == 'prvc') & ~async_mask]
        self.pp_prvc_no_async = self.analyze_per_patient_df(self.bb_prvc_no_async)

        # analyze asynchronous breathing PRVC only
        self.bb_prvc_only_async = self.proc_results[(self.proc_results.ventmode == 'prvc') & async_mask]
        self.pp_prvc_only_async = self.analyze_per_patient_df(self.bb_prvc_only_async)

        # analyze only pressure breathing
        self.bb_all_pressure_only = self.proc_results[self.proc_results.ventmode.isin(['pc', 'prvc'])]
        self.pp_all_pressure_only = self.analyze_per_patient_df(self.bb_all_pressure_only)

        # analyze non-asynchronous pressure related breathing. PC/PRVC
        self.bb_all_pressure_no_async = self.proc_results[(self.proc_results.ventmode.isin(['pc', 'prvc'])) & ~async_mask]
        self.pp_all_pressure_no_async = self.analyze_per_patient_df(self.bb_all_pressure_no_async)

        # analyze asynchronous pressure related breathing. PC/PRVC
        self.bb_all_pressure_only_async = self.proc_results[(self.proc_results.ventmode.isin(['pc', 'prvc'])) & async_mask]
        self.pp_all_pressure_only_async = self.analyze_per_patient_df(self.bb_all_pressure_only_async)

        # save a processed results container because this method takes the
        # longest time out of all the other methods to run
        self.full_analysis_done = True
        pd.to_pickle(self, self.results_dir.joinpath('ResultsContainer.pkl'))

    def calc_wmd(self, df):
        for patiend_id, pt_df in df.groupby('patient_id'):
            for algo in self.algos_used:
                df.loc[pt_df.index, '{}_wm'.format(algo)] = pt_df[algo].rolling(self.wmd_n, min_periods=1).apply(lambda x: np.nanmedian(x))

        for algo in self.algos_used:
            wm_colname = '{}_wm'.format(algo)
            wmd_colname = '{}_wmd'.format(algo)
            diff_colname = '{}_diff'.format(algo)
            df[wmd_colname] = df.gold_stnd_compliance - df[wm_colname]
            # make sure algo calcs are null if not available for specific mode
            if algo in FileCalculations.algos_unavailable_for_vc:
                df.loc[df.ventmode == 'vc', [wm_colname, wmd_colname, diff_colname]] = np.nan
            elif algo in FileCalculations.algos_unavailable_for_pc_prvc:
                df.loc[df.ventmode != 'vc', [wm_colname, wmd_colname, diff_colname]] = np.nan

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
                # I've found that mean can blow up in the presence of outliers. So instead use
                # the median
                algo_median = df[algo].median(skipna=True)
                algo_mad = median_absolute_deviation(df[algo].values, nan_policy='omit')
                # XXX will probably need to change this with wmd impl
                # XXX
                # multiply the algo mad by 15 because we want to be absolutely sure we arent
                # removing any results that may be within range of the actual ground truth
                self.proc_results.loc[df[df[algo].abs() >= (algo_median + 15*algo_mad)].index, algo] = np.nan

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
        data = [
            # general patient stats
            'n patients',
            '% Female',
            'Age (IQR)',
            'median RASS (IQR)',
            '% paralyzed',
            # ventmode n
            'n vc patients',
            'n pc patients',
            'n prvc patients',
            # general breath counts
            'total breaths',
            'total vc breaths',
            'total pc breaths',
            'total prvc breaths',
            'mean breaths per patient',
            'mean vc per patient',
            'mean pc per patient',
            'mean prvc per patient',
            # asynchronous breath counts
            'total vc async breaths',
            'total pc async breaths',
            'total prvc async breaths',
            'mean vc async per patient',
            'mean pc async per patient',
            'mean prvc async per patient',
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
        async_mask = ((self.proc_results.dta != 0) | (self.proc_results.bsa != 0) | (self.proc_results.fa != 0) | (self.proc_results.static_dca != 0) | (self.proc_results.dyn_dca != 0))
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
        display(HTML('<h2>{}</h2>'.format("Data Descriptive Statistics")))
        display(HTML(table.get_html_string()))

    def get_masks(self):
        return {
            'async': (
                (self.proc_results.dta != 0) |
                (self.proc_results.bsa != 0) |
                (self.proc_results.fa != 0) |
                (self.proc_results.static_dca != 0) |
                (self.proc_results.dyn_dca != 0)
            ),
            'vc_only': (self.proc_results.ventmode == 'vc'),
            'pc_only': (self.proc_results.ventmode == 'pc'),
            'prvc_only': (self.proc_results.ventmode == 'prvc'),
            'pc_prvc': self.proc_results.ventmode.isin(['pc', 'prvc']),
        }

    def preprocess_mad_std_in_df(self, df, use_wmd):
        # Do scatter of MAD by std
        mad_std = {algo: [[], [], None, None, None] for algo in self.algos_used}
        algo_dists = []
        if not use_wmd:
            mad_col = 'mad_pt'
            std_col = 'std_pt'
        else:
            mad_col = 'mad_wmd'
            std_col = 'std_wmd'

        for algo in self.algos_used:
            # mad on x axis, std on y
            mad_std[algo][0] = df[df.algo == algo][mad_col]
            if np.isnan(mad_std[algo][0]).all():
                del mad_std[algo]
                continue
            mad_std[algo][1] = df[df.algo == algo][std_col]
            mean_mad = np.nanmean(mad_std[algo][0])
            mean_std = np.nanmean(mad_std[algo][1])
            mad_std[algo][2] = mean_mad
            mad_std[algo][3] = mean_std
            # l1 distance to origin (0, 0). uses l1 because l2 places higher emphasis on std
            mad_std[algo][4] = mean_mad+mean_std
            algo_dists.append(mean_mad+mean_std)

        algos_used = [algo for algo in copy(self.algos_used) if algo in mad_std]
        # sort by alphabetical order
        algos_in_order = sorted(algos_used)
        return mad_std, algos_in_order

    def plot_algo_scatter(self, df, use_wmd, plt_title, figname, individual_patients, std_lim):
        """
        Perform scatterplot for all available algos based on an input per-patient dataframe.

        X-axis is MAD and Y-axis is std.
        """
        mad_std, algos_in_order = self.preprocess_mad_std_in_df(df, use_wmd)
        markers = cycle(['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'P', 'X', 'D', 'd', 'H', '$\Join$', '$\clubsuit$', '$\spadesuit$', '$\heartsuit$', '$\$$', '$\dag$', '$\ddag$', '$\P$'])
        colors = [cc.cm.glasbey(i) for i in range(len(self.algos_used))]
        algo_dict = {algo: {'m': next(markers), 'c': colors[i]} for i, algo in enumerate(self.algos_used)}
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
                zorder=len(self.algos_used)+2-i,
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
        fig.legend(fontsize=16, loc='center right')
        ax.grid()
        ax.set_title(plt_title, fontsize=20)
        fig.savefig(str(self.results_dir.joinpath(figname).resolve()).replace('.png', '-use-wmd-{}.png'.format(use_wmd)), dpi=self.dpi)
        return mad_std, algos_in_order

    def plot_algo_mad_std_boxplots(self, df, algo_ordering, use_wmd, figname_prefix):
        fig, ax = plt.subplots(figsize=(3*8, 3*2))
        # you can use bootstrap too if you want, but for now I'm not going to
        if not use_wmd:
            mad_col = 'mad_pt'
            std_col = 'std_pt'
        else:
            mad_col = 'mad_wmd'
            std_col = 'std_wmd'

        sns.boxplot(x='algo', y=mad_col, data=df, order=algo_ordering, notch=True, showfliers=False)
        ax.set_ylabel('MAD (Median Absolute Deviation) of (Compliance - Algo)', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.set_ylim(-.4, 26)
        fig.savefig(self.results_dir.joinpath('{}_mad_wmd_{}_boxplot_result.png'.format(use_wmd, figname_prefix)).resolve(), dpi=self.dpi)

        fig, ax = plt.subplots(figsize=(3*8, 3*2))
        sns.boxplot(x='algo', y=std_col, data=df, order=algo_ordering, notch=True, showfliers=False)
        ax.set_ylabel('Standard Deviation ($\sigma$) of Algo', fontsize=16)
        ax.set_xlabel('Algorithm', fontsize=16)
        ax.set_ylim(-.4, 31)
        fig.savefig(self.results_dir.joinpath('{}_std_wmd_{}_boxplot_result.png'.format(use_wmd, figname_prefix)).resolve(), dpi=self.dpi)

    def show_individual_breath_by_breath_frame_results(self, df, figname):
        fig, ax = plt.subplots(figsize=(3*8, 3*2))
        # XXX add WMD option
        diff_cols = ["{}_diff".format(algo) for algo in self.algos_used]
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
        table = PrettyTable()
        table.field_names = ['Algorithm', 'Median Diff', '25% IQR', '75% IQR', 'std']
        medians = medians.round(2)
        iqr = iqr.round(2)
        stds = stds.round(2)
        for i, algo in enumerate(algos_in_order):
            if not np.isnan(medians[i]):
                table.add_row([algo, medians[i], iqr[i, 0], iqr[i, 1], stds[i]])
            else:
                table.add_row([algo, '-', '-', '-', '-'])

        # XXX perform boldface formatting with best median/iqr
        #
        # How to do? There's got to be a method that's been done before.
        # Why don't you just do something for now to get the code down and
        # then refine your method later.
        soup = BeautifulSoup(table.get_html_string())
        min_median = np.nanargmin(abs(medians))
        min_iqr_rel_to_0 = np.nanargmin(abs(iqr).sum(axis=1))
        min_std = np.nanargmin(stds)
        # the +1 is because the header is embedded in a <tr> element
        min_med_elem = soup.find_all('tr')[min_median+1]
        min_iqr_elem = soup.find_all('tr')[min_iqr_rel_to_0+1]
        min_std_elem = soup.find_all('tr')[min_std+1]

        self._change_td_to_bold(soup, min_med_elem.find_all('td')[1])
        self._change_td_to_bold(soup, min_iqr_elem.find_all('td')[2])
        self._change_td_to_bold(soup, min_iqr_elem.find_all('td')[3])
        self._change_td_to_bold(soup, min_std_elem.find_all('td')[4])

        display(HTML('<h2>{}</h2>'.format(title)))
        display(HTML(soup.prettify()))
        plt.show(fig)

    def _change_td_to_bold(self, soup, td):
        val = td.string
        td.clear()
        td.insert(0, soup.new_tag('b'))
        td.b.string = val

    def plot_breath_by_breath_results(self, only_patient=None, exclude_cols=[]):
        only_patient_wrapper = lambda df, pt: df[df.patient == pt] if pt is not None else df
        # XXX need to check if this actually works.
        exclude_algos_wrapper = lambda df, cols: df.drop(cols, axis=1) if exclude_cols else df
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.proc_results, only_patient), exclude_cols),
            'breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_no_async, only_patient), exclude_cols),
            'no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_vc_only, only_patient), exclude_cols),
            'vc_only_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_vc_no_async, only_patient), exclude_cols),
            'vc_no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_vc_only_async, only_patient), exclude_cols),
            'vc_only_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_pc_only, only_patient), exclude_cols),
            'pc_only_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_pc_no_async, only_patient), exclude_cols),
            'pc_no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_pc_only_async, only_patient), exclude_cols),
            'pc_only_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_prvc_only, only_patient), exclude_cols),
            'prvc_only_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_prvc_no_async, only_patient), exclude_cols),
            'prvc_no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_prvc_only_async, only_patient), exclude_cols),
            'prvc_only_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_all_pressure_only, only_patient), exclude_cols),
            'pc_prvc_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_all_pressure_no_async, only_patient), exclude_cols),
            'pc_prvc_no_async_breath_by_breath_results.png'
        )
        self.show_individual_breath_by_breath_frame_results(
            exclude_algos_wrapper(only_patient_wrapper(self.bb_pressure_only_async, only_patient), exclude_cols),
            'pc_prvc_only_async_breath_by_breath_results.png'
        )

    def plot_per_patient_results(self, use_wmd, individual_patients=False, show_boxplots=True, std_lim=None):
        """
        Plot patient by patient results

        :param use_wmd: use wmd or no?
        :param individual_patients: show individual_patients scatter points
        :param show_boxplots: show boxplots after scatter plots
        :param std_lim: limit graphs by standard deviation within certain
                        factor. Normally is set to None (no limit). But can be
                        set to any floating value > 0.
        """
        # Patient by Patient. All breathing
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_all, use_wmd, 'Patient by patient results. No filters', 'patient_by_patient_result.png', individual_patients, std_lim)
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_all, algos_in_order, use_wmd, 'patient_by_patient')

        # Patient by patient. no asynchronies
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_no_async, use_wmd, 'Patient by patient results. No Asynchronies', 'patient_by_patient_no_async_result.png', individual_patients, std_lim)
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_no_async, algos_in_order, use_wmd, 'patient_by_patient_no_async')

        # Asynchronies only
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_async_results, use_wmd, 'Patient by patient results. Asynchronies only', 'patient_by_patient_asynchronies_only.png', individual_patients, std_lim)
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_async_results, algos_in_order, use_wmd, 'asynchronies_only_pbp')

        # VC only patient by patient.
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_vc_only, use_wmd, 'Patient by patient results. VC only', 'patient_by_patient_vc_only.png', individual_patients, std_lim)
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_vc_only, algos_in_order, use_wmd, 'vc_only_pbp')

        # VC only, non-asynchronies
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_vc_no_async, use_wmd, 'Patient by patient results. VC, No Asynchronies', 'patient_by_patient_vc_no_asynchronies.png', individual_patients, std_lim)

        # VC only, asynchronies only
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_vc_only_async, use_wmd, 'Patient by patient results. VC Asynchronies only', 'patient_by_patient_vc_asynchronies_only.png', individual_patients, std_lim)

        # PC only patient by patient.
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_pc_only, use_wmd, 'Patient by patient results. PC only', 'patient_by_patient_pc_only.png', individual_patients, std_lim)
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_pc_only, algos_in_order, use_wmd, 'pc_only_pbp')

        # PC only, non-asynchronies
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_pc_no_async, use_wmd, 'Patient by patient results. PC, No Asynchronies', 'patient_by_patient_pc_no_asynchronies.png', individual_patients, std_lim)

        # PC only, asynchronies only
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_pc_only_async, use_wmd, 'Patient by patient results. PC Asynchronies only', 'patient_by_patient_pc_asynchronies_only.png', individual_patients, std_lim)

        # PRVC only patient by patient.
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_prvc_only, use_wmd, 'Patient by patient results. PRVC only', 'patient_by_patient_prvc_only.png', individual_patients, std_lim)
        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_prvc_only, algos_in_order, use_wmd, 'prvc_only_pbp')

        # PRVC only, non-asynchronies
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_prvc_no_async, use_wmd, 'Patient by patient results. PRVC, No Asynchronies', 'patient_by_patient_prvc_no_asynchronies.png', individual_patients, std_lim)

        # PRVC only, asynchronies only
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_prvc_only_async, use_wmd, 'Patient by patient results. PRVC Asynchronies only', 'patient_by_patient_prvc_asynchronies_only.png', individual_patients, std_lim)

        # PC/PRVC only
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_all_pressure_only, use_wmd, 'Patient by patient results. PC/PRVC only', 'patient_by_patient_pc_prvc_only.png', individual_patients, std_lim)

        if show_boxplots:
            self.plot_algo_mad_std_boxplots(self.pp_pressure_only, algos_in_order, use_wmd, 'pressure_only_pbp')

        # Artifacts only
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_artifacts, use_wmd, 'Patient by patient results. Artifacts only', 'patient_by_patient_artifacts_only.png', individual_patients, std_lim)

        # group asynchronies/artifacts by mode
        mad_std, algos_in_order = self.plot_algo_scatter(self.pp_pressure_only_async, use_wmd, 'Patient by patient results. Pressure Mode Asynchronies only', 'patient_by_patient_pressure_mode_asynchronies_only.png', individual_patients, std_lim)

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

    def visualize_patient(self, patient, algos, extra_mask=None):
        columns = algos + ['gold_stnd_compliance']
        if not isinstance(extra_mask, type(None)):
            patient_df = self.proc_results[(self.proc_results.patient == patient) & extra_mask][columns]
        else:
            patient_df = self.proc_results[self.proc_results.patient == patient][columns]
        patient_df.plot(figsize=(3*8, 4*3), colormap=cc.cm.glasbey, fontsize=14)
        plt.title(patient, fontsize=20)
        plt.plot(patient_df.gold_stnd_compliance, label='gt')
        plt.show()
        plt.close()


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
