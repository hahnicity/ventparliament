"""
main
~~~~
"""
import argparse
from pathlib import Path
import sys
import traceback
from warnings import warn

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import median_absolute_deviation
import seaborn as sns

from parliament.analyze import FileCalculations


class ResultsContainer(object):

    def __init__(self, experiment_name):
        self.proc_results = []
        self.algos_used = []
        self.experiment_name = experiment_name
        self.raw_results = []
        self.results_dir = Path(__file__).parent.joinpath('results', experiment_name)

    @classmethod
    def load_from_proc_results(cls, experiment_name):
        """
        Load a results container object if we already have saved the results dataframe
        """
        # this column matching could change in the future
        res = ResultsContainer(experiment_name)
        res.proc_results = pd.read_pickle(res.results_dir.joinpath('proc_results.pkl').resolve())
        res.algos_used = res.proc_results.columns[3:-3]
        return res

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

        # all breath results
        # XXX

        # per patient results
        all_patient_results = []
        pt_cols = ['patient_id', 'algo', 'mad', 'std']
        for patient_id, df in self.proc_results.groupby('patient_id'):
            # find MAD per patient, per algo
            for algo in self.algos_used:
                row = [patient_id, algo]
                df['{}_diff'.format(algo)] = df['gold_stnd_compliance'] - df[algo]
                self.proc_results.loc[df.index, '{}_diff'.format(algo)] = df['{}_diff'.format(algo)]
                row.append(median_absolute_deviation(df['{}_diff'.format(algo)], nan_policy='omit'))
                row.append(df[algo].std())
                all_patient_results.append(row)

        self.all_patient_results = pd.DataFrame(all_patient_results, columns=pt_cols)

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
            # results to get a statistically valid plateau pressure
            for i, row in frame.iterrows():
                plats_in_range = valid_plats[(valid_plats.abs_bs - pd.Timedelta(hours=0.5) < row.abs_bs) & (row.abs_bs < valid_plats.abs_bs + pd.Timedelta(hours=0.5))]
                proc_results.loc[i, 'gold_stnd_compliance'] = plats_in_range.gold_stnd_compliance.mean()
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
                # multiply the algo mad by 15 because we want to be absolutely sure we arent
                # removing any results that may be within range of the actual ground truth
                self.proc_results.loc[df[df[algo].abs() >= (algo_median + 15*algo_mad)].index, algo] = np.nan

    def plot_per_patient_results(self):
        """
        Plot patient by patient results
        """
        # Do scatter of MAD by std
        mad_std = {algo: [[], []] for algo in self.algos_used}
        for patient_id, df in self.all_patient_results.groupby('patient_id'):
            for algo in self.algos_used:
                row = df[df.algo == algo].iloc[0]
                # mad on x axis, std on y
                mad_std[algo][0].append(row['mad'])
                mad_std[algo][1].append(row['std'])

        algo_dists = []
        for algo in self.algos_used:
            mean_mad = np.nanmean(mad_std[algo][0])
            mean_std = np.nanmean(mad_std[algo][1])
            algo_dists.append([np.sqrt(mean_mad**2+mean_std**2), mean_mad, mean_std])
        # sort items so we can give them higher z order based on precedence
        algo_ordering = np.argsort(np.array(algo_dists)[:,0])
        inverse_ordering = list(np.argsort(algo_ordering))
        algos_in_order = np.array(self.algos_used)[algo_ordering]

        # XXX I want to have custom math text for my markers. But that means I also need
        # to individually find markers. So gonna punt on this for now
        markers = {
            'predator': 'P'
        }
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'P', 'X', 'D', 'd', 'H', 4]
        fig, ax = plt.subplots(figsize=(3*6.5, 3*2.5))

        for i, algo in enumerate(algos_in_order):
            color = cc.cm.glasbey(i)
            ax.scatter(x=mad_std[algo][0], y=mad_std[algo][1], marker=markers[i], color=color, label=algo, alpha=0.4, s=100, zorder=1)

        for i, algo in enumerate(algos_in_order):
            color = cc.cm.glasbey(i)
            orig_loc = inverse_ordering.index(i)
            ax.scatter(
                x=algo_dists[orig_loc][1],
                y=algo_dists[orig_loc][2],
                marker=markers[i],
                color=color,
                alpha=.9,
                s=350,
                edgecolors='black',
                zorder=len(self.algos_used)+2-i,
                linewidths=1,
            )
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylabel('Standard Deviation ($\sigma$) of Algo', fontsize=16)
        ax.set_xlabel('MAD (Median Absolute Deviation) of (Compliance - Algo)', fontsize=16)
        fig.legend(fontsize=16, loc='center right')
        ax.grid()
        ax.set_title('Patient by patient results. No filters')
        fig.savefig(self.results_dir.joinpath('patient_by_patient_result.png').resolve(), dpi=600)
        fig.show()

        fig, ax = plt.subplots(figsize=(3*8, 3*2))
        sns.boxplot(x='algo', y='mad', data=self.all_patient_results, order=algos_in_order)
        ax.set_ylabel('MAD (Median Absolute Deviation) of (Compliance - Algo)', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        fig.savefig(self.results_dir.joinpath('patient_by_patient_mad_boxplot_result.png').resolve(), dpi=600)
        fig.show()

        fig, ax = plt.subplots(figsize=(3*8, 3*2))
        sns.boxplot(x='algo', y='std', data=self.all_patient_results, order=algos_in_order)
        ax.set_ylabel('Standard Deviation ($\sigma$) of Algo', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        fig.savefig(self.results_dir.joinpath('patient_by_patient_std_boxplot_result.png').resolve(), dpi=600)
        fig.show()
        # XXX run analysis by ventmode


        # XXX next plot based on asynchronous data

    def save_results(self):
        if not self.results_dir.parent.exists():
            self.results_dir.parent.mkdir()
        if not self.results_dir.exists():
            self.results_dir.mkdir()
        pd.to_pickle(self, str(self.results_dir.joinpath('ResultsContainer.pkl')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('--only-patient', help='only run results for specific patient', nargs='*')
    parser.add_argument('--algos', nargs="*", default='all')
    parser.add_argument('--tc-algo', choices=['al_rawas', 'lourens', 'brunner'], default='al_rawas')
    parser.add_argument('-ltc', '--lourens-tc-choice', type=int, default=75)
    parser.add_argument('-dp', '--data-path', default=str(Path(__file__).parent.joinpath('../dataset/processed_data')))

    args = parser.parse_args()

    all_patient_dirs = Path(args.data_path).glob('*')
    algo = 'predator'
    baseline = 'insp_least_squares'
    results = ResultsContainer(args.experiment_name)

    for dir_ in sorted(list(all_patient_dirs)):
        patient_id = dir_.name
        for file in dir_.glob('*.raw.npy'):
            patient_id = file.parent.name
            if args.only_patient and patient_id not in args.only_patient:
                continue
            extra = pd.read_pickle(str(file).replace('raw.npy', 'extra.pkl'))
            calcs = FileCalculations(str(file), args.algos, 9, extra, tc_algo=args.tc_algo, lourens_tc_choice=args.lourens_tc_choice)
            try:
                calcs.analyze_file()
            except Exception as err:
                print('Failed on file: {}'.format(str(file)))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback)
                print(err)
                return
            results.add_results_df(patient_id, calcs.results)
    algos_used = list(set(calcs.results.columns).intersection(set(list(calcs.algo_mapping.keys()))))
    results.collate_data(algos_used)
    results.save_results()

#    # XXX debug
    import matplotlib.pyplot as plt
    patient_results = results.proc_results
    for pt, df in results.proc_results.groupby('patient_id'):
        df[algos_used].plot(title=pt, figsize=(3*8, 4*3), fontsize=6, colormap=cc.cm.glasbey)
        gt = patient_results['gold_stnd_compliance']
        plt.plot(gt, label='gt')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
