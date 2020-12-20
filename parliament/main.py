"""
main
~~~~
"""
import argparse
from pathlib import Path
import sys
import traceback

import colorcet as cc
import numpy as np
import pandas as pd
from scipy.stats import median_absolute_deviation

from parliament.analyze import FileCalculations


class ResultsContainer(object):
    def __init__(self):
        self.all_results = []

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
        self.all_results.append(dataframe)

    def analyze_by_patient(self, patient):
        """
        of our algorithms to the gold standard
        """
        pass

    def analyze_results(self):
        """
        Calculate Median Average Deviation (MAD) between gold standard and a calculation
        and the MAD within a calculation
        """


        # XXX this is all old code
        analysis = {val: [0, 0, 0, 0, [], []] for val in self.algorithms_to_use}
        for bn in self.results:
            gld = self.results[bn]['gold_stnd_compliance']
            if gld is np.nan:
                continue
            for algo in self.algorithms_to_use:
                val = self.results[bn][algo]
                if val is np.nan:
                    continue
                analysis[algo][2] += 1
                analysis[algo][-2].append(val)
                analysis[algo][-1].append(gld)

        for algo in self.algorithms_to_use:
            vals = np.array(analysis[algo][-2])
            glds = np.array(analysis[algo][-1])
            median_val = np.median(vals)
            inter_mad = np.median(np.abs(vals - median_val))
            gld_mad = np.median(np.abs(vals - glds))
            analysis[algo][0] = gld_mad
            analysis[algo][1] = inter_mad
            analysis[algo][3] = np.mean(glds)
            # delete intermediate data for algo values and gold calcs
            del analysis[algo][-1]
            del analysis[algo][-1]
        return analysis

    def collate_data(self, algos_used):
        """
        Now that (presumably) all patient results have been tabulated we can finally determine
        what our gold standard compliances are for specific time points. Filter
        """
        all_results = pd.concat(self.all_results)
        all_results.index = range(len(all_results))
        for patient, frame in all_results.groupby('patient_id'):
            valid_plats = frame[frame.is_valid_plat == True]
            # some rows will have multiple plats overlapping with them. we can average the plat
            # results to get a statistically valid plateau pressure
            for i, row in frame.iterrows():
                plats_in_range = valid_plats[(valid_plats.abs_bs - pd.Timedelta(hours=0.5) < row.abs_bs) & (row.abs_bs < valid_plats.abs_bs + pd.Timedelta(hours=0.5))]
                all_results.loc[i, 'gold_stnd_compliance'] = plats_in_range.gold_stnd_compliance.mean()
        self.all_results = all_results
        # filter outliers by patient
        for algo in algos_used:
            for patient_id, df in self.all_results.groupby('patient_id'):
                inf_idxs = df[(df[algo] == np.inf) | (df[algo] == -np.inf)].index
                df.loc[inf_idxs, algo] = np.nan
                self.all_results.loc[inf_idxs, algo] = np.nan
                # I've found that mean can blow up in the presence of outliers. So instead use
                # the median
                algo_median = df[algo].median(skipna=True)
                algo_mad = median_absolute_deviation(df[algo].values, nan_policy='omit')
                # multiply the algo mad by 20 because we want to be absolutely sure we arent
                # removing any results that may be within range of the actual ground truth
                self.all_results.loc[df[df[algo].abs() >= (algo_median + 20*algo_mad)].index, algo] = np.nan

    def save_results(self, experiment_name):
        results_dir = Path(__file__).parent.joinpath('results')
        if not results_dir.exists():
            results_dir.mkdir()
        experiment_dir = results_dir.joinpath(experiment_name)
        if not experiment_dir.exists():
            experiment_dir.mkdir()
        self.all_results.to_pickle(str(experiment_dir.joinpath('all_results.pkl')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    parser.add_argument('--only-patient', help='only run results for specific patient', nargs='*')
    parser.add_argument('-dp', '--data-path', default=str(Path(__file__).parent.joinpath('../dataset/processed_data')))
    args = parser.parse_args()

    all_patient_dirs = Path(args.data_path).glob('*')
    algo = 'predator'
    baseline = 'insp_least_squares'
    results = ResultsContainer()

    for dir_ in sorted(list(all_patient_dirs)):
        patient_id = dir_.name
        for file in dir_.glob('*.raw.npy'):
            patient_id = file.parent.name
            if args.only_patient and patient_id not in args.only_patient:
                continue
            extra = pd.read_pickle(str(file).replace('raw.npy', 'extra.pkl'))
            calcs = FileCalculations(str(file), 'all', 9, extra)
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
    results.save_results(args.experiment_name)

#    # XXX debug
    import matplotlib.pyplot as plt
    patient_results = results.all_results
    for pt, df in results.all_results.groupby('patient_id'):
        df[algos_used].plot(title=pt, figsize=(3*8, 4*3), fontsize=6, colormap=cc.cm.glasbey)
        gt = patient_results['gold_stnd_compliance']
        plt.plot(gt, label='gt')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
