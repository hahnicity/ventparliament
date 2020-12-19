"""
main
~~~~
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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

    def collate_data(self):
        """
        Now that (presumably) all patient results have been tabulated we can finally determine
        what our gold standard compliances are for specific time points.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default=str(Path(__file__).parent.joinpath('../dataset/processed_data')))
    args = parser.parse_args()

    all_patient_dirs = Path(args.data_path).glob('*')
    algo = 'predator'
    baseline = 'insp_least_squares'
    results = ResultsContainer()

    for dir_ in sorted(list(all_patient_dirs)):
        patient_id = dir_.name
        for file in dir_.glob('*.raw.npy'):
            # XXX debug just keep this line around in case a patient is failing for now
            if '0210RPI' not in str(file):
                continue
            # XXX debug
            extra = pd.read_pickle(str(file).replace('raw.npy', 'extra.pkl'))
            calcs = FileCalculations(str(file), 'all', 9, extra)
            calcs.analyze_file()
            results.add_results_df(patient_id, calcs.results)
    results.collate_data()
    algos_used = set(results.all_results.columns).intersection(set(list(calcs.algo_mapping.keys())))

#    # XXX debug
    #import IPython; IPython.embed()
    import matplotlib.pyplot as plt
    patient_results = results.all_results
    for algo in algos_used:
        preds = patient_results[algo]
        plt.plot(preds, label=algo)
    gt = patient_results['gold_stnd_compliance']
    plt.plot(gt, label='gt')
    plt.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
