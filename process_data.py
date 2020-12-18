"""
process_data
~~~~~~~~~~~~

Process data so it in a form usable by our algos
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ventmap.raw_utils import extract_raw, process_breath_file
from ventmap.SAM import check_if_plat_occurs


class Processing(object):
    def __init__(self, cohort_filepath, raw_data_dir, processed_data_dir, min_plat_time, flow_bound, any_or_all):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.cohort = pd.read_csv(cohort_filepath)
        # very generous error margin
        self.plat_wiggle_time = pd.Timedelta(hours=0.5)
        self.min_plat_time = min_plat_time
        self.flow_bound = flow_bound
        self.any_or_all = any_or_all

    def add_ventmode_metadata(self, extra_br_metadata, ventmode_file, raw_file):
        if not ventmode_file.exists():
            raise Exception('ventmode file for {} doesnt exist'.format(raw_file.name))
        vm = pd.read_csv(ventmode_file)
        study_supported_modes = ['vc', 'pc', 'prvc']
        saved_bns = extra_br_metadata[:, 0]
        mode_df = vm[vm.bn.isin(saved_bns)]
        if mode_df[study_supported_modes].sum().sum() != len(mode_df):
            raise Exception('file {} has ventmodes that are not supported by this study'.format(raw_file.name))
        mode_df['ventmode'] = ''
        for mode in study_supported_modes:
            mode_df.loc[mode_df[mode] == 1, 'ventmode'] = mode
        return np.append(extra_br_metadata, np.expand_dims(mode_df['ventmode'].values, axis=1), axis=1)

    def check_is_qi_cohort_plat(self, rows, plat_time):
        for i, row in rows.iterrows():
            approx_time = pd.to_datetime(row['approx_plat_time'])
            if row['is_qi_cohort_plat'] == 'y' and approx_time - self.plat_wiggle_time <= plat_time <= approx_time + self.plat_wiggle_time:
                return True
        else:
            return False

    def iterate_on_pt(self, rows):
        """
        Iterate over all transferred raw patient files, and only keep breaths within
        the time window of information that we desire. Ensure we preprocess and save
        all necessary data for future use in analysis.
        """
        # alright the additional information i want to include is:
        #
        # rel_bn: relative breath num
        # is_valid_plat: true/false
        plats = []
        patient_id = rows.iloc[0]['patient_id']
        for f in sorted(list(self.raw_data_dir.joinpath(patient_id).glob('*.csv'))):
            # just start at the beginning of time for sanity purposes and saving extra if/then blocks
            with open(str(f), encoding='ascii', errors='ignore') as desc:
                gen = extract_raw(desc, False)
                for br in gen:
                    is_plat = check_if_plat_occurs(
                        br['flow'],
                        br['pressure'],
                        br['dt'],
                        min_time=self.min_plat_time,
                        flow_bound=self.flow_bound,
                        flow_bound_any_or_all=self.any_or_all,
                    )
                    dt = pd.to_datetime(br['abs_bs'], format='%Y-%m-%d %H-%M-%S.%f')
                    # here we have a hierarchy of plats. plats in the QI cohort will
                    # always be trusted, and plats outside the QI cohort must've had an
                    # additional clinician verification of authenticity.
                    is_valid_plat = False
                    if is_plat and self.check_is_qi_cohort_plat(rows, dt):
                        plats.append((f, br['rel_bn'], dt))
                    elif is_plat and not self.check_is_qi_cohort_plat(rows, dt) and br['rel_bn'] in rows.validated_rel_bn.values:
                        plats.append((f, br['rel_bn'], dt))

        br_to_save = {f: [] for f in sorted(list(self.raw_data_dir.joinpath(patient_id).glob('*.csv')))}
        for f, bn, plat_time in plats:
            br_to_save[f].append((bn, True))

        for f in sorted(list(self.raw_data_dir.joinpath(patient_id).glob('*.csv'))):
            with open(str(f), encoding='ascii', errors='ignore') as desc:
                gen = extract_raw(desc, False)

                for br in gen:
                    dt = pd.to_datetime(br['abs_bs'], format='%Y-%m-%d %H-%M-%S.%f')
                    for _, __, plat_time in plats:

                        if (f, br['rel_bn'], dt) in plats:
                            continue
                        elif plat_time - pd.Timedelta(hours=0.5) < dt < plat_time + pd.Timedelta(hours=0.5):
                            br_to_save[f].append((br['rel_bn'], False))
                            break

        for f, vals in br_to_save.items():
            vals = sorted(vals, key=lambda x: x[0])
            if len(vals) == 0:
                continue

            with open(str(f), encoding='ascii', errors='ignore') as desc:
                extra_br_metadata = np.array(vals)
                ventmode_file = f.parent.joinpath('../../ventmodes/{}'.format(f.name.replace('.csv', '_1-ventmode-output.csv')))
                extra_br_metadata = self.add_ventmode_metadata(extra_br_metadata, ventmode_file, f)
                output_fname = self.processed_data_dir.joinpath(patient_id, f.name.replace('.csv', ''))
                if not output_fname.parent.exists():
                    output_fname.parent.mkdir()

                extra_output_fname = self.processed_data_dir.joinpath(patient_id, f.name.replace('.csv', '.extra.pkl'))
                process_breath_file(desc, False, str(output_fname), spec_rel_bns=extra_br_metadata[:, 0])
                pd.DataFrame(extra_br_metadata, columns=['rel_bn', 'is_valid_plat', 'ventmode']).to_pickle(str(extra_output_fname))

    def iter_raw_dir(self, only_patient):
        for patient_id, rows in self.cohort.groupby('patient_id'):
            if only_patient and only_patient == patient_id:
                self.iterate_on_pt(rows)
            elif not only_patient:
                self.iterate_on_pt(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cohort', default='cohort.csv')
    parser.add_argument('-rdp', '--raw-dataset-path', default='dataset/raw_data')
    parser.add_argument('-pdp', '--processed-dataset-path', default='dataset/processed_data')
    parser.add_argument('--flow-bound', type=float, default=0.2, help='flow bound for plat pressures')
    parser.add_argument('--min-plat-time', type=float, default=0.4, help='minimum amount of time a plat must occur for')
    parser.add_argument('--any-or-all', choices=['any', 'all'], default='any')
    parser.add_argument('--only-patient', help='only run specific patient')
    args = parser.parse_args()

    proc = Processing(args.cohort, args.raw_dataset_path, args.processed_dataset_path, args.min_plat_time, args.flow_bound, args.any_or_all)
    proc.iter_raw_dir(args.only_patient)


if __name__ == '__main__':
    main()
