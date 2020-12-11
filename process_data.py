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
    def __init__(self, cohort_filepath, raw_data_dir, processed_data_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.cohort = pd.read_csv(cohort_filepath)

    def iterate_on_pt(self, row, min_time, flow_bound):
        # alright the additional information i want to include is:
        #
        # rel_bn: relative breath num
        # is_valid_plat: true/false
        # is_30_min_after_valid_plat: true/false
        # is_1hr_after_valid_plat: true/false
        for f in sorted(list(self.raw_data_dir.joinpath(row['patient_id']).glob('*.csv'))):
            should_save = False
            br_to_save = []
            extra_br_metadata = []
            # just start at the beginning of time for sanity purposes and saving extra if/then blocks
            last_valid_plat = datetime.fromtimestamp(0)
            with open(str(f), encoding='ascii', errors='ignore') as desc:
                # only want to pickle data that is a valid plat or up to an hour
                # over a valid plat. Everything after that we can forget about. This
                # will save us logic in future analysis steps
                gen = extract_raw(desc, False)
                for br in gen:
                    is_plat = check_if_plat_occurs(
                        br['flow'],
                        br['pressure'],
                        br['dt'],
                        min_time=min_time,
                        flow_bound=flow_bound
                    )
                    dt = pd.to_datetime(br['abs_bs'], format='%Y-%m-%d %H-%M-%S.%f')
                    # here we have a hierarchy of plats. plats in the QI cohort will
                    # always be trusted, and plats outside the QI cohort must've had an
                    # additional clinician verification of authenticity.
                    if is_plat and row['is_qi_cohort_plat'] == 'n':
                        is_valid_plat = (row['validated_rel_bn'] == br['rel_bn'])
                    elif is_plat and row['is_qi_cohort_plat'] == 'y':
                        is_valid_plat = True
                    else:
                        is_valid_plat = False

                    # determine whether we should change state of whether to save current breath
                    # or not
                    if is_valid_plat:
                        should_save = True
                        last_valid_plat = dt
                    elif not is_valid_plat and dt - last_valid_plat >= pd.Timedelta(hours=1):
                        should_save = False

                    if should_save:
                        br_to_save.append(br['rel_bn'])
                        extra_br_metadata.append([
                            br['rel_bn'],
                            is_valid_plat,
                            dt - last_valid_plat < pd.Timedelta(hours=0.5),
                            dt - last_valid_plat < pd.Timedelta(hours=1),
                        ])

                desc.seek(0)
                output_fname = self.processed_data_dir.joinpath(row['patient_id'], f.name.replace('.csv', ''))
                if not output_fname.parent.exists():
                    output_fname.parent.mkdir()

                if len(br_to_save) != 0:
                    extra_output_fname = self.processed_data_dir.joinpath(row['patient_id'], f.name.replace('.csv', '.extra.npy'))
                    process_breath_file(desc, False, str(output_fname), spec_rel_bns=br_to_save)
                    np.save(str(extra_output_fname), extra_br_metadata)

    def iter_raw_dir(self, min_time, flow_bound):
        for i, row in self.cohort.iterrows():
            self.iterate_on_pt(row, min_time, flow_bound)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cohort', default='cohort.csv')
    parser.add_argument('-rdp', '--raw-dataset-path', default='dataset/raw_data')
    parser.add_argument('-pdp', '--processed-dataset-path', default='dataset/processed_data')
    parser.add_argument('--flow-bound', type=float, default=0.2)
    parser.add_argument('--min-time', type=float, default=0.5)
    args = parser.parse_args()

    proc = Processing(args.cohort, args.raw_dataset_path, args.processed_dataset_path)
    proc.iter_raw_dir(args.min_time, args.flow_bound)


if __name__ == '__main__':
    main()
