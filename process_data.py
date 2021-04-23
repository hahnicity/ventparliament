"""
process_data
~~~~~~~~~~~~

Process data so it in a form usable by our algos
"""
import argparse
from datetime import datetime
from pathlib import Path

# this is from our private async detection repository. You are welcome to use other
# methods to detect asynchronies. There are a number of machine learning methods
# that are free and available for use. Either way though, you will have access to the
# asynchrony predictions in our dataset
from algorithms.dca import extract_new_feature
from algorithms.flow_asynchrony import get_gen_flow_async
from algorithms.tor5 import detectPVI
from fuzzy_clust_algos.gk import GK
import numpy as np
import pandas as pd
from ventmap.breath_meta import get_production_breath_meta
from ventmap.raw_utils import extract_raw, process_breath_file, read_processed_file
from ventmap.SAM import check_if_plat_occurs

from parliament.other_calcs import calc_volumes


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

    def add_ventmode_metadata(self, extra_br_metadata, ventmode_file, raw_file, is_cvc):
        if not ventmode_file.exists():
            raise Exception('ventmode file for {} doesnt exist'.format(raw_file.name))
        vm = pd.read_csv(ventmode_file)
        study_supported_modes = ['vc', 'pc', 'prvc']
        cvc_modes = ['vc', 'pc', 'prvc', 'cpap_sbt', 'ps']
        saved_bns = extra_br_metadata[:, 0]
        mode_df = vm[vm.bn.isin(saved_bns)]
        if mode_df[study_supported_modes].sum().sum() != len(mode_df) and not is_cvc:
            raise Exception('file {} has ventmodes that are not supported by this study'.format(raw_file.name))
        mode_df['ventmode'] = ''

        if not is_cvc:
            for mode in study_supported_modes:
                mode_df.loc[mode_df[mode] == 1, 'ventmode'] = mode
        else:
            for mode in cvc_modes:
                mode_df.loc[mode_df[mode] == 1, 'ventmode'] = mode

        try:
            return np.append(extra_br_metadata, np.expand_dims(mode_df['ventmode'].values, axis=1), axis=1)
        except:
            import IPython; IPython.embed()

    def check_is_qi_cohort_plat(self, rows, plat_time):
        for i, row in rows.iterrows():
            approx_time = pd.to_datetime(row['approx_plat_time'])
            if row['is_qi_cohort_plat'] == 'y' and approx_time - self.plat_wiggle_time <= plat_time <= approx_time + self.plat_wiggle_time:
                return True
        else:
            return False

    def get_plats_for_patient(self, patient_id, rows):
        plats = []
        for f in sorted(list(self.raw_data_dir.joinpath(patient_id).glob('*.csv'))):
            # just start at the beginning of time for sanity purposes and
            # saving extra if/then blocks
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
        return plats

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

        # Step 1: check for valid plats and add them to an array
        patient_id = rows.iloc[0]['patient_id']
        is_cvc = {'y': True, 'n': False}[rows.iloc[0].is_cvc]
        if not is_cvc:
            plats = self.get_plats_for_patient(patient_id, rows)
        else:
            # XXX for cvc data it might be helpful in the future to utilize
            # measured plat rather than the QuickLung plat setting.
            plats = []

        # Step 2: find which breaths that we want to save which are within range
        # of our plateau pressures
        br_to_save = {f: [] for f in sorted(list(self.raw_data_dir.joinpath(patient_id).glob('*.csv')))}
        for f, bn, plat_time in plats:
            br_to_save[f].append((bn, True))

        for f in sorted(list(self.raw_data_dir.joinpath(patient_id).glob('*.csv'))):
            with open(str(f), encoding='ascii', errors='ignore') as desc:
                gen = extract_raw(desc, False)

                for br in gen:
                    dt = pd.to_datetime(br['abs_bs'], format='%Y-%m-%d %H-%M-%S.%f')
                    if is_cvc:
                        br_to_save[f].append((br['rel_bn'], False))
                        continue

                    for _, __, plat_time in plats:

                        if (f, br['rel_bn'], dt) in plats:
                            continue
                        elif plat_time - pd.Timedelta(hours=0.5) < dt < plat_time + pd.Timedelta(hours=0.5):
                            br_to_save[f].append((br['rel_bn'], False))
                            break

        # Step 3: run ventmode and PVA algos. ventmode can be just used to add more
        # information and double check we arent using data we shouldn't be using. PVA
        # algos are used to add more information for analytics.
        for f, vals in br_to_save.items():
            vals = sorted(vals, key=lambda x: x[0])
            if len(vals) == 0:
                continue

            with open(str(f), encoding='ascii', errors='ignore') as desc:
                extra_br_metadata = np.array(vals)
                ventmode_file = f.parent.joinpath('../../ventmodes/{}'.format(f.name.replace('.csv', '_1-ventmode-output.csv')))
                extra_br_metadata = self.add_ventmode_metadata(extra_br_metadata, ventmode_file, f, is_cvc)
                output_fname = self.processed_data_dir.joinpath(patient_id, f.name.replace('.csv', ''))
                if not output_fname.parent.exists():
                    output_fname.parent.mkdir()

                extra_output_fname = self.processed_data_dir.joinpath(patient_id, f.name.replace('.csv', '.extra.pkl'))
                process_breath_file(desc, False, str(output_fname), spec_rel_bns=extra_br_metadata[:, 0])
                extra_results_frame = pd.DataFrame(extra_br_metadata, columns=['rel_bn', 'is_valid_plat', 'ventmode'])

            # this is where we use our private PVA detection software. For outside purposes, we've
            # saved the results to file. You can use the results of the algorithm in your own
            # reproduction if you need.
            breaths = list(read_processed_file(str(output_fname)+'.raw.npy'))
            pva, pva_fused = detectPVI(breaths, output_subdir='/tmp', write_results=False)
            if len(breaths) != len(extra_results_frame):
                raise Exception('not supposed to happen, pt: {}'.format(patient_id))

            vc_only = [b for idx, b in enumerate(breaths) if extra_results_frame.iloc[idx]['ventmode'] == 'vc']
            pressure_only = [b for idx, b in enumerate(breaths) if extra_results_frame.iloc[idx]['ventmode'] in ['pc', 'prvc']]
            flow_asyncs = get_gen_flow_async(vc_only)
            dca = extract_new_feature(pva, pressure_only)
            # this stands for double trigger asynchrony
            extra_results_frame['dta'] = pva['dbl.4']
            # this stands for breath stacking asynchrony
            extra_results_frame['bsa'] = pva['bs.1or2']
            # this stands for breathing artifacts like suction, cough, etc.
            extra_results_frame['artifact'] = pva['cosumtvd']
            extra_results_frame = extra_results_frame.merge(flow_asyncs, on='rel_bn', how='left')
            extra_results_frame.rename(columns={'severity': 'fa', 'location': 'fa_loc'}, inplace=True)
            no_vc_mask = extra_results_frame.fa.isna()
            extra_results_frame.loc[no_vc_mask, 'fa'] = 0
            extra_results_frame.loc[no_vc_mask, 'fa_loc'] = 'NA'
            dca_cols = ['Dyna_DCA', 'dDCA', 'Static_DCA']
            dca = dca[['rel_bn']+dca_cols]
            extra_results_frame = extra_results_frame.merge(dca, on='rel_bn', how='left')
            no_pc_mask = extra_results_frame.Dyna_DCA.isna()
            extra_results_frame.loc[no_pc_mask, dca_cols] = 0
            extra_results_frame = extra_results_frame.rename(columns={'Dyna_DCA': 'dyn_dca', 'Static_DCA': 'static_dca', 'dDCA': 'dyn_dca_timing'})
            extra_results_frame.to_pickle(str(extra_output_fname), protocol=4)

    def iter_raw_dir(self, only_patient, cvc_only):
        if only_patient and only_patient not in self.cohort['patient_id'].unique():
            raise Exception('patient {} is not in the cohort definition.'.format(only_patient))
        for patient_id, rows in self.cohort.groupby('patient_id'):
            if cvc_only and 'cvc' not in patient_id:
                continue
            if only_patient and only_patient == patient_id:
                self.iterate_on_pt(rows)
            elif not only_patient:
                self.iterate_on_pt(rows)

    def make_gk_clust(self, min_obs, only_patient):
        """
        Make GK clustering obj using Babuska's fuzzy clustering algo for the exp. curve only

        Not actually too sure what I was doing here. Looks like something experimental
        """
        z = []
        for patient_dir in self.processed_data_dir.glob('*RPI*'):
            patient_id = patient_dir.name
            if only_patient and patient_id != only_patient:
                continue
            for filename in patient_dir.glob('*.raw.npy'):
                gen = read_processed_file(str(filename.resolve()))
                for breath in gen:
                    flow = np.array(breath['flow'])/60
                    bm = get_production_breath_meta(breath, to_series=True)
                    if bm.x0_index >= len(flow)-1:
                        continue
                    min_f = np.argmin(flow)
                    if len(flow[min_f:]) < min_obs:
                        continue
                    vols = calc_volumes(flow, 0.02)
                    z.append(np.array([vols[min_f:], flow[min_f:]]).T)
        # concat is faster. But breathmeta and GK fit take up vast majority of the time
        z = np.concatenate(z)
        # only use 2 clusters because we just divide the paper's n clusters by 2.
        # m=2 was the param used in the paper
        gk = GK(n_clusters=2, m=2)
        z = np.array(z)
        gk.fit(z)
        objs_dir = self.processed_data_dir.joinpath('pickled_objs')
        if not objs_dir.exists():
            objs_dir.mkdir()
        gk_pathname = objs_dir.joinpath('gk.pkl').resolve()
        pd.to_pickle(gk, gk_pathname, compression=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cohort', default='cohort.csv')
    parser.add_argument('-rdp', '--raw-dataset-path', default='dataset/raw_data')
    parser.add_argument('-pdp', '--processed-dataset-path', default='dataset/processed_data')
    parser.add_argument('--flow-bound', type=float, default=0.2, help='flow bound for plat pressures')
    parser.add_argument('--min-plat-time', type=float, default=0.4, help='minimum amount of time a plat must occur for')
    parser.add_argument('--any-or-all', choices=['any', 'all'], default='any', help='refers to plat check boundary')
    parser.add_argument('--only-patient', help='only run specific patient')
    parser.add_argument('--only-gk', action='store_true', help='only perform gk clustering for fuzzy clustering algo')
    parser.add_argument('--no-gk', action='store_true', help='dont run gk clustering')
    parser.add_argument('--cvc-only', action='store_true', help='only run cvc patients')
    args = parser.parse_args()

    proc = Processing(args.cohort, args.raw_dataset_path, args.processed_dataset_path, args.min_plat_time, args.flow_bound, args.any_or_all)
    if not args.only_gk:
        proc.iter_raw_dir(args.only_patient, args.cvc_only)
    if not args.no_gk:
        proc.make_gk_clust(10, args.only_patient)


if __name__ == '__main__':
    main()
