import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default=str(Path(__file__).parent.joinpath('../dataset/processed_data')))
    parser.add_argument('--cvc-only', action='store_true', help='only analyze cvc data')
    parser.add_argument('--no-cvc', action='store_true', help='dont analyze cvc data')
    args = parser.parse_args()

    all_patient_dirs = Path(args.data_path).glob('*')
    all_extra = []

    for dir_ in sorted(list(all_patient_dirs)):
        if args.cvc_only and 'cvc' not in str(dir_):
            continue

        if args.no_cvc and 'cvc' in str(dir_):
            continue

        patient_id = dir_.name
        for file in dir_.glob('*.raw.npy'):
            patient_id = file.parent.name
            extra = pd.read_pickle(str(file).replace('raw.npy', 'extra.pkl'))
            all_extra.append(extra)
    cohort = pd.concat(all_extra)
    import IPython; IPython.embed()


if __name__ == '__main__':
    main()
