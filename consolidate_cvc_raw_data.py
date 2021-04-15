import argparse

import numpy as np
from pathlib import Path
from ventmap.raw_utils import consolidate_files, extract_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('globs', help='use glob for pattern matching. patterns should be done in accordance to documentation in pythons glob.glob function', nargs='+')
    parser.add_argument('experiment_name', help='cvc experiment name')
    parser.add_argument('--spec-files', nargs='*', help='for files you dont want to glob match, just add specific files')
    args = parser.parse_args()

    paths = []
    for glob in args.globs:
        paths.extend(list(Path(args.dir).glob(glob)))
    print('\n\nselected from glob:\n\n{}\n\n'.format('\n'.join([str(p) for p in sorted(paths)])))
    correct = input('is this selection correct? [y]/n  ')
    if correct == 'n' or correct == 'N':
        return
    paths += [Path(f) for f in args.spec_files] if args.spec_files is not None else []
    # Given the current workflow I think it will go like this:
    # 1. consolidate data into a single file per experiment
    # 2. do ventmode annotation
    # 3. move each cvc experiment into its own patient dir in raw
    # 4. figure out how to utilize pre-recorded compliance. Will likely jus
    #    have a small addition in raw dir to cover relevant info.
    # 5. run process_data script with small code addition to cover pre-recorded
    #    compliance.
    output_dir = Path(__file__).parent.joinpath('dataset/raw_data/{}'.format(args.experiment_name))
    if not output_dir.exists():
        output_dir.mkdir()
    consolidate_files(paths, False, output_dir, to_npy=False, to_csv=True)


if __name__ == "__main__":
    main()
