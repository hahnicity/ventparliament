"""
main
~~~~
"""
import argparse
from pathlib import Path

from parliament.analyze import FileCalculations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default=str(Path(__file__).parent.joinpath('../dataset/processed_data')))
    args = parser.parse_args()

    all_patient_dirs = Path(args.data_path).glob('*')
    for dir_ in sorted(list(all_patient_dirs)):
        for file in dir_.glob('*.raw.npy'):
            algo = 'brunner'
            calcs = FileCalculations(str(file), [algo, 'insp_least_squares'], 9)
            analysis = calcs.analyze_file()
            preds = calcs.results[algo]
            gt = calcs.results['gold_stnd_compliance']
            import IPython; IPython.embed()

if __name__ == '__main__':
    main()
