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
            # XXX debug just keep this line around in case a patient is failing for now
            if '0641RPI' not in str(file):
                pass
            # XXX debug
            print('run file {}'.format(str(file)))
            algo = 'iimipr'
            algo2 = 'iipredator'
            baseline = 'insp_least_squares'
            calcs = FileCalculations(str(file), [algo, algo2, baseline], 9)
            calcs.analyze_file()
            preds = calcs.results[algo]
            gt = calcs.results['gold_stnd_compliance']
            # XXX debug
            import matplotlib.pyplot as plt
            plt.plot(preds, label=algo)
            plt.plot(calcs.results[algo2], label=algo2)
            plt.plot(calcs.results[baseline], label=baseline)
            plt.plot(gt, label='gt')
            plt.legend()
            plt.show()
            plt.close()

if __name__ == '__main__':
    main()
