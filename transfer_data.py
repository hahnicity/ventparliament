from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path
import subprocess

import pandas as pd


def glob_gather_files(out_dir, patient_id, glob_pathing, hostname, host_data_dir):
    glob_pat = Path(host_data_dir, patient_id, glob_pathing)
    proc = subprocess.Popen(
        ['rsync', '{}:{}'.format(hostname, str(glob_pat)), str(out_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()


def get_plat_data(output_base_dir, patient_id, plat_time, hostname, host_data_dir):
    out_dir = Path(output_base_dir, patient_id)
    try:
        out_dir.mkdir()
    except OSError:
        pass
    plat_time = pd.to_datetime(plat_time)
    file_glob = "*{:04d}-{:02d}-{:02d}-{}-*"
    print('Get data for patient: {}, at plat time: {}'.format(patient_id, plat_time.strftime('%Y-%m-%d-%H-%M')))
    # the -3/+1 is just there because at max, files operate in 2 hr increments. -3 is there for a
    # super rare corner case that half hour prior would be in a file marked 3 hours behind when
    # the approximate plat time was.
    delta_time = plat_time + timedelta(hours=1)
    hour_glob = "{{{}}}".format(",".join(["{:02d}".format(i) for i in range(plat_time.hour-3, delta_time.hour+1)]))
    start_glob = file_glob.format(plat_time.year, plat_time.month, plat_time.day, hour_glob)
    glob_gather_files(out_dir, patient_id, start_glob, hostname, host_data_dir)

    # we advanced the clock a whole day
    if delta_time.hour < plat_time.hour:
        hour_glob = "{{{}}}".format(",".join(["{:02d}".format(i) for i in range(0, delta_time.hour+1)]))
        end_glob = file_glob.format(delta_time.year, delta_time.month, delta_time.day, hour_glob)
        glob_gather_files(out_dir, patient_id, end_glob, hostname, host_data_dir)



def main():
    parser = ArgumentParser()
    parser.add_argument('-o', '--output-dir', default='dataset/raw_data/')
    parser.add_argument('-d', '--cohort-description', default='cohort.csv', help='Path to file describing the cohort')
    parser.add_argument('-p', '--only-patient', help='Only gather data for specific patient id')
    parser.add_argument('-ho', '--hostname', help='server hostname to grab data from', default='b2c-main')
    parser.add_argument('-dd', '--host-data-dir', help='dirname on server where data resides', default='~')

    args = parser.parse_args()

    df = pd.read_csv(args.cohort_description)
    if args.only_patient:
        enrollment = df[df.patient_id == args.only_patient]
        if len(enrollment) == 0:
            raise Exception('Could not find any rows in cohort for patient {}'.format(args.only_patient))
    else:
        enrollment = df

    for idx, row in enrollment.iterrows():
        get_plat_data(args.output_dir, row.patient_id, row.approx_plat_time, args.hostname, args.host_data_dir)


if __name__ == "__main__":
    main()
