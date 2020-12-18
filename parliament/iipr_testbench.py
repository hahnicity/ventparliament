import argparse

import numpy as np
import pandas as pd
from ventmap.breath_meta import get_production_breath_meta
from ventmap.constants import META_HEADER
from ventmap.raw_utils import extract_raw

from parliament.iipr import perform_iipr_algo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('file')
    p.add_argument('rel_bn', type=int)
    a = p.parse_args()

    gen = extract_raw(open(a.file, encoding='ascii', errors='ignore'), False)
    for b in gen:
        if b['rel_bn'] == a.rel_bn:
            break
    else:
        raise Exception('breath not found')

    bm = pd.Series(get_production_breath_meta(b), index=META_HEADER)
    flow = np.array(b['flow']) / 60
    pressure = np.array(b['pressure'])
    peep = bm.PEEP
    x0 = bm.x0_index
    out = perform_iipr_algo(flow, pressure, x0, peep, b['dt'])
    print("compliance: {}, residual: {}, code: {}".format(out[0], out[2], out[3]))


if __name__ == "__main__":
    main()
