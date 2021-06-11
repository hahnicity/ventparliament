from nose.tools import eq_
import numpy as np

from parliament.main import ResultsContainer, rolling_nan_mean, rolling_nan_median


def test_rolling_nan_median_win2():
    vals = np.array([1, 2, 3, 4, 5, 6])
    out = rolling_nan_median(vals, 2)
    assert (out[1:] == np.array([1.5, 2.5, 3.5, 4.5, 5.5])).all(), out[1:]
    assert np.isnan(out[0])


def test_rolling_nan_median_win3():
    vals = np.array([1, 2, 3, 4, 5, 6])
    out = rolling_nan_median(vals, 3)
    assert (out[2:] == np.array([2, 3, 4, 5])).all(), out[2:]
    assert np.isnan(out[0])
    assert np.isnan(out[1])


def test_rolling_nan_mean_win3():
    vals = np.array([2, 2, 5, 11, -4, 11])
    out = rolling_nan_mean(vals, 3)
    assert (out[2:] == np.array([3, 6, 4, 6])).all(), out[2:]
    assert np.isnan(out[0])
    assert np.isnan(out[1])


class TestResultsContainer(object):
    def __init__(self):
        pass

    def setup(self):
        self.test_con = ResultsContainer.load_from_experiment_name('testing_base')
        self.test_con.window_n = 2

    def test_calc_windows(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        r = self.test_con.proc_results
        # sanity checks
        assert (r.gold_stnd_compliance ==  [50]*5+[20]*5).all()
        assert self.test_con.window_n == 2
        # now for some actual columns
        for algo in self.test_con.algos_used:
            for pt, pt_df in r.groupby('patient_id'):
                assert np.isnan(pt_df.iloc[0][algo + '_wm_2'])
                for i in range(1, 5):
                    out = np.nanmedian(pt_df.iloc[i-1:i+1][algo])
                    if np.isnan(out):
                        assert np.isnan(pt_df.iloc[i][algo + '_wm_2'])
                    else:
                        eq_(pt_df.iloc[i][algo + '_wm_2'], out)

        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['dtw_wm_2'])
            for i in range(1, 5):
                out = np.nanmedian(pt_df.iloc[i-1:i+1]['dtw'])
                if np.isnan(out):
                    assert np.isnan(pt_df.iloc[i]['dtw_wm_2'])
                else:
                    eq_(pt_df.iloc[i]['dtw_wm_2'], out)

        for algo in self.test_con.algos_used:
            for pt, pt_df in r.groupby('patient_id'):
                assert np.isnan(pt_df.iloc[0][algo + '_wmd_2'])
                for i in range(1, 5):
                    out = np.nanmedian(pt_df.iloc[i-1:i+1][algo])
                    gold = pt_df.iloc[i].gold_stnd_compliance
                    if np.isnan(out):
                        assert np.isnan(pt_df.iloc[i][algo + '_wmd_2'])
                    else:
                        eq_(pt_df.iloc[i][algo + '_wmd_2'], gold-out)

        # XXX need to do sm/smd
