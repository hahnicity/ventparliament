from nose.tools import assert_list_equal, eq_
import numpy as np

from parliament.main import ResultsContainer, rolling_nan_mean, rolling_nan_median, sequential_nan_median


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


def test_sequential_nan_median_win3():
    vals = np.array([2, 2, 5, 11, -4, 11])
    out = sequential_nan_median(vals, 3)
    assert (out[~np.isnan(out)] == np.array([2, 11])).all(), out[~np.isnan(out)]
    nan_idxs = [0, 1, 3, 4]
    for idx in nan_idxs:
        assert np.isnan(out[idx]), out


def test_sequential_nan_median_win4():
    vals = np.array([2, 2, 5, 11, -4, 11])
    out = sequential_nan_median(vals, 4)
    assert (out[~np.isnan(out)] == np.array([3.5])).all(), out[~np.isnan(out)]
    nan_idxs = [0, 1, 2, 4, 5]
    for idx in nan_idxs:
        assert np.isnan(out[idx]), out


class TestResultsContainer(object):
    def setup(self):
        self.test_con = ResultsContainer.load_from_experiment_name('testing_base')
        self.test_con.window_n = 2

    def test_calc_windows_diff(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        r = self.test_con.proc_results
        # sanity checks
        assert (r.gold_stnd_compliance ==  [50]*5+[20]*5).all()
        assert self.test_con.window_n == 2

        for algo in self.test_con.algos_used:
            for pt, pt_df in r.groupby('patient_id'):
                mask = np.isnan(pt_df[algo].values)
                if mask.any():
                    assert np.isnan(pt_df[algo+'_diff'][mask]).all()

                if pt == '0210RPI05':
                    assert_list_equal(list(pt_df[algo + '_diff'][~mask]), list(50 - pt_df[algo][~mask]))
                elif pt == '0640RPI28':
                    assert_list_equal(list(pt_df[algo + '_diff'][~mask]), list(20 - pt_df[algo][~mask]))

    def test_calc_windows_wm(self):
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

    def test_calc_windows_dtw_wm(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        r = self.test_con.proc_results
        # sanity checks
        assert (r.gold_stnd_compliance ==  [50]*5+[20]*5).all()
        assert self.test_con.window_n == 2

        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['dtw_wm_2'])
            for i in range(1, 5):
                out = np.nanmedian(pt_df.iloc[i-1:i+1]['dtw'])
                if np.isnan(out):
                    assert np.isnan(pt_df.iloc[i]['dtw_wm_2'])
                else:
                    eq_(pt_df.iloc[i]['dtw_wm_2'], out)

    def test_calc_windows_dtw_wmd(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        r = self.test_con.proc_results
        # sanity checks
        assert (r.gold_stnd_compliance ==  [50]*5+[20]*5).all()
        assert self.test_con.window_n == 2
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

    def test_calc_windows_dtw_sm(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        r = self.test_con.proc_results
        # sanity checks
        assert (r.gold_stnd_compliance ==  [50]*5+[20]*5).all()
        assert self.test_con.window_n == 2
        for algo in self.test_con.algos_used:
            for pt, pt_df in r.groupby('patient_id'):
                out1 = np.nanmedian(pt_df.iloc[0:2][algo])
                out2 = np.nanmedian(pt_df.iloc[2:4][algo])
                for idx, o in [(1, out1), (3, out2)]:
                    if np.isnan(o):
                        assert np.isnan(pt_df.iloc[idx][algo + '_sm_2'])
                    else:
                        eq_(o, pt_df.iloc[idx][algo + '_sm_2'])
                nan_idxs = [0, 2, 4]
                for idx in nan_idxs:
                    assert np.isnan(pt_df.iloc[idx][algo + '_sm_2']), out

    def test_calc_windows_dtw_smd(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        r = self.test_con.proc_results
        # sanity checks
        assert (r.gold_stnd_compliance ==  [50]*5+[20]*5).all()
        assert self.test_con.window_n == 2
        for algo in self.test_con.algos_used:
            for pt, pt_df in r.groupby('patient_id'):
                out1 = pt_df.iloc[0].gold_stnd_compliance - np.nanmedian(pt_df.iloc[0:2][algo])
                out2 = pt_df.iloc[0].gold_stnd_compliance - np.nanmedian(pt_df.iloc[2:4][algo])
                for idx, o in [(1, out1), (3, out2)]:
                    if np.isnan(o):
                        assert np.isnan(pt_df.iloc[idx][algo + '_smd_2'])
                    else:
                        eq_(o, pt_df.iloc[idx][algo + '_smd_2'])
                nan_idxs = [0, 2, 4]
                for idx in nan_idxs:
                    assert np.isnan(pt_df.iloc[idx][algo + '_smd_2']), out

    def test_calc_async_index_preprocessing_assertions(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        dtas = r.loc[r.dta > 0, 'dta'].values
        assert (dtas < 2).all()
        assert len(dtas[dtas>0]) == 2  # I manually set 2 breaths to be dta
        assert not (np.isnan(r.insp_efforting.values)).any()
        assert (r.loc[r.fa == 1, 'fa_mild'] == 1).values.all()
        assert (r.loc[r.fa == 2, 'fa_mod'] == 1).values.all()
        assert (r.loc[r.fa == 3, 'fa_sev'] == 1).values.all()
        assert (r.loc[r.fa == 2, 'fa_mild'] != 1).values.all()
        assert (r.loc[r.fa == 3, 'fa_mild'] != 1).values.all()
        assert (r.loc[r.fa == 1, 'fa_mod'] != 1).values.all()
        assert (r.loc[r.fa == 3, 'fa_mod'] != 1).values.all()
        assert (r.loc[r.fa == 1, 'fa_sev'] != 1).values.all()
        assert (r.loc[r.fa == 2, 'fa_sev'] != 1).values.all()
        assert len(r.loc[r.fa_mild == 1].values) == 2
        assert len(r.loc[r.fa_mod == 1].values) == 1
        assert len(r.loc[r.fa_sev == 1].values) == 1

    def test_calc_async_index_asynci(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['asynci_2'])
            # do manual check because it provides additional safety against function
            # failure
            if pt == '0210RPI05':
                assert pt_df.iloc[1]['asynci_2'] == 1.0
                assert pt_df.iloc[2]['asynci_2'] == 1.0
                assert pt_df.iloc[3]['asynci_2'] == 1.0
                assert pt_df.iloc[4]['asynci_2'] == 0.5
            elif pt == '0640RPI28':
                assert pt_df.iloc[1]['asynci_2'] == 1.0
                assert pt_df.iloc[2]['asynci_2'] == 1.0
                assert pt_df.iloc[3]['asynci_2'] == 1.0
                assert pt_df.iloc[4]['asynci_2'] == 0.5

    def test_calc_async_index_asynci_no_fam(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['asynci_no_fam_2'])
            # do manual check because it provides additional safety against function
            # failure
            if pt == '0210RPI05':
                assert pt_df.iloc[1]['asynci_no_fam_2'] == 1.0
                assert pt_df.iloc[2]['asynci_no_fam_2'] == 1.0
                assert pt_df.iloc[3]['asynci_no_fam_2'] == 0.5
                assert pt_df.iloc[4]['asynci_no_fam_2'] == 0.0
            elif pt == '0640RPI28':
                assert pt_df.iloc[1]['asynci_no_fam_2'] == 1.0
                assert pt_df.iloc[2]['asynci_no_fam_2'] == 1.0
                assert pt_df.iloc[3]['asynci_no_fam_2'] == 1.0
                assert pt_df.iloc[4]['asynci_no_fam_2'] == 0.5

    def test_calc_async_index_bsi(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['bsi_2'])
            # do manual check because it provides additional safety against function
            # failure
            if pt == '0210RPI05':
                assert pt_df.iloc[1]['bsi_2'] == 0.5
                assert pt_df.iloc[2]['bsi_2'] == 0.5
                assert pt_df.iloc[3]['bsi_2'] == 0.5
                assert pt_df.iloc[4]['bsi_2'] == 0.0
            elif pt == '0640RPI28':
                assert pt_df.iloc[1]['bsi_2'] == 0.0
                assert pt_df.iloc[2]['bsi_2'] == 0.0
                assert pt_df.iloc[3]['bsi_2'] == 0.0
                assert pt_df.iloc[4]['bsi_2'] == 0.0

    def test_calc_async_index_dci(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['dci_2'])
            # do manual check because it provides additional safety against function
            # failure
            if pt == '0210RPI05':
                assert pt_df.iloc[1]['dci_2'] == 0.0
                assert pt_df.iloc[2]['dci_2'] == 0.0
                assert pt_df.iloc[3]['dci_2'] == 0.0
                assert pt_df.iloc[4]['dci_2'] == 0.0
            elif pt == '0640RPI28':
                assert pt_df.iloc[1]['dci_2'] == 1.0
                assert pt_df.iloc[2]['dci_2'] == 1.0
                assert pt_df.iloc[3]['dci_2'] == 1.0
                assert pt_df.iloc[4]['dci_2'] == 0.5

    def test_calc_async_index_dti(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['dti_2'])
            # do manual check because it provides additional safety against function
            # failure
            if pt == '0210RPI05':
                assert pt_df.iloc[1]['dti_2'] == 1.0
                assert pt_df.iloc[2]['dti_2'] == 0.5
                assert pt_df.iloc[3]['dti_2'] == 0.0
                assert pt_df.iloc[4]['dti_2'] == 0.0
            elif pt == '0640RPI28':
                assert pt_df.iloc[1]['dti_2'] == 0.0
                assert pt_df.iloc[2]['dti_2'] == 0.0
                assert pt_df.iloc[3]['dti_2'] == 0.0
                assert pt_df.iloc[4]['dti_2'] == 0.0

    def test_calc_async_index_fai(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['fai_2'])
            # do manual check because it provides additional safety against function
            # failure
            if pt == '0210RPI05':
                assert pt_df.iloc[1]['fai_2'] == 1.0
                assert pt_df.iloc[2]['fai_2'] == 1.0
                assert pt_df.iloc[3]['fai_2'] == 1.0
                assert pt_df.iloc[4]['fai_2'] == 0.5
            elif pt == '0640RPI28':
                assert pt_df.iloc[1]['fai_2'] == 0.0
                assert pt_df.iloc[2]['fai_2'] == 0.0
                assert pt_df.iloc[3]['fai_2'] == 0.0
                assert pt_df.iloc[4]['fai_2'] == 0.0

    def test_calc_async_index_fai_no_fam(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['fai_no_fam_2'])
            # do manual check because it provides additional safety against function
            # failure
            if pt == '0210RPI05':
                assert pt_df.iloc[1]['fai_no_fam_2'] == 0.5
                assert pt_df.iloc[2]['fai_no_fam_2'] == 1.0
                assert pt_df.iloc[3]['fai_no_fam_2'] == 0.5
                assert pt_df.iloc[4]['fai_no_fam_2'] == 0.0
            elif pt == '0640RPI28':
                assert pt_df.iloc[1]['fai_no_fam_2'] == 0.0
                assert pt_df.iloc[2]['fai_no_fam_2'] == 0.0
                assert pt_df.iloc[3]['fai_no_fam_2'] == 0.0
                assert pt_df.iloc[4]['fai_no_fam_2'] == 0.0

    def test_calc_async_index_insp_effi(self):
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        for pt, pt_df in r.groupby('patient_id'):
            assert np.isnan(pt_df.iloc[0]['insp_effi_2'])
            # do manual check because it provides additional safety against function
            # failure
            if pt == '0210RPI05':
                assert pt_df.iloc[1]['insp_effi_2'] == 1.0
                assert pt_df.iloc[2]['insp_effi_2'] == 1.0
                assert pt_df.iloc[3]['insp_effi_2'] == 1.0
                assert pt_df.iloc[4]['insp_effi_2'] == 1.0
            elif pt == '0640RPI28':
                assert pt_df.iloc[1]['insp_effi_2'] == 0.0
                assert pt_df.iloc[2]['insp_effi_2'] == 0.0
                assert pt_df.iloc[3]['insp_effi_2'] == 0.0
                assert pt_df.iloc[4]['insp_effi_2'] == 0.0

    def test_analyze_per_patient_df_mad_std(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        pp_all = self.test_con.analyze_per_patient_df(r)

        # sanity
        assert set(pp_all['algo'].values) == set(['iipr', 'vicario_nieap'])
        assert len(pp_all) == 4

        pt0210 = pp_all[pp_all.patient_id == '0210RPI05']
        pt0640 = pp_all[pp_all.patient_id == '0640RPI28']
        dfs = [pt0210, pt0640]
        for idx, (pt, pt_df) in enumerate(r.groupby('patient_id')):
            d = dfs[idx]
            # sanity
            assert d.patient_id.unique()[0] == pt
            iipr_row = d[d.algo == 'iipr'].iloc[0]
            vic_row = d[d.algo == 'vicario_nieap'].iloc[0]
            eq_(vic_row.mad_pt, np.nanmedian(pt_df['vicario_nieap_diff'].abs()))
            eq_(vic_row.std_pt, np.nanstd(pt_df['vicario_nieap']))
            if pt == '0640RPI28':
                assert np.isnan(iipr_row.mad_pt)
                assert np.isnan(iipr_row.std_pt)
                continue
            eq_(iipr_row.std_pt, np.nanstd(pt_df['iipr']))
            eq_(iipr_row.mad_pt, np.nanmedian(pt_df['iipr_diff'].abs()))

    def test_analyze_per_patient_df_wmd(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        pp_all = self.test_con.analyze_per_patient_df(r)

        # sanity
        assert set(pp_all['algo'].values) == set(['iipr', 'vicario_nieap'])
        assert len(pp_all) == 4

        pt0210 = pp_all[pp_all.patient_id == '0210RPI05']
        pt0640 = pp_all[pp_all.patient_id == '0640RPI28']
        dfs = [pt0210, pt0640]
        for idx, (pt, pt_df) in enumerate(r.groupby('patient_id')):
            d = dfs[idx]
            # sanity
            assert d.patient_id.unique()[0] == pt
            iipr_row = d[d.algo == 'iipr'].iloc[0]
            vic_row = d[d.algo == 'vicario_nieap'].iloc[0]
            eq_(vic_row.mad_wmd, np.nanmedian(pt_df['vicario_nieap_wmd_2'].abs()))
            eq_(vic_row.std_wmd, np.nanstd(pt_df['vicario_nieap_wmd_2']))
            if pt == '0640RPI28':
                assert np.isnan(iipr_row.mad_wmd)
                assert np.isnan(iipr_row.std_wmd)
                continue
            eq_(iipr_row.mad_wmd, np.nanmedian(pt_df['iipr_wmd_2'].abs()))
            eq_(iipr_row.std_wmd, np.nanstd(pt_df['iipr_wmd_2']))

    def test_analyze_per_patient_df_smd(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        pp_all = self.test_con.analyze_per_patient_df(r)

        # sanity
        assert set(pp_all['algo'].values) == set(['iipr', 'vicario_nieap'])
        assert len(pp_all) == 4

        pt0210 = pp_all[pp_all.patient_id == '0210RPI05']
        pt0640 = pp_all[pp_all.patient_id == '0640RPI28']
        dfs = [pt0210, pt0640]
        for idx, (pt, pt_df) in enumerate(r.groupby('patient_id')):
            d = dfs[idx]
            # sanity
            assert d.patient_id.unique()[0] == pt
            iipr_row = d[d.algo == 'iipr'].iloc[0]
            vic_row = d[d.algo == 'vicario_nieap'].iloc[0]
            eq_(vic_row.mad_smd, np.nanmedian(pt_df['vicario_nieap_smd_2'].abs()))
            eq_(vic_row.std_smd, np.nanstd(pt_df['vicario_nieap_smd_2']))
            if pt == '0640RPI28':
                assert np.isnan(iipr_row.mad_smd)
                assert np.isnan(iipr_row.std_smd)
                continue
            eq_(iipr_row.mad_smd, np.nanmedian(pt_df['iipr_smd_2'].abs()))
            eq_(iipr_row.std_smd, np.nanstd(pt_df['iipr_smd_2']))

    def test_preprocess_mad_std_in_df_no_window(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        pp_all = self.test_con.analyze_per_patient_df(r)
        mad_std, algos = self.test_con.preprocess_mad_std_in_df(pp_all, None)

        assert len(mad_std) == 2
        assert_list_equal(algos, ['iipr', 'vicario_nieap'])
        mads = pp_all[pp_all.algo == 'iipr']['mad_pt']
        assert (mad_std['iipr'][0]==mads[~mads.isna()]).all()
        mads = pp_all[pp_all.algo == 'vicario_nieap']['mad_pt']
        assert (mad_std['vicario_nieap'][0]==mads[~mads.isna()]).all()

        stds = pp_all[pp_all.algo == 'iipr']['std_pt']
        assert (mad_std['iipr'][1]==stds[~stds.isna()]).all()
        stds = pp_all[pp_all.algo == 'vicario_nieap']['std_pt']
        assert (mad_std['vicario_nieap'][1]==stds[~stds.isna()]).all()

        mads = pp_all[pp_all.algo == 'iipr']['mad_pt']
        assert (mad_std['iipr'][2]==np.nanmean(mads))
        mads = pp_all[pp_all.algo == 'vicario_nieap']['mad_pt']
        assert (mad_std['vicario_nieap'][2]==np.nanmean(mads))

        stds = pp_all[pp_all.algo == 'iipr']['std_pt']
        assert (mad_std['iipr'][3]==np.nanmean(stds))
        stds = pp_all[pp_all.algo == 'vicario_nieap']['std_pt']
        assert (mad_std['vicario_nieap'][3]==np.nanmean(stds))

    def test_preprocess_mad_std_in_df_wmd(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        pp_all = self.test_con.analyze_per_patient_df(r)
        mad_std, algos = self.test_con.preprocess_mad_std_in_df(pp_all, 'wmd')

        assert len(mad_std) == 2
        assert_list_equal(algos, ['iipr', 'vicario_nieap'])
        mads = pp_all[pp_all.algo == 'iipr']['mad_wmd']
        assert (mad_std['iipr'][0]==mads[~mads.isna()]).all()
        mads = pp_all[pp_all.algo == 'vicario_nieap']['mad_wmd']
        assert (mad_std['vicario_nieap'][0]==mads[~mads.isna()]).all()

        stds = pp_all[pp_all.algo == 'iipr']['std_wmd']
        assert (mad_std['iipr'][1]==stds[~stds.isna()]).all()
        stds = pp_all[pp_all.algo == 'vicario_nieap']['std_wmd']
        assert (mad_std['vicario_nieap'][1]==stds[~stds.isna()]).all()

        mads = pp_all[pp_all.algo == 'iipr']['mad_wmd']
        assert (mad_std['iipr'][2]==np.nanmean(mads))
        mads = pp_all[pp_all.algo == 'vicario_nieap']['mad_wmd']
        assert (mad_std['vicario_nieap'][2]==np.nanmean(mads))

        stds = pp_all[pp_all.algo == 'iipr']['std_wmd']
        assert (mad_std['iipr'][3]==np.nanmean(stds))
        stds = pp_all[pp_all.algo == 'vicario_nieap']['std_wmd']
        assert (mad_std['vicario_nieap'][3]==np.nanmean(stds))

    def test_preprocess_mad_std_in_df_smd(self):
        self.test_con.calc_windows(self.test_con.proc_results)
        self.test_con.calc_async_index(self.test_con.proc_results)
        r = self.test_con.proc_results
        pp_all = self.test_con.analyze_per_patient_df(r)
        mad_std, algos = self.test_con.preprocess_mad_std_in_df(pp_all, 'smd')

        assert len(mad_std) == 2
        assert_list_equal(algos, ['iipr', 'vicario_nieap'])
        mads = pp_all[pp_all.algo == 'iipr']['mad_smd']
        assert (mad_std['iipr'][0]==mads[~mads.isna()]).all()
        mads = pp_all[pp_all.algo == 'vicario_nieap']['mad_smd']
        assert (mad_std['vicario_nieap'][0]==mads[~mads.isna()]).all()

        stds = pp_all[pp_all.algo == 'iipr']['std_smd']
        assert (mad_std['iipr'][1]==stds[~stds.isna()]).all()
        stds = pp_all[pp_all.algo == 'vicario_nieap']['std_smd']
        assert (mad_std['vicario_nieap'][1]==stds[~stds.isna()]).all()

        mads = pp_all[pp_all.algo == 'iipr']['mad_smd']
        assert (mad_std['iipr'][2]==np.nanmean(mads))
        mads = pp_all[pp_all.algo == 'vicario_nieap']['mad_smd']
        assert (mad_std['vicario_nieap'][2]==np.nanmean(mads))

        stds = pp_all[pp_all.algo == 'iipr']['std_smd']
        assert (mad_std['iipr'][3]==np.nanmean(stds))
        stds = pp_all[pp_all.algo == 'vicario_nieap']['std_smd']
        assert (mad_std['vicario_nieap'][3]==np.nanmean(stds))
