import numpy as np
import pandas as pd
from scipy.integrate import simps
from ventmap.breath_meta import get_production_breath_meta
from ventmap.constants import META_HEADER
from ventmap.raw_utils import read_processed_file
from ventmap.SAM import calc_expiratory_plateau, calc_inspiratory_plateau

from parliament.iipr import perform_iipr_algo, perform_iipr_pressure_reconstruction
from parliament.mipr import perform_mipr
from parliament.other_calcs import (
    al_rawas_calcs,
    al_rawas_expiratory_const,
    brunner,
    expiratory_least_squares,
    howe_expiratory_least_squares,
    inspiratory_least_squares,
    lourens_time_const,
    vicario_nieap
)
from parliament.polynomial_model import perform_polynomial_model
from parliament.predator import perform_predator_algo
from parliament.pressure_ctrl_correction import perform_algo as kannangara
from parliament.vicario_constrained import perform_constrained_optimization


class FileCalculations(object):
    def __init__(self, filename, algorithms_to_use, peeps_to_use, recorded_compliance=None, **kwargs):
        """
        Calculate lung compliance for an entire file using a variety of algorithms

        :param filename: filename to analyze
        :param algorithms_to_use: Algorithms you want to include in your analysis. Should
        be a list and consist of choices: 'vicario_co', 'kannangara', 'insp_least_squares',
        'brunner', 'vicario_nieap', ' al_rawas'
        :param peeps_to_use: Number of PEEPs to use when we calculate a median
        :param recorded_compliance: Compliance pre-recorded for the file.
        """
        self.algorithms_to_use = algorithms_to_use
        self.peeps_to_use = peeps_to_use
        self.results = []
        self.results_cols = ['gold_stnd_compliance'] + algorithms_to_use
        self.breath_data = list(read_processed_file(filename))
        self.breath_metadata = pd.DataFrame([
            get_production_breath_meta(breath) for breath in self.breath_data
        ], columns=META_HEADER)
        self.algo_mapping = {
            "al_rawas": self.al_rawas,
            "exp_least_squares": self.exp_least_squares,
            "howe_least_squares": self.howe_least_squares,
            'iipr': self.iipr,
            'iipredator': self.iipredator,
            "insp_least_squares": self.insp_least_squares,
            "kannangara": self.kannangara,
            'mipr': self.mipr,
            'polynomial': self.polynomial,
            'predator': self.predator,
            "vicario_co": self.vicario_constrained,
            "vicario_nieap": self.vicario_nieap,
        }
        self.last_gold = np.nan if not recorded_compliance else recorded_compliance
        self.iipred_reconstructed_pressures = {}
        self.peeps = {}
        # Al-Rawas finds the expiratory time const via a regression on the
        # expiratory flow. If the residual is less than this value then
        # declare the exp const as unachievable.
        self.al_rawas_tol = kwargs.get('al_rawas_tol', .95)
        # Al-Rawas has another weird constant in his equation. This probably
        # varies from patient to patient in different circumstances and theres
        # probably no really good way of finding it besides iterating over all
        # possible indices and finding what works best for a specific breath. For speed
        # sake here, we just set it to a single number.
        self.al_rawas_idx = kwargs.get('al_rawas_idx', 15)
        # this is the number of iterations to run brunner's algo for
        self.brunner_iters = kwargs.get('brunner_iters', 2)
        # this is the auc threshold to determine if a breath is asynchronous or not
        # for the Kannangara algo
        self.kannangara_thresh = kwargs.get('kannangara_thresh', 0.05)
        # lourens tc to use. Options available are 25, 50, 75, 100
        self.lourens_tc_choice = kwargs.get('lourens_tc_choice', 50)
        # number of iters to run mipr for
        self.mipr_iters = kwargs.get('mipr_iters', 20)
        # This is a constant used in PREDATOR to determine how many breaths backward we
        # should be looking to make our approximation of the pressure (or flow) curve
        self.predator_n_breaths = kwargs.get('predator_n_breaths', 5)
        # time const algo to use. by default we use al-rawas because its fiddly constant is
        # relatively easy to pick. Brunner has a weird non-converging iters term, and lourens
        # you need to pick which time const you want to use, which can vary from patient to
        # patient and breath to breath.
        self.tc_algo = {
            'al_rawas': self.al_rawas_tau,
            'brunner': self.brunner,
            'lourens': self.lourens_tau
        }[kwargs.get('tc_algo', 'al_rawas')]
        # this is the m_index for vicario. This is another const that is supposed to
        # be found by sensitivity analysis, but here we just set to a const
        self.vicario_co_m_idx = kwargs.get('vicario_co_m_idx', 15)
        # because vicario constrained can sometimes fail miserably, filter vicario
        # breaths above this residual in vicario co
        self.vicario_co_residual = kwargs.get('vicario_co_residual', 200)
        self.filename = filename

    def _perform_least_squares(self, breath_idx, func):
        """
        Helper method for the least squares methods
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        pressure = breath['pressure']
        peep = self._get_median_peep(breath_idx)
        plat, compliance, res, K, residual = func(
            flow_l_s, pressure, bm.x0_index, breath['dt'], peep, bm.tvi/1000.0
        )
        return compliance

    def _perform_predator(self, pressures, breath_idx):
        """
        Convenience method for PREDATOR. Assumes you've already gathered your pressures.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow = np.array(breath['flow']) / 60
        dt = breath['dt']
        peep = self._get_median_peep(breath_idx)
        plat, compliance, res, K, residual = perform_predator_algo(pressures, flow, bm.x0_index, dt, peep, bm.tvi/1000.0)
        return compliance

    def _get_median_peep(self, breath_idx):
        # implement caching because we can have multiple algos looking for same peep
        if breath_idx in self.peeps:
            return self.peeps[breath_idx]

        min_idx = 0 if breath_idx-self.peeps_to_use < 0 else breath_idx-self.peeps_to_use
        peep = self.breath_metadata.iloc[min_idx:breath_idx+1].PEEP.median()
        self.peeps[breath_idx] = peep
        return peep

    def al_rawas(self, breath_idx):
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow_l_s = [v/60.0 for v in breath['flow']]
        tau, plat, comp, res = al_rawas_calcs(
            flow_l_s,
            breath['pressure'],
            bm.x0_index,
            breath['dt'],
            bm.PIP,
            self._get_median_peep(breath_idx),
            bm.tvi/1000.0,
            self.al_rawas_idx,
            self.al_rawas_tol
        )
        return comp

    def al_rawas_tau(self, breath_idx):
        """
        Get Al-Rawas tau time const

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        tvi = bm.tvi/1000.0
        return al_rawas_expiratory_const(flow_l_s, bm.x0_index, breath['dt'], tvi, self.al_rawas_tol)

    def brunner(self, breath_idx):
        """
        Get the brunner tau time constant.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        # brunner recommends using a MA filter to smooth the flow curve and perform approximations.
        # for our purpose currently we will not implement this. We can move to optimizations later.
        tau = brunner(bm.tve/1000, bm.eTime, abs(min(breath['flow'])/60), self.brunner_iters)
        return tau

    def exp_least_squares(self, breath_idx):
        """
        Perform least squares analysis on the expiratory flow waveform.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        return self._perform_least_squares(breath_idx, expiratory_least_squares)

    def howe_least_squares(self, breath_idx):
        """
        Perform least squares analysis using Howe's method of modifying the expiratory waveform

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        return self._perform_least_squares(breath_idx, howe_expiratory_least_squares)

    def iipr(self, breath_idx):
        """
        Perform IIPR for reconstructing pressure waveforms in volume control

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        peep = self._get_median_peep(breath_idx)
        com, resist, residual, code = perform_iipr_algo(flow, pressure, bm.x0_index, peep, breath['dt'])
        return com

    def iipredator(self, breath_idx):
        """
        Perform IIPR on a breath first, then perform PREDATOR. Follows Redmonds 2019 paper:

        Evaluation of model-based methods in estimating respiratory mechanics in the presence of variable patient effort

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        # we do not have enough breaths previously to perform PREDATOR
        if breath_idx < self.predator_n_breaths - 1:
            return np.nan

        min_idx = breath_idx-self.predator_n_breaths+1
        max_idx = breath_idx

        for i in range(min_idx, max_idx+1):
            if i not in self.iipred_reconstructed_pressures:
                tmp_breath = self.breath_data[i]
                tmp_bm = self.breath_metadata.iloc[i]
                tmp_flow = np.array(tmp_breath['flow']) / 60
                peep = self._get_median_peep(i)
                recon, code = perform_iipr_pressure_reconstruction(
                    tmp_flow, tmp_breath['pressure'], tmp_bm.x0_index, peep, tmp_breath['dt']
                )
                if code in [0, 5]:
                    self.iipred_reconstructed_pressures[i] = recon
                else:
                    self.iipred_reconstructed_pressures[i] = tmp_breath['pressure']

        # +1 makes sure to include current breath
        pressures = [self.iipred_reconstructed_pressures[i] for i in range(min_idx, max_idx+1)]
        return self._perform_predator(pressures, breath_idx)

    def insp_least_squares(self, breath_idx):
        """
        Perform standard least squares analysis on the inspiratory flow waveform.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        return self._perform_least_squares(breath_idx, inspiratory_least_squares)

    def kannangara(self, breath_idx):
        """
        Perform Kannangara's method of pressure control correction

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow = breath['flow']
        peep = self._get_median_peep(breath_idx)
        sols = kannangara(flow, breath['pressure'], bm.x0_index, self.kannangara_thresh)
        # return compliance only
        if sols[0]:
            return sols[0]
        return np.nan

    def lourens_tau(self, breath_idx):
        """
        Perform lourens time const and return RCfv25, ... RCfv100 depending on your choice

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow = np.array(breath['flow']) / 60
        tve = bm.tve / 1000
        tc_option = {25: 0, 50: 1, 75: 2, 100: 3}[self.lourens_tc_choice]
        return lourens_time_const(flow, tve, bm.x0_index, breath['dt'])[tc_option]

    def mipr(self, breath_idx):
        """
        Performs MIPR

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        peep = self._get_median_peep(breath_idx)
        comp, resist, residual, code = perform_mipr(flow, pressure, bm.x0_index, peep, self.mipr_iters)
        return comp

    def polynomial(self, breath_idx):
        """
        Performs Redmond's polynomial model

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        peep = self._get_median_peep(breath_idx)
        tvi = bm.tvi/1000
        comp, resist, resid, code = perform_polynomial_model(flow, pressure, bm.x0_index, peep, tvi)
        return comp

    def predator(self, breath_idx):
        """
        Perform PREDATOR algorithm.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        # we do not have enough breaths previously to perform PREDATOR
        if breath_idx < self.predator_n_breaths - 1:
            return np.nan
        # +1 makes sure to include current breath
        #
        # Don't save the predator reconstructed pressure to file because then predator would
        # start becoming a bit of a self fulfilling prophecy/algo
        pressures = [b['pressure'] for b in self.breath_data[breath_idx+1-self.predator_n_breaths:breath_idx+1]]
        return self._perform_predator(pressures, breath_idx)

    def vicario_constrained(self, breath_idx):
        """
        Implement the vicario constrained algorithm.

        this function will sometimes fail to optimize at all, and elastance, resistance, p_mus are
        just set to our initial guess. This situation has been improved by using a random initial
        guess but it does not solve the problem. A better solution in the future would be to
        have a mean/median elastance, residual, patient effort value and then just use that
        as our initial guess.

        However, as a result of these failures, if our optimization has a residual above a
        certain value then return nan.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow = breath['flow']
        dt = breath['dt']
        # need to divide by 60 here because we are expressing dx in terms of seconds. If we
        # didnt want to divide by 60 then we would need to express dx in terms of minutes
        vols = np.array([0] + [simps(flow[:i]/60, dx=dt) for i in range(2, len(flow)+1)])
        elas, res, p_mus, pao_preds, residual = perform_constrained_optimization(
            flow, vols, breath['pressure'], bm.x0_index, self.vicario_co_m_idx
        )
        if residual > self.vicario_co_residual:
            return np.nan
        return 1 / elas

    def vicario_nieap(self, breath_idx):
        """
        Perform vicario's method for NonInvasive Estimation of Alveolar Pressure (NIEAP).

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        pressure = breath['pressure']
        tvi = bm.tvi / 1000
        # Have option of using a variety of time constants, but lets just use al-rawas for now.
        # We can add configurability in the future.
        peep = self._get_median_peep(breath_idx)
        tau = self.tc_algo(breath_idx)
        if tau is not np.nan:
            plat, comp, res = vicario_nieap(flow_l_s, pressure, bm.x0_index, peep, tvi, tau)
            return comp
        return np.nan

    def analyze_breath(self, breath_idx):
        """
        Analyze a single breath with all algorithms that we have specified to use. Save results
        to some object so we can analyze them later.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow = np.array(breath['flow'])
        pressure = np.array(breath['pressure'])
        dt = breath['dt']
        tvi = bm.tvi
        peep = self._get_median_peep(breath_idx)
        found_plat, plat = calc_inspiratory_plateau(flow, pressure, dt)
        if found_plat:
            gold = (tvi / 1000.0) / (plat - peep)
            self.last_gold = gold
            breath_results = [gold]
        else:
            breath_results = [self.last_gold]

        for algo in self.algorithms_to_use:
            breath_results.append(self.algo_mapping[algo](breath_idx))
        return breath_results

    def analyze_file(self):
        # XXX need to incorporate extra data to differentiate whats within 30 min and 1hr of
        # the last plat
        #
        # XXX Also need to ensure that we're only using valid plats as well, and not any
        # plat possible
        for idx in range(len(self.breath_data)):
            breath_results = self.analyze_breath(idx)
            self.results.append(breath_results)
        self.results = pd.DataFrame(self.results, columns=self.results_cols)
        # XXX need to fix this so its compatible for use with a dataframe
        #return self.results_analysis()

    # XXX need to redo this function
    def results_analysis(self):
        """
        Calculate Median Average Deviation (MAD) between gold standard and a calculation
        and the MAD within a calculation
        """
        # should be {algo: [mad gld stnd, mad inter, breaths, mean_gld, [val1, val2, ...], [gld1, gld2, ...]], ...}
        analysis = {val: [0, 0, 0, 0, [], []] for val in self.algorithms_to_use}
        for bn in self.results:
            gld = self.results[bn]['gold_stnd_compliance']
            if gld is np.nan:
                continue
            for algo in self.algorithms_to_use:
                val = self.results[bn][algo]
                if val is np.nan:
                    continue
                analysis[algo][2] += 1
                analysis[algo][-2].append(val)
                analysis[algo][-1].append(gld)

        for algo in self.algorithms_to_use:
            vals = np.array(analysis[algo][-2])
            glds = np.array(analysis[algo][-1])
            median_val = np.median(vals)
            inter_mad = np.median(np.abs(vals - median_val))
            gld_mad = np.median(np.abs(vals - glds))
            analysis[algo][0] = gld_mad
            analysis[algo][1] = inter_mad
            analysis[algo][3] = np.mean(glds)
            # delete intermediate data for algo values and gold calcs
            del analysis[algo][-1]
            del analysis[algo][-1]
        return analysis
