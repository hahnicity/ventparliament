import numpy as np
import pandas as pd
from scipy.integrate import simps
from ventmap.breath_meta import get_production_breath_meta
from ventmap.constants import META_HEADER
from ventmap.raw_utils import read_processed_file
from ventmap.SAM import calc_expiratory_plateau, calc_inspiratory_plateau

from parliament.other_calcs import al_rawas_calcs, brunner, least_squares_method, vicario_nieap
from parliament.predator import perform_predator_algo
from parliament.pressure_ctrl_correction import perform_algo as kannangara
from parliament.vicario_constrained import perform_constrained_optimization


class FileCalculations(object):
    def __init__(self, filename, algorithms_to_use, peeps_to_use, recorded_compliance=None, **kwargs):
        """
        Calculate lung compliance for an entire file using a variety of algorithms

        :param filename: filename to analyze
        :param algorithms_to_use: Algorithms you want to include in your analysis. Should
        be a list and consist of choices: 'vicario_co', 'kannangara', 'least_squares',
        'brunner', 'vicario_nieap', ' al_rawas'
        :param peeps_to_use: Number of PEEPs to use when we calculate a median
        :param recorded_compliance: Compliance pre-recorded for the file.
        """
        self.algorithms_to_use = algorithms_to_use
        self.peeps = []
        self.peeps_to_use = peeps_to_use
        self.results = []
        self.results_cols = ['gold_stnd_compliance'] + algorithms_to_use
        self.breath_data = list(read_processed_file(filename))
        self.breath_metadata = pd.DataFrame([
            get_production_breath_meta(breath) for breath in self.breath_data
        ], columns=META_HEADER)
        self.algo_mapping = {
            "vicario_co": self.vicario_constrained,  # functional
            "kannangara": self.kannangara,  # functional
            "least_squares": self.least_squares,  # functional
            "brunner": self.brunner,
            "vicario_nieap": self.vicario_nieap,
            "al_rawas": self.al_rawas,  # functional
            'predator': self.predator,  # functional
        }
        self.last_gold = np.nan if not recorded_compliance else recorded_compliance
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
        # This is a constant used in PREDATOR to determine how many breaths backward we
        # should be looking to make our approximation of the pressure (or flow) curve
        self.predator_n_breaths = kwargs.get('predator_n_breaths', 5)
        # this is the m_index for vicario. This is another const that is supposed to
        # be found by sensitivity analysis, but here we just set to a const
        self.vicario_co_m_idx = kwargs.get('vicario_co_m_idx', 15)
        # because vicario constrained can sometimes fail miserably, filter vicario
        # breaths above this residual in vicario co
        self.vicario_co_residual = kwargs.get('vicario_co_residual', 200)
        self.filename = filename

    def al_rawas(self, breath_idx):
        flow_l_s = [v/60.0 for v in self.flow]
        tau, plat, comp, res = al_rawas_calcs(
            flow_l_s,
            self.pressure,
            self.x0,
            self.dt,
            self.pip,
            self.peep,
            self.tvi/1000.0,
            self.al_rawas_idx,
            self.al_rawas_tol
        )
        return comp

    def brunner(self, breath_idx):
        # XXX
        pass

    def kannangara(self, breath_idx):
        # XXX I want the algos to basically pull the info they need, and not be reliant
        # on the analyze_breath function to do their work. This way we allow users to have
        # more programmatic control over the object in question
        sols = kannangara(self.flow, self.pressure, self.x0, self.peep, 0.05)
        # return compliance only
        if sols[0]:
            return sols[0]
        return np.nan

    def least_squares(self, breath_idx):
        """
        Perform standard least squares analysis on the inspiratory flow waveform.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        pressure = breath['pressure']
        plat, compliance, res, K, residual = least_squares_method(
            flow_l_s, pressure, bm.x0_index, breath['dt'], bm.PEEP, bm.tvi/1000.0
        )
        return compliance

    def predator(self, breath_idx):
        """
        Perform PREDATOR algorithm.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        # we do not have enough breaths previously to perform PREDATOR
        if breath_idx < self.predator_n_breaths - 1:
            return np.nan
        # +1 makes sure to include current breath
        breaths = self.breath_data[breath_idx+1-self.predator_n_breaths:breath_idx+1]
        pressure_data = [b['pressure'] for b in breaths]
        flow_l_s = np.array(breaths[-1]['flow']) / 60
        dt = breaths[-1]['dt']
        bm = self.breath_metadata.iloc[breath_idx]
        plat, compliance, res, K, residual = perform_predator_algo(pressure_data, flow_l_s, bm.x0_index, dt, bm.PEEP, bm.tvi/1000.0)
        return compliance

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
        elas, res, p_mus, pao_preds, residual = perform_constrained_optimization(
            self.flow, self.volumes, self.pressure, self.x0, self.vicario_co_m_idx
        )
        if residual > self.vicario_co_residual:
            return np.nan
        return 1 / elas

    def vicario_nieap(self, breath_idx):
        """
        XXX
        """
        # XXX ensure that tc is part of the breath_metadata information
        if tc is not np.nan:
            flow_l_s = self.flow / 60
            return vicario_nieap(flow_l_s, self.pressure, self.x0, self.peep, self.tvi/1000.0, self.tc)
        else:
            return np.nan

    def analyze_breath(self, breath_idx):
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata.iloc[breath_idx]
        rel_bn = breath['rel_bn']
        self.dt = breath['dt']
        self.e_time = bm.eTime
        # returns tvi and tve in terms of ml
        self.tvi = bm.tvi
        self.tve = bm.tve
        self.pip = bm.PIP
        self.peeps.append(bm.PEEP)
        self.peep = np.median(self.peeps[-self.peeps_to_use:])
        self.x0 = bm.x0_index
        self.flow = np.array(breath['flow'])
        self.pressure = np.array(breath['pressure'])
        # gets volumes in liters for entire breath
        #
        # need to divide by 60 here because we are expressing dx in terms of seconds. If we
        # didnt want to divide by 60 then we would need to express dx in terms of minutes
        self.volumes = np.array([0] + [simps(self.flow[:i]/60, dx=self.dt) for i in range(2, len(self.flow)+1)])
        # XXX needs method to find tau (exp time const)

        found_plat, plat = calc_inspiratory_plateau(self.flow, self.pressure, self.dt)
        if found_plat:
            gold = (self.tvi / 1000.0) / (plat - self.peep)
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
