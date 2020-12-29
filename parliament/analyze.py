import numpy as np
import pandas as pd
from scipy.integrate import simps
from ventmap.breath_meta import get_production_breath_meta
from ventmap.constants import META_HEADER
from ventmap.raw_utils import read_processed_file
from ventmap.SAM import calc_expiratory_plateau, calc_inspiratory_plateau

from parliament.iipr import perform_iipr_algo, perform_iipr_pressure_reconstruction
from parliament.mccay.interface import McCayInterface
from parliament.mipr import perform_mipr
from parliament.other_calcs import *
from parliament.polynomial_model import perform_polynomial_model
from parliament.predator import max_pool_pressure_reconstruction, perform_pressure_reconstruction as predator_reconstruction
from parliament.pressure_ctrl_correction import perform_algo as kannangara
from parliament.vicario_constrained import perform_constrained_optimization, perform_constrained_optimization_insp_lim_only


class MetadataRow(object):
    """
    Basically does the same thing I want with pd.Series, but its faster in doing its job
    """
    def __init__(self, data):
        self.data = data
        self.attr_to_idx = {attr: i for i, attr in enumerate(META_HEADER)}

    def __getattr__(self, attr):
        return self.data[self.attr_to_idx[attr]]


class FileCalculations(object):
    def __init__(self, filename, algorithms_to_use, peeps_to_use, extra_breath_info, recorded_compliance=None, **kwargs):
        """
        Calculate lung compliance for an entire file using a variety of algorithms

        :param filename: filename to analyze
        :param algorithms_to_use: Algorithms you want to include in your analysis. To use all of
        the available algos specific 'all'. To use specific ones, this argument should
        be a list and consist of choices available in the self.algo_mapping variable,
        :param peeps_to_use: Number of PEEPs to use when we calculate a median
        :param extra_breath_info: DataFrame of additional information like location of valid plat pressures for breath
        :param recorded_compliance: Compliance pre-recorded for the file. Relevant for CVC data
        """
        self.algo_mapping = {
            "al_rawas": self.al_rawas,
            'ft_insp_lstsq': self.ft_inspiratory_least_squares,
            "howe_lstsq": self.howe_least_squares,
            'iimipr': self.iimipr,
            'iipr': self.iipr,
            'iipredator': self.iipredator,
            "kannangara": self.kannangara,
            # XXX mccay is currently working from a software perspective but the results are off.
            #'mccay': self.mccay,
            'major': self.major,
            'mipr': self.mipr,
            'polynomial': self.polynomial,
            'predator': self.predator,
            "pt_exp_lstsq": self.exp_least_squares,
            "pt_insp_lstsq": self.insp_least_squares,
            "vicario_co": self.vicario_constrained,
            "vicario_co_insp": self.vicario_constrained_insp_only,
            "vicario_nieap": self.vicario_nieap,
        }
        self.algos_with_tc = ['al_rawas', 'vicario_nieap']
        if algorithms_to_use == 'all' or 'all' in algorithms_to_use:
            self.algorithms_to_use = list(self.algo_mapping.keys())
        elif not isinstance(algorithms_to_use, list):
            raise Exception('algorithms_to_use var must either be a list of algos to use or "all"')
        else:
            self.algorithms_to_use = algorithms_to_use
        # XXX add all pressure-predicting least squares methods here.
        self.algos_unavailable_for_pc_prvc = ['iimipr', 'iipr', 'iipredator', 'mipr', 'predator', 'major', 'polynomial']
        # XXX need to add flow-predicting least squares methods.
        self.algos_unavailable_for_vc = ['kannangara']
        self.extra_breath_info = extra_breath_info
        self.filename = filename
        if 'mccay' in self.algorithms_to_use:
            # XXX not sure where these params come from. But I used them awhile back
            self.mccay_interface = McCayInterface([.5, 15.], .01, True)
        self.peeps_to_use = peeps_to_use
        self.breath_data = list(read_processed_file(filename))
        if len(self.breath_data) == 0:
            raise Exception('ventmap found 0 breaths in file: {}! Is this an error?'.format(filename))
        self.dt = self.breath_data[0]['dt']
        self.breath_metadata = [
            MetadataRow(get_production_breath_meta(breath)) for breath in self.breath_data
        ]
        self.recorded_gold = np.nan if not recorded_compliance else recorded_compliance
        # XXX want to make this into a dict eventually
        self.reconstruction_methods = {
            'iipr',
            'kannangara',
            'mipr',
            'predator',
        }
        self.non_reconstruction_methods = set(self.algo_mapping.keys()).difference(self.reconstruction_methods)
        self.breath_volumes = {}
        self.ii_reconstructed_pressures = {}
        self.kan_reconstructed_flows = {}
        self.mipr_reconstructed_pressures = {}
        self.predator_reconstructed_pressures = {}
        self.peeps = {}
        # Al-Rawas finds the expiratory time const via a regression on the
        # expiratory flow. If the residual is less than this value then
        # declare the exp const as unachievable.
        self.al_rawas_tol = kwargs.get('al_rawas_tol', .95)
        # Al-Rawas has another weird constant in his equation. This probably
        # varies from patient to patient in different circumstances and theres
        # probably no really good way of finding it besides iterating over all
        # possible indices and finding what works best for a specific breath. So
        # we just specify to use the median of the compliance curve
        self.al_rawas_idx = kwargs.get('al_rawas_idx', 'median')
        # this is the number of iterations to run brunner's algo for
        self.brunner_iters = kwargs.get('brunner_iters', 2)
        # this is the auc threshold to determine if a breath is asynchronous or not
        # for the Kannangara algo. This is determined as the fraction of difference
        # between the predicted flow waveform using least squares, and the actual
        # observed flow waveform.
        self.kannangara_thresh = kwargs.get('kannangara_thresh', 0.005)
        # lourens tc to use. Options available are any percentage between 1 and 100
        self.lourens_tc_choice = kwargs.get('lourens_tc_choice', 75)
        # number of iters to run mipr for
        self.mipr_iters = kwargs.get('mipr_iters', 20)
        # This is a constant used in PREDATOR to determine how many breaths backward we
        # should be looking to make our approximation of the pressure (or flow) curve
        self.predator_n_breaths = kwargs.get('predator_n_breaths', 5)
        # time const algo to use. by default we use al-rawas because its fiddly constant is
        # relatively easy to pick and just specifies a tolerance at which the time constant
        # is not linear anymore. Brunner has a weird non-converging iters term, and lourens
        # you need to pick which time const you want to use, which can vary from patient to
        # patient and breath to breath.
        self.tc_algo_mapping = {
            'al_rawas': self.al_rawas_tau,
            'brunner': self.brunner,
            'ikeda': self.ikeda_tau,
            'lourens': self.lourens_tau
        }
        if not kwargs.get('tc_algos'):
            self.tc_algos = ['al_rawas']
        elif kwargs.get('tc_algos') == ['all']:
            self.tc_algos = list(self.tc_algo_mapping.keys())
        elif isinstance(kwargs.get('tc_algos'), list):
            self.tc_algos = kwargs.get('tc_algos')
        tc_algo_prefixes = {'al_rawas': 'ar', 'brunner': 'bru', 'lourens': 'lren'}
        # this is the m_index for vicario. This is another const that is supposed to
        # be found by sensitivity analysis, but here we just set to a const
        self.vicario_co_m_idx = kwargs.get('vicario_co_m_idx', 15)
        # because vicario constrained can sometimes fail miserably, filter vicario
        # breaths above this residual in vicario co
        self.vicario_co_residual = kwargs.get('vicario_co_residual', 200)

        # setup results data list
        self.results = []
        self.results_cols = ['rel_bn', 'abs_bs', 'gold_stnd_compliance', 'ventmode', 'dta', 'bsa', 'artifact']
        non_algo_cols = len(self.results_cols)
        for algo in self.algorithms_to_use:
            if algo in self.algos_with_tc:
                self.results_cols += [algo + '_' + tc_algo_prefixes[tc] for tc in self.tc_algos]
            else:
                self.results_cols += [algo]
        self.algos_used = self.results_cols[non_algo_cols:]

    def _calc_breath_volume(self, breath_idx):
        """
        Calculate and cache breath volume. Volumes will be expressed in L

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        if breath_idx in self.breath_volumes:
            return self.breath_volumes[breath_idx]
        else:
            breath = self.breath_data[breath_idx]
            # need to divide by 60 here because we are expressing dx in terms of seconds. If we
            # didnt want to divide by 60 then we would need to express dx in terms of minutes
            flow = np.array(breath['flow']) / 60
            vols = calc_volumes(flow, self.dt)
            self.breath_volumes[breath_idx] = vols
            return vols

    def _get_median_peep(self, breath_idx):
        # implement caching because we can have multiple algos looking for same peep
        if breath_idx in self.peeps:
            return self.peeps[breath_idx]

        min_idx = 0 if breath_idx-self.peeps_to_use < 0 else breath_idx-self.peeps_to_use
        peep = np.median([row.PEEP for row in self.breath_metadata[min_idx:breath_idx+1]])
        self.peeps[breath_idx] = peep
        return peep

    def _perform_least_squares(self, breath_idx, func):
        """
        Helper method for the least squares methods
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        peep = self._get_median_peep(breath_idx)
        vols = self._calc_breath_volume(breath_idx)
        plat, compliance, res, K, residual = func(
            flow_l_s, vols, pressure, bm.x0_index, self.dt, peep, bm.tvi/1000.0
        )
        return compliance

    def _perform_predator(self, breath_idx, reconstructed_pressure):
        """
        Convenience method for PREDATOR. Assumes you've already reconstructed your pressure.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        peep = self._get_median_peep(breath_idx)
        vols = self._calc_breath_volume(breath_idx)
        plat, comp, res, K, resid = pt_inspiratory_least_squares(
            flow, vols, reconstructed_pressure, bm.x0_index, self.dt, peep, bm.tvi
        )
        return comp

    def perform_algo_with_tc(self, breath_idx, algo):
        """
        Perform some algorithm that uses a time constant. Using this function allows us to
        use multiple time const methods with the same algorithm and aggregate their results
        in one location.
        For example if we were to run al_rawas we could do `self.perform_algo_with_tc(breath_idx, self.al_rawas)`

        :param breath_idx: relative index of the breath we want to analyze in our file.
        :returns list: [<compliance results with tc_1>, <compliance results with tc_2>, ...]
        """
        tc_results = []
        for tc_algo in self.tc_algos:
            tau = self.tc_algo_mapping[tc_algo](breath_idx)
            tc_results.append(algo(breath_idx, tau))
        return tc_results

    def al_rawas(self, breath_idx, tau):
        """
        Perform Al-Rawas' algorithm for finding compliance.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        :param tau: Expiratory time constant for our breath
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        vols = self._calc_breath_volume(breath_idx)
        tau, plat, comp, res = al_rawas_calcs(
            flow_l_s,
            vols,
            breath['pressure'],
            bm.x0_index,
            self.dt,
            bm.PIP,
            self._get_median_peep(breath_idx),
            bm.tvi/1000.0,
            self.al_rawas_idx,
            self.al_rawas_tol,
            tau=tau,
        )
        return comp

    def al_rawas_tau(self, breath_idx):
        """
        Get Al-Rawas tau time const

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        return al_rawas_expiratory_const(flow_l_s, bm.x0_index, self.dt, self.al_rawas_tol)

    def brunner(self, breath_idx):
        """
        Get the brunner tau time constant.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        # brunner recommends using a MA filter to smooth the flow curve and perform approximations.
        # for our purpose currently we will not implement this. We can move to optimizations later.
        tau = brunner(bm.tve/1000, bm.eTime, abs(min(breath['flow'])/60), self.brunner_iters)
        return tau

    def exp_least_squares(self, breath_idx):
        """
        Perform least squares analysis on the expiratory flow waveform.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        return self._perform_least_squares(breath_idx, pt_expiratory_least_squares)

    def ft_inspiratory_least_squares(self, breath_idx):
        """
        Perform flow-targeted least squares on inspiratory waveform

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        return self._perform_least_squares(breath_idx, ft_inspiratory_least_squares)

    def howe_least_squares(self, breath_idx):
        """
        Perform least squares analysis using Howe's method of modifying the expiratory waveform

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        return self._perform_least_squares(breath_idx, howe_expiratory_least_squares)

    def iimipr(self, breath_idx):
        """
        Run IIPR and then run MIPR on the reconstructed pressure.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        if breath_idx in self.ii_reconstructed_pressures:
            pressure = self.ii_reconstructed_pressures[breath_idx]
        else:
            pressure = self.iipr_pressure_reconstruction(breath_idx)
        flow = np.array(breath['flow']) / 60
        peep = self._get_median_peep(breath_idx)
        vols = self._calc_breath_volume(breath_idx)
        comp, resist, residual, code = perform_mipr(flow, vols, pressure, bm.x0_index, peep, self.dt, self.mipr_iters)
        return comp

    def iipr(self, breath_idx):
        """
        Perform IIPR for reconstructing pressure waveforms in volume control

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        # XXX if you want this to go faster on the full run you can probably saved cached reconstructions
        # and then pull them when you're re-running on a new algo.
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        peep = self._get_median_peep(breath_idx)
        vols = self._calc_breath_volume(breath_idx)
        com, resist, residual, code = perform_iipr_algo(flow, vols, pressure, bm.x0_index, peep, self.dt)
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
            if i not in self.ii_reconstructed_pressures:
                self.iipr_pressure_reconstruction(i)

        # +1 makes sure to include current breath
        pressures = [self.ii_reconstructed_pressures[i] for i in range(min_idx, max_idx+1)]
        recon = predator_reconstruction(pressures)
        return self._perform_predator(breath_idx, recon)

    def iipr_pressure_reconstruction(self, breath_idx):
        """
        Performs IIPR pressure reconstruction on a breath, caches it and returns the saved reconstruction
        to the caller. If reconstruction fails, then the original breath pressure will be saved/returned

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        peep = self._get_median_peep(breath_idx)
        pressure = breath['pressure']
        vols = self._calc_breath_volume(breath_idx)
        recon, code = perform_iipr_pressure_reconstruction(flow, vols, pressure, bm.x0_index, peep, self.dt)
        if code in [0, 5]:
            self.ii_reconstructed_pressures[breath_idx] = recon
            return recon
        else:
            self.ii_reconstructed_pressures[breath_idx] = pressure
            return pressure

    def ikeda_tau(self, breath_idx):
        """
        Get tau_e using Ikeda's method (very similar to Lourens method).

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        tvi = bm.tvi / 1000
        tve = bm.tve / 1000
        return ikeda_time_const(flow, tvi, tve, self.dt)

    def insp_least_squares(self, breath_idx):
        """
        Perform standard least squares analysis on the inspiratory flow waveform.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        return self._perform_least_squares(breath_idx, pt_inspiratory_least_squares)

    def kannangara(self, breath_idx):
        """
        Perform Kannangara's method of pressure control correction

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60.0
        peep = self._get_median_peep(breath_idx)
        vols = self._calc_breath_volume(breath_idx)
        sols = kannangara(flow, vols, breath['pressure'], bm.x0_index, peep, self.dt, self.kannangara_thresh)
        # return compliance only
        return sols[0]

    def lourens_tau(self, breath_idx):
        """
        Perform lourens time const and return RCfv25, ... RCfv100 depending on your choice

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        tve = bm.tve / 1000
        return lourens_time_const(flow, tve, bm.x0_index, self.dt, self.lourens_tc_choice)

    def major(self, breath_idx):
        """
        Perform the max pooling (Major's) method for pressure reconstruction. It's really just
        PREDATOR underneath. Use the same number of breaths that we normally use in PREDATOR.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        if breath_idx < self.predator_n_breaths - 1:
            return np.nan
        pressures = [b['pressure'] for b in self.breath_data[breath_idx+1-self.predator_n_breaths:breath_idx+1]]
        recon = max_pool_pressure_reconstruction(pressures)
        return self._perform_predator(breath_idx, recon)

    def mccay(self, breath_idx):
        """
        Perform McCay's method for finding compliance

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        peep = self._get_median_peep(breath_idx)
        self.mccay_interface.analyze_breath(breath, peep)
        return self.mccay_interface.results[breath['rel_bn']]['mean_compliance']

    def mipr(self, breath_idx):
        """
        Performs MIPR

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        peep = self._get_median_peep(breath_idx)
        vols = self._calc_breath_volume(breath_idx)
        comp, resist, residual, code = perform_mipr(flow, vols, pressure, bm.x0_index, peep, self.dt, self.mipr_iters)
        return comp

    def polynomial(self, breath_idx):
        """
        Performs Redmond's polynomial model

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        peep = self._get_median_peep(breath_idx)
        tvi = bm.tvi/1000
        vols = self._calc_breath_volume(breath_idx)
        comp, resist, resid, code = perform_polynomial_model(flow, vols, pressure, bm.x0_index, peep, tvi)
        return comp

    def predator(self, breath_idx):
        """
        Perform PREDATOR algorithm.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        if breath_idx < self.predator_n_breaths - 1:
            return np.nan
        recon = self.predator_pressure_reconstruction(breath_idx)
        return self._perform_predator(breath_idx, recon)

    def predator_pressure_reconstruction(self, breath_idx):
        # we do not have enough breaths previously to perform PREDATOR
        if breath_idx in self.predator_reconstructed_pressures:
            return self.predator_reconstructed_pressures[breath_idx]
        # +1 makes sure to include current breath
        pressures = [b['pressure'] for b in self.breath_data[breath_idx+1-self.predator_n_breaths:breath_idx+1]]
        breath = self.breath_data[breath_idx]
        recon = predator_reconstruction(pressures)
        self.predator_reconstructed_pressures[breath_idx] = recon
        return recon

    def vicario_constrained(self, breath_idx):
        """
        Implement the vicario constrained algorithm.

        This function will sometimes fail to optimize at all, and elastance, resistance, p_mus are
        just set to our initial guess. This situation has been improved by using a random initial
        guess but it does not solve the problem. A better solution in the future would be to
        have a mean/median elastance, residual, patient effort value and then just use that
        as our initial guess.

        However, as a result of these failures, if our optimization has a residual above a
        certain value then return nan.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        vols = self._calc_breath_volume(breath_idx)
        elas, res, p_mus, pao_preds, residual = perform_constrained_optimization(
            flow, vols, pressure, bm.x0_index, self.vicario_co_m_idx
        )
        if residual > self.vicario_co_residual:
            return np.nan
        return 1 / elas

    def vicario_constrained_insp_only(self, breath_idx):
        """
        Implement the vicario constrained algorithm but only on the inspiratory limb

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        vols = self._calc_breath_volume(breath_idx)
        elas, res, p_mus, pao_preds, residual = perform_constrained_optimization_insp_lim_only(
            flow, vols, pressure, bm.x0_index, self.vicario_co_m_idx
        )
        if residual > self.vicario_co_residual:
            return np.nan
        return 1 / elas

    def vicario_nieap(self, breath_idx, tau):
        """
        Perform vicario's method for NonInvasive Estimation of Alveolar Pressure (NIEAP).

        :param breath_idx: relative index of the breath we want to analyze in our file.
        :param tau: Expiratory time constant for our breath
        """
        if tau is np.nan:
            return np.nan
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        pressure = breath['pressure']
        tvi = bm.tvi / 1000
        # Have option of using a variety of time constants, but lets just use al-rawas for now.
        # We can add configurability in the future.
        peep = self._get_median_peep(breath_idx)
        plat, comp, res = vicario_nieap(flow_l_s, pressure, bm.x0_index, peep, tvi, tau)
        return comp

    def analyze_breath(self, breath_idx):
        """
        Analyze a single breath with all algorithms that we have specified to use. Save results
        to some object so we can analyze them later.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        rel_bn = breath['rel_bn']
        flow = np.array(breath['flow'])
        pressure = np.array(breath['pressure'])
        abs_bs = breath['abs_bs']
        ei_row = self.extra_breath_info[self.extra_breath_info.rel_bn == rel_bn].iloc[0]
        tvi = bm.tvi
        peep = self._get_median_peep(breath_idx)
        ventmode = ei_row.ventmode
        if ei_row.is_valid_plat == 1:
            # we can relax the search criteria a bit for plats here because we already know
            # where the proper plats are. Now the only thing we need is to calc the actual
            # plat pressure
            found_plat, plat = calc_inspiratory_plateau(flow, pressure, self.dt, min_time=0.4, flow_bound_any_or_all='all')
            if not found_plat:
                raise Exception(
                    'this breath is supposed to be a plat, but no plat was found! Check ' +
                    'params for calc_inspiratory_plateau.'
                )
            gold = (tvi / 1000.0) / (plat - peep)
            breath_results = [rel_bn, abs_bs, gold, ventmode, ei_row.dta, ei_row.bsa, ei_row.artifact]
        else:
            breath_results = [rel_bn, abs_bs, self.recorded_gold, ventmode, ei_row.dta, ei_row.bsa, ei_row.artifact]

        for algo in self.algorithms_to_use:
            if ventmode in ['pc', 'prvc'] and algo in self.algos_unavailable_for_pc_prvc:
                breath_results.append(np.nan)
                continue
            elif ventmode == 'vc' and algo in self.algos_unavailable_for_vc:
                breath_results.append(np.nan)
                continue

            func = self.algo_mapping[algo]
            if algo in self.algos_with_tc:
                breath_results.extend(self.perform_algo_with_tc(breath_idx, func))
            else:
                breath_results.append(func(breath_idx))
        return breath_results

    def analyze_file(self):
        for idx in range(len(self.breath_data)):
            breath_results = self.analyze_breath(idx)
            self.results.append(breath_results)
        self.results = pd.DataFrame(self.results, columns=self.results_cols)
