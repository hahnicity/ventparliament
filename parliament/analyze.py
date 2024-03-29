from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import simps
from ventmap.breath_meta import get_production_breath_meta
from ventmap.constants import META_HEADER
from ventmap.raw_utils import PB840File, read_processed_file
from ventmap.SAM import calc_expiratory_plateau, calc_inspiratory_plateau

from parliament.howe_main import howe_expiratory_least_squares
from parliament.iipr import perform_iipr_algo, perform_iipr_pressure_reconstruction
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
    # friendly names for the algorithms
    algo_name_mapping = {
        'al_rawas': 'Al-Rawas',
        'al_rawas_ar': 'Al-Rawas',
        'al_rawas_bru': 'Al-Rawas w/ Brunner TC',
        'al_rawas_fuz': 'Al-Rawas w/ Fuzzy Clust TC',
        'al_rawas_ikd': 'Al-Rawas w/ Ikeda TC',
        'al_rawas_lren': 'Al-Rawas w/ Lourens TC',
        'al_rawas_vic': 'Al-Rawas w/ Vicarios TC',
        'al_rawas_wri': 'Al-Rawas w/ Wiriyaporn TC',
        'ft_insp_lstsq': 'Flow-targeted Inspiratory Least Squares',
        'howe_lstsq': "Howe Least Squares",
        'iimipr': 'IIMIPR',
        'iipr': 'IIPR',
        'iipredator': 'IIPredator',
        'kannangara': 'Kannangara Pressure Mode Correction',
        'major': "Major",
        'mipr': 'MIPR',
        'polynomial': 'Polynomial Method',
        'predator': 'PREDATOR',
        'pt_exp_lstsq': 'Pressure Targeted Expiratory Least Squares',
        'pt_insp_lstsq': 'Pressure Targeted Inspiratory Least Squares',
        'vicario_co': 'Vicario Constrained Optimization',
        'vicario_nieap': 'Vicario Non-Invasive Estimation of Alveolar Pressure',
        'vicario_nieap_ar': 'Vicario w/ Al-Rawas TC',
        'vicario_nieap_bru': 'Vicario w/ Brunner TC',
        'vicario_nieap_fuz': 'Vicario w/ Fuzzy Clust TC',
        'vicario_nieap_ikd': 'Vicario w/ Ikeda TC',
        'vicario_nieap_lren': 'Vicario w/ Lourens TC',
        'vicario_nieap_vic': 'Vicario',
        'vicario_nieap_wri': 'Vicario w/ Wiriyaporn TC',

    }
    shorthand_name_mapping = {
        'al_rawas': 'Al-Rawas',
        'al_rawas_ar': 'Al-Rawas',
        'al_rawas_bru': 'Al-Rawas w/ Brunner TC',
        'al_rawas_fuz': 'Al-Rawas w/ Fuzzy Clust TC',
        'al_rawas_ikd': 'Al-Rawas w/ Ikeda TC',
        'al_rawas_lren': 'Al-Rawas w/ Lourens TC',
        'al_rawas_vic': 'Al-Rawas w/ Vicarios TC',
        'al_rawas_wri': 'Al-Rawas w/ Wiriyaporn TC',
        'ft_insp_lstsq': 'Flow Insp Least Sq',
        'howe_lstsq': "Howe",
        'iimipr': 'IIMIPR',
        'iipr': 'IIPR',
        'iipredator': 'IIPredator',
        'kannangara': 'Kannangara',
        'major': "Major",
        'mipr': 'MIPR',
        'polynomial': 'Polynomial',
        'predator': 'PREDATOR',
        'pt_exp_lstsq': 'Pressure Exp Least Sq',
        'pt_insp_lstsq': 'Pressure Insp Least Sq',
        'vicario_co': 'Constrained Optim',
        'vicario_nieap': 'Vicario Non-Inv Estim',
        'vicario_nieap_ar': 'Vicario w/ Al-Rawas TC',
        'vicario_nieap_bru': 'Vicario w/ Brunner TC',
        'vicario_nieap_fuz': 'Vicario w/ Fuzzy Clust TC',
        'vicario_nieap_ikd': 'Vicario w/ Ikeda TC',
        'vicario_nieap_lren': 'Vicario w/ Lourens TC',
        'vicario_nieap_vic': 'Vicario',
        'vicario_nieap_wri': 'Vicario w/ Wiriyaporn TC',
    }
    # pt exp least squares algos are available for pc/prvc because pressure
    # should theoretically operate similarly between different modes during
    # expiration
    algos_unavailable_for_pc_prvc = [
        'iimipr', 'iipr', 'iipredator', 'mipr', 'predator', 'major',
        'polynomial', 'pt_insp_lstsq',
    ]
    algos_unavailable_for_vc = ['kannangara', 'ft_insp_lstsq']
    # XXX leave this off for now because we have decided not to add this
    # analysis in the current paper
    algos_with_tc = ['al_rawas', 'vicario_nieap']
    #algos_with_tc = []

    def __init__(self, patient, filename, algorithms_to_use, peeps_to_use, extra_breath_info, recorded_compliance=None, recorded_plat=None, no_algo_restrict=False, **kwargs):
        """
        Calculate lung compliance for an entire file using a variety of algorithms

        :param patient: patient id
        :param filename: filename to analyze
        :param algorithms_to_use: Algorithms you want to include in your analysis. To use all of
        the available algos specific 'all'. To use specific ones, this argument should
        be a list and consist of choices available in the self.algo_mapping variable,
        :param peeps_to_use: Number of PEEPs to use when we calculate a median
        :param extra_breath_info: DataFrame of additional information like location of valid plat pressures for breath
        :param recorded_compliance: Compliance pre-recorded for the file. Relevant for CVC data
        :param recorded_plat: Plateau Pressure pre-recorded for the file. Relevant for CVC data
        :param no_algo_restrict: Do not restrict algorithms to their developed modes. Run in all modes available
        """
        self.algo_mapping = {
            "al_rawas": self.al_rawas,
            # flow targeted inspiratory least squares
            'ft_insp_lstsq': self.ft_inspiratory_least_squares,
            # Howe's least squares
            "howe_lstsq": self.howe_least_squares,
            'iimipr': self.iimipr,
            'iipr': self.iipr,
            'iipredator': self.iipredator,
            "kannangara": self.kannangara,
            # mccay is way too slow
            #'mccay': self.mccay,
            'major': self.major,
            'mipr': self.mipr,
            'polynomial': self.polynomial,
            'predator': self.predator,
            # pressure targeted expiratory least squares
            "pt_exp_lstsq": self.exp_least_squares,
            # pressure targeted inspiratory least squares
            "pt_insp_lstsq": self.insp_least_squares,
            "vicario_co": self.vicario_constrained,
            # don't do this because its not described in the literature
            #"vicario_co_insp": self.vicario_constrained_insp_only,
            "vicario_nieap": self.vicario_nieap,
        }
        if algorithms_to_use == 'all' or 'all' in algorithms_to_use:
            self.algorithms_to_use = list(self.algo_mapping.keys())
        elif not isinstance(algorithms_to_use, list):
            raise Exception('algorithms_to_use var must either be a list of algos to use or "all"')
        else:
            self.algorithms_to_use = algorithms_to_use
        self.extra_breath_info = extra_breath_info
        self.filename = filename
        self.patient = patient
        self.peeps_to_use = peeps_to_use
        if filename.endswith('.raw.npy'):
            self.breath_data = list(read_processed_file(filename))
            self.dt = self.breath_data[0]['dt']
        elif filename.endswith('.csv'):
            self.breath_data = PB840File(open(filename, encoding='ascii', errors='ignore')).extract_raw(False)
            self.dt = 0.02
        self.no_algo_restrict = no_algo_restrict
        self.breath_metadata = [
            MetadataRow(get_production_breath_meta(breath)) for breath in self.breath_data
        ]
        self.recorded_gold = np.nan if not recorded_compliance else recorded_compliance
        self.recorded_plat = np.nan if not recorded_plat else recorded_plat
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
        # fuzzy clustering alpha
        self.fuzzy_clust_alpha = kwargs.get('fuzzy_clust_alpha', 0.8)
        # min number obs we must have for fuzzy clustering
        self.fuzzy_clust_min_obs = kwargs.get('fuzzy_clust_min_obs', 10)
        # path to pickled fuzzy clustering filename
        self.fuzzy_clust_cls = pd.read_pickle(kwargs.get('fuzzy_clust_path', str(Path(__file__).parent.joinpath('../dataset/processed_data/pickled_objs/gk.pkl').resolve())))
        # which cluster to extract time const from. by default is 1, which is last cluster in
        # expiratory phase and tends to have the longest exhalatory const.
        self.fuzzy_clust_which_clust = kwargs.get('fuzzy_clust_which_clust', 0)
        # this is the auc threshold to determine if a breath is asynchronous or not
        # for the Kannangara algo. This is determined as the fraction of difference
        # between the predicted flow waveform using least squares, and the actual
        # observed flow waveform.
        self.kannangara_thresh = kwargs.get('kannangara_thresh', 0.005)
        # lourens tc to use. Options available are any percentage between 1 and 100
        self.lourens_tc_choice = kwargs.get('lourens_tc_choice', 50)
        # number of iters to run mipr for
        self.mipr_iters = kwargs.get('mipr_iters', 20)
        # this is a float that corresponds to what kind of tolerance we are looking for
        # when we attempt look for where effort is being made on expiratory lim of
        # breath
        self.peep_tolerance = kwargs.get('peep_tolerance', .5)
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
            'fuzzy': self.fuzzy_clust_tau,
            'ikeda': self.ikeda_tau,
            'lourens': self.lourens_tau,
            'vicario': self.vicario_nieap_tau,
            'wiri': self.wiriyaporn_tau,
        }
        self.tvis = []
        self.tvis_len = kwargs.get('tvis_len', 3)
        if not kwargs.get('tc_algos'):
            self.tc_algos = ['al_rawas']
        elif kwargs.get('tc_algos') == ['all'] or kwargs.get('tc_algos') == 'all':
            self.tc_algos = list(self.tc_algo_mapping.keys())
        elif isinstance(kwargs.get('tc_algos'), list):
            self.tc_algos = kwargs.get('tc_algos')
        tc_algo_prefixes = {
            'al_rawas': 'ar',
            'brunner': 'bru',
            'fuzzy': 'fuz',
            'lourens': 'lren',
            'ikeda': 'ikd',
            'vicario': 'vic',
            'wiri': 'wri',
        }
        # breath file ventmodes so we dont have to keep recalculating
        self.ventmodes = dict()
        # this is the m_index for vicario. This is another const that is supposed to
        # be found by sensitivity analysis, but here we just set to a const
        self.vicario_co_m_idx = kwargs.get('vicario_co_m_idx', 15)
        # because vicario constrained can sometimes fail miserably, filter vicario
        # breaths above this residual in vicario co
        self.vicario_co_residual = kwargs.get('vicario_co_residual', 200)

        # setup results data list
        self.results = []
        self.results_cols = [
            'patient', 'rel_bn', 'abs_bs', 'gold_stnd_compliance', 'ventmode', 'dta',
            'bsa', 'fa', 'fa_loc', 'static_dca', 'dyn_dca', 'dyn_dca_timing', 'artifact',
            'early_efforting', 'insp_efforting', 'exp_efforting', 'peep', 'tvi', 'p_plat',
            'p_driving', 'dtw',
        ]
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

    def _get_breath_peep_from_ei_data(self, rel_bn, breath, breath_idx):
        ei_row = self.extra_breath_info[self.extra_breath_info.rel_bn == rel_bn].iloc[0]
        ventmode = ei_row.ventmode
        per_breath_peep = np.mean(breath['pressure'][-5:])

        if ventmode == 'prvc':
            peep = per_breath_peep
        else:
            min_idx = 0 if breath_idx-self.peeps_to_use < 0 else breath_idx-self.peeps_to_use
            peep = np.median([row.PEEP for row in self.breath_metadata[min_idx:breath_idx+1]])
        return peep

    def _get_breath_peep(self, breath_idx):
        """
        Get PEEP for the breath. Generally uses a median algorithm to find this,
        but for PRVC it just utilizes the current breath.

        Theres an argument to be made that you can balance when to use median PEEP
        versus when to use per-breath PEEP. Then there's additional circumstances
        when there's too much asynchrony and median peep is wrecked there. So you
        would need to use the last calculated PEEP at a stable time.

        It's totally possible that a decent algorithm would be a set it and forget
        it until you can't kind of idea. Where you set the PEEP, and then only revisit
        it if the calculations are too off for too long. But of course the details on
        this get equally tricky.
        """
        # implement caching because we can have multiple algos looking for same peep
        if breath_idx in self.peeps:
            return self.peeps[breath_idx]

        breath = self.breath_data[breath_idx]
        rel_bn = breath['rel_bn']

        if len(self.extra_breath_info) > 0:
            peep = self._get_breath_peep_from_ei_data(rel_bn, breath, breath_idx)
        else:
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
        peep = self._get_breath_peep(breath_idx)
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
        peep = self._get_breath_peep(breath_idx)
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
        return np.array(tc_results)

    def al_rawas(self, breath_idx, tau):
        """
        Perform Al-Rawas' algorithm for finding compliance.

        :param breath_idx: relative index of the breath we want to analyze in our file.
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
            self._get_breath_peep(breath_idx),
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

    def check_for_early_efforting(self, breath_idx):
        """
        Check for early efforting in the breath by searching for the following condition:

        PEEP - P0 > 1 cm/H20

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        ventmode = self.ventmodes[breath_idx]
        # if we don't have enough breaths in our file to capture an informed PEEP
        # then we can't pretend that we know efforting is happening.
        if breath_idx < self.peeps_to_use and ventmode != 'prvc':
            return 0

        breath = self.breath_data[breath_idx]
        peep = self._get_breath_peep(breath_idx)
        pressure = breath['pressure']
        # looking through this, vent disconnects are the most common form
        # of CVC breath that messes this rule up. But then again, that's
        # not a big deal because we can just filter them. with the vd algo.
        if peep - pressure[0] >= 1:
            return 1
        else:
            return 0

    def check_for_inspiratory_efforting(self, breath_idx):
        """
        Check for following conditions:

        VC: flow asynchrony
        PC/PRVC: TVi variation > 10% for 3 breath rolling average.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        rel_bn = breath['rel_bn']
        ei_row = self.extra_breath_info[self.extra_breath_info.rel_bn == rel_bn].iloc[0]
        ventmode = self.ventmodes[breath_idx]
        if ventmode == 'vc' and ei_row.fa > 0:
            return 1
        elif ventmode == 'vc':
            return 0
        elif ventmode in ['pc', 'prvc'] and len(self.tvis) == 0:
            bm = self.breath_metadata[breath_idx]
            self.tvis.append(bm.tvi)
            return 0
        elif ventmode in ['pc', 'prvc']:
            if len(self.tvis) == self.tvis_len:
                self.tvis.pop(0)

            bm = self.breath_metadata[breath_idx]
            self.tvis.append(bm.tvi)
            diffs = abs(np.diff(np.array(list(combinations(self.tvis, 2)))))
            if (diffs > np.array(self.tvis)*.1).any():
                return 1
            return 0
        else:
            return 0

    def check_for_late_efforting(self, breath_idx):
        """
        Check for late efforting in the breath by searching for the following condition:

        Pressure +/- 1 from set PEEP during negative flow.

        This only counts for a certain time after the pressure reaches close to PEEP.
        Otherwise this condition would be true on every breath immediately after x0.

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        ventmode = self.ventmodes[breath_idx]
        # if we dont know peep, then dont bother looking at this
        if breath_idx < self.peeps_to_use and ventmode != 'prvc':
            return 0

        peep = self._get_breath_peep(breath_idx)
        pressure = np.array(self.breath_data[breath_idx]['pressure'])
        delta_thesh = self.peep_tolerance + peep
        bm = self.breath_metadata[breath_idx]
        x0_index = bm.x0_index

        # this could be due to transient artifact or asynchrony. In either case just
        # return nothing and let other asynchrony/artifact algos do their work.
        if len(pressure[x0_index:]) <= 10:
            return 0

        # the reason why x0_index+10 is here is because there can be recoil forces
        # acting upon the lung immediately during expiration.
        delta_idx = np.argmin(np.logical_not((pressure[x0_index+10:]<delta_thesh))) + x0_index+10
        # the peep algo actually works pretty well. The problems occur when peep is
        # changed, and then the median takes a bit of time to catch up.
        if (peep-1 > pressure[delta_idx:]).any() or (peep+1 < pressure[delta_idx:]).any():
            return 1
        return 0

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

    def fuzzy_clust_tau(self, breath_idx):
        """
        Get tau based on Babuska's fuzzy clustering algo

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        flow = np.array(breath['flow'])/60
        vols = self._calc_breath_volume(breath_idx)
        bm = self.breath_metadata[breath_idx]
        return fuzzy_clustering_time_const(
            flow, vols, bm.x0_index, self.fuzzy_clust_min_obs, self.fuzzy_clust_alpha, self.fuzzy_clust_cls
        )[self.fuzzy_clust_which_clust]

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
        peep = self._get_breath_peep(breath_idx)
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
        peep = self._get_breath_peep(breath_idx)
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
        peep = self._get_breath_peep(breath_idx)
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
        peep = self._get_breath_peep(breath_idx)
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

    def mipr(self, breath_idx):
        """
        Performs MIPR

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow = np.array(breath['flow']) / 60
        pressure = np.array(breath['pressure'])
        peep = self._get_breath_peep(breath_idx)
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
        peep = self._get_breath_peep(breath_idx)
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
        """
        if tau is np.nan:
            return np.nan
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        pressure = breath['pressure']
        tvi = bm.tvi / 1000
        peep = self._get_breath_peep(breath_idx)
        # Have option of using a variety of time constants, but lets just use al-rawas for now.
        # We can add configurability in the future.
        plat, comp, res = vicario_nieap(flow_l_s, pressure, bm.x0_index, peep, tvi, tau)
        return comp

    def vicario_nieap_tau(self, breath_idx):
        """
        Perform method of calculating tau as explained in Vicario's 2016
        paper "Noninvasive estimation of alveolar pressure."

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        vols = self._calc_breath_volume(breath_idx)
        flow_l_s = np.array(breath['flow']) / 60
        return vicario_nieap_tau(flow_l_s, breath['pressure'], vols, bm.x0_index)

    def wiriyaporn_tau(self, breath_idx):
        """
        Compute exp const using Wiriyaporn's method

        :param breath_idx: relative index of the breath we want to analyze in our file.
        """
        breath = self.breath_data[breath_idx]
        bm = self.breath_metadata[breath_idx]
        flow_l_s = np.array(breath['flow']) / 60
        try:
            return wiriyaporn_time_const_exp(flow_l_s, bm.x0_index, self.dt)
        except RuntimeError:  # exponential method fails
            return wiriyaporn_time_const_linear(flow_l_s, bm.x0_index, self.dt)

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
        peep = self._get_breath_peep(breath_idx)
        ventmode = ei_row.ventmode
        self.ventmodes[breath_idx] = ventmode

        # look for generic efforting
        early_efforting = self.check_for_early_efforting(breath_idx)
        insp_efforting = self.check_for_inspiratory_efforting(breath_idx)
        exp_efforting = self.check_for_late_efforting(breath_idx)

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
            # gold standard is formatted in ml/cm H2O
            gold = tvi / (plat - peep)
            # plats are forwarded in post-processing
            breath_results = [
                self.patient, rel_bn, abs_bs, gold, ventmode, ei_row.dta, ei_row.bsa, ei_row.fa,
                ei_row.fa_loc, ei_row.static_dca, ei_row.dyn_dca, ei_row.dyn_dca_timing,
                ei_row.artifact, early_efforting, insp_efforting, exp_efforting, peep,
                tvi, plat, plat-peep, ei_row.dtw,
            ]
        else:
            breath_results = [
                self.patient, rel_bn, abs_bs, self.recorded_gold, ventmode, ei_row.dta, ei_row.bsa,
                ei_row.fa, ei_row.fa_loc, ei_row.static_dca, ei_row.dyn_dca, ei_row.dyn_dca_timing,
                ei_row.artifact, early_efforting, insp_efforting, exp_efforting, peep,
                tvi, self.recorded_plat, self.recorded_plat-peep, ei_row.dtw,
            ]

        for algo in self.algorithms_to_use:
            if ventmode in ['pc', 'prvc'] and algo in self.algos_unavailable_for_pc_prvc and not self.no_algo_restrict:
                breath_results.append(np.nan)
                continue
            elif ventmode == 'vc' and algo in self.algos_unavailable_for_vc and not self.no_algo_restrict:
                breath_results.append(np.nan)
                continue
            func = self.algo_mapping[algo]

            # make sure to format things in ml
            if algo in self.algos_with_tc:
                breath_results.extend(self.perform_algo_with_tc(breath_idx, func)*1000)
            else:
                breath_results.append(func(breath_idx)*1000)
        return breath_results

    def analyze_file(self):
        """
        Analyze a file under range of scrutiny for DTW/efforting/ventmode
        characteristics along with plats, previous gold standard findings, and
        all algorithms desired. Should only be used in cases where scientific
        results are desired.
        """
        for idx in range(len(self.breath_data)):
            breath_results = self.analyze_breath(idx)
            self.results.append(breath_results)
        self.results = pd.DataFrame(self.results, columns=self.results_cols)

    def quick_analyze_file(self):
        """
        Analyze a file just using the algorithms that we desire to run. Makes
        no assumption on ventmode restrictions or any other data characteristics
        """
        all_results = []
        for breath_idx in range(len(self.breath_data)):
            breath = self.breath_data[breath_idx]
            bm = self.breath_metadata[breath_idx]
            rel_bn = breath['rel_bn']
            flow = np.array(breath['flow'])
            pressure = np.array(breath['pressure'])
            abs_bs = breath['abs_bs']
            vent_bn = breath['vent_bn']
            tvi = bm.tvi
            peep = self._get_breath_peep(breath_idx)

            breath_results = [self.patient, rel_bn, vent_bn, abs_bs, peep, tvi]
            for algo in self.algorithms_to_use:
                func = self.algo_mapping[algo]

                # make sure to format things in ml
                if algo in self.algos_with_tc:
                    breath_results.extend(self.perform_algo_with_tc(breath_idx, func)*1000)
                else:
                    breath_results.append(func(breath_idx)*1000)
            all_results.append(breath_results)
        return pd.DataFrame(all_results, columns=['patient', 'rel_bn', 'vent_bn', 'abs_bs', 'peep', 'tvi']+self.algorithms_to_use)
