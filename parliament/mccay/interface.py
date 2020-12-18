"""
interface
~~~~~~~~~

Based off McCay's 2018 thesis:

Combined Respiratory Circuit-Computational Fluid Dynamics Modeling of
Partial Endotracheal Tube Obstruction
"""
import numpy as np

from parliament.mccay.LRC_pc_functions import pc_LRC, pc_LRC_iterate


class McCayInterface(object):
    def __init__(self, fr_range, recon_tolerance, average_c):
        """
        :param fr_range:
        :param recon_tolerance:
        :param average_c:
        """
        self.fr_range = fr_range
        self.recon_tolerance = recon_tolerance
        self.average_c = average_c
        self.results = {}

    def analyze_breath(self, breath, peep):
        flow = breath['flow']
        try:
            flr = pc_LRC_iterate(
                self.fr_range,
                self.recon_tolerance,
                breath['flow'],
                breath['pressure'],
                peep,
                average_c=self.average_c
            )
            p_recon, component_dict, interp_dict = pc_LRC(
                flr, breath['flow'], breath['pressure'], peep, average_c=self.average_c
            )
        except:
            interp_dict = {
                'R': [np.nan],
                'C_mean': np.nan,
            }
        self.results[breath['rel_bn']] = {
            'resistance': interp_dict['R'],
            'mean_compliance': interp_dict['C_mean'],
        }
