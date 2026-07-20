# -*- coding: utf-8 -*-
"""Named dataset normalization profiles and legacy Bebop 1 constants.

The available profiles are ``bebop1``, ``bebop2_tau_0_06`` and
``bebop2_tau_0_03``. Use :func:`get_normalization_limits` instead of importing
one profile directly when the choice comes from configuration.
"""

import torch


BEBOP1_MIN = {'dx': -7.08528922e+00,
              'dy': -7.04967721e+00, 
              'dz': -5.81295321e+00, 
              'vx': -4.70970619e+00, 
              'vy': -4.78749992e+00, 
              'vz': -2.32903691e+00, 
              'phi': -1.38556128e+00, 
              'theta': -1.32969594e+00, 
              'psi': -4.10490676e+00, 
              'p': -1.56213680e+01, 
              'q': -1.09344636e+01, 
              'r': -4.31710720e+00, 
              'Mx_ext': -3.99990950e-02, 
              'My_ext': -3.99979472e-02, 
              'Mz_ext': -9.99980775e-03,  
              'omega_min': 5.00000000e+03,
              'distance_error': 0.0,
              'attitude_error': 0.0}
BEBOP1_MAX = {'dx': 6.94915898e+00,
              'dy': 7.00480089e+00, 
              'dz': 5.66367014e+00, 
              'vx': 4.73818446e+00, 
              'vy': 4.79300936e+00, 
              'vz': 2.19937277e+00, 
              'phi': 1.32117585e+00, 
              'theta': 1.39802603e+00, 
              'psi': 4.10320250e+00, 
              'p': 1.26680307e+01, 
              'q': 1.00665052e+01, 
              'r': 4.38410885e+00, 
              'Mx_ext': 3.99996276e-02, 
              'My_ext': 3.99993403e-02, 
              'Mz_ext': 9.99977873e-03,  
              'omega_max': 1.00000000e+04,
              'distance_error': 7.410003182390156,
              'attitude_error': 4.169492767070163}

BEBOP2_TAU_0_06_MIN = {
    'dx': -7.07153780,
    'dy': -7.05284256,
    'dz': -5.56030491,
    'vx': -4.63237996,
    'vy': -4.66044117,
    'vz': -1.65918198,
    'phi': -1.14164557,
    'theta': -0.99209529,
    'psi': -3.95187639,
    'p': -9.32495591,
    'q': -9.22844874,
    'r': -3.88378675,
    'Mx_ext': -0.03999910,
    'My_ext': -0.03999795,
    'Mz_ext': -0.00999981,
    'omega_min': 5000.01181729,
}
BEBOP2_TAU_0_06_MAX = {
    'dx': 6.99718001,
    'dy': 7.01489217,
    'dz': 5.54857129,
    'vx': 4.61399152,
    'vy': 4.73603565,
    'vz': 1.83613092,
    'phi': 1.05933083,
    'theta': 1.14382555,
    'psi': 3.90809746,
    'p': 9.40760500,
    'q': 8.90656303,
    'r': 4.01474406,
    'Mx_ext': 0.03999963,
    'My_ext': 0.03999934,
    'Mz_ext': 0.00999978,
    'omega_max': 9999.99372531,
}

BEBOP2_TAU_0_03_MIN = {
    'dx': -6.96744555,
    'dy': -7.04052501,
    'dz': -5.51240056,
    'vx': -6.44774352,
    'vy': -5.48342527,
    'vz': -3.11553576,
    'phi': -1.36970327,
    'theta': -0.95853628,
    'psi': -3.79619307,
    'p': -11.60305509,
    'q': -12.67025273,
    'r': -3.82729624,
    'Mx_ext': -0.03999910,
    'My_ext': -0.03999795,
    'Mz_ext': -0.00999981,
    'omega_min': 5000.01181729,
}
BEBOP2_TAU_0_03_MAX = {
    'dx': 7.01908439,
    'dy': 6.99986305,
    'dz': 5.42546494,
    'vx': 4.60531997,
    'vy': 4.74213529,
    'vz': 3.84449264,
    'phi': 1.12346143,
    'theta': 1.09485112,
    'psi': 3.86395705,
    'p': 11.12088733,
    'q': 13.11717113,
    'r': 4.03790416,
    'Mx_ext': 0.03999963,
    'My_ext': 0.03999934,
    'Mz_ext': 0.00999978,
    'omega_max': 9999.99999382,
}


def _add_derived_feature_limits(limits_min, limits_max):
    """Add bounds for norm-based features derived from component channels."""
    distance_axes = ('dx', 'dy', 'dz')
    attitude_axes = ('phi', 'theta', 'psi')
    limits_min['distance_error'] = 0.0
    limits_max['distance_error'] = sum(
        max(abs(limits_min[key]), abs(limits_max[key])) ** 2 for key in distance_axes
    ) ** 0.5
    limits_min['attitude_error'] = 0.0
    limits_max['attitude_error'] = sum(
        max(abs(limits_min[key]), abs(limits_max[key])) ** 2 for key in attitude_axes
    ) ** 0.5


_add_derived_feature_limits(BEBOP2_TAU_0_06_MIN, BEBOP2_TAU_0_06_MAX)
_add_derived_feature_limits(BEBOP2_TAU_0_03_MIN, BEBOP2_TAU_0_03_MAX)

NORMALIZATION_LIMITS = {
    'bebop1': (BEBOP1_MIN, BEBOP1_MAX),
    'bebop2_tau_0_06': (BEBOP2_TAU_0_06_MIN, BEBOP2_TAU_0_06_MAX),
    'bebop2_tau_0_03': (BEBOP2_TAU_0_03_MIN, BEBOP2_TAU_0_03_MAX),
}

_NORMALIZATION_ALIASES = {
    'bebop2': 'bebop2_tau_0_06',
    'bebop2_tau_0.06': 'bebop2_tau_0_06',
    'bebop2_tau_0.03': 'bebop2_tau_0_03',
    # Legacy module names accepted at CLI boundaries. The old modules no longer
    # need to exist; all values still come from this file.
    'utils.normalization_limits': 'bebop1',
    'lnn_behavioural_cloning_quadrotor.utils.normalization_limits': 'bebop1',
    'utils.normalization_limits_bebop2': 'bebop2_tau_0_06',
    'lnn_behavioural_cloning_quadrotor.utils.normalization_limits_bebop2': 'bebop2_tau_0_06',
    'utils.normalization_limits_bebop2_new': 'bebop2_tau_0_06',
    'lnn_behavioural_cloning_quadrotor.utils.normalization_limits_bebop2_new': 'bebop2_tau_0_06',
    'utils.normalization_limits_bebop2_tau_0_03': 'bebop2_tau_0_03',
    'lnn_behavioural_cloning_quadrotor.utils.normalization_limits_bebop2_tau_0_03': 'bebop2_tau_0_03',
}


def resolve_normalization_profile(name='bebop1'):
    """Resolve aliases and return the canonical normalization profile name."""
    profile = str(name or 'bebop1').strip().lower()
    profile = _NORMALIZATION_ALIASES.get(profile, profile)
    if profile not in NORMALIZATION_LIMITS:
        available = ', '.join(NORMALIZATION_LIMITS)
        raise ValueError(
            f"Unknown normalization_limits profile {name!r}. Available profiles: {available}."
        )
    return profile


def get_normalization_limits(name='bebop1'):
    """Return the min/max dictionaries for a named normalization profile."""
    return NORMALIZATION_LIMITS[resolve_normalization_profile(name)]


# Backwards-compatible names: historically this module represented Bebop 1.
GLOBAL_MIN = BEBOP1_MIN
GLOBAL_MAX = BEBOP1_MAX
G = torch.tensor(9.81, dtype=torch.float32)
MASS = torch.tensor(0.389, dtype=torch.float32)
IXX = torch.tensor(0.000906, dtype=torch.float32)
IYY = torch.tensor(0.001242, dtype=torch.float32)
IZZ = torch.tensor(0.002054, dtype=torch.float32)
KX = torch.tensor(1.07933887e-05, dtype=torch.float32)
KY = torch.tensor(9.65250793e-06, dtype=torch.float32)
KZ = torch.tensor(2.7862899e-05, dtype=torch.float32)
KOMEGA = torch.tensor(4.36301076e-08, dtype=torch.float32)
KH = torch.tensor(0.06255013, dtype=torch.float32)
KP = torch.tensor(1.4119331e-09, dtype=torch.float32)
KPV = torch.tensor(-0.00797102, dtype=torch.float32)
KQ = torch.tensor(1.21601884e-09, dtype=torch.float32)
KQV = torch.tensor(0.01292637, dtype=torch.float32)
KR1 = torch.tensor(2.57035545e-06, dtype=torch.float32)
KR2 = torch.tensor(4.10923364e-07, dtype=torch.float32)
KRR = torch.tensor(0.00081293, dtype=torch.float32)
OMEGA_MAX = torch.tensor(1.00000000e+04, dtype=torch.float32)
OMEGA_MIN = torch.tensor(5.00000000e+03, dtype=torch.float32)
TAU = torch.tensor(0.06, dtype=torch.float32)
