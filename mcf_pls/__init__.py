"""
mcf_pls - Pump-keyed multi-core fiber physical-layer encryption package.

Modules
-------
physics     : FiberParams, MCFGeometry, HexMCFGeometry, CMESimulator, MCFEncryption
nonlinear   : SPM/XPM helpers, saturable absorber, security metrics, QAM utils
adversaries : Neural Eve models (EveRNN, EveMLP, EveTransformer, EveHybridMLP)
evaluation  : Attack evaluators, dataset builders, observation models
benchmarks  : Full publication benchmark suite (called by run.py)
utils       : I/O helpers, calibration writers, validation figures
"""
from mcf_pls.physics import FiberParams, MCFGeometry, HexMCFGeometry, CMESimulator, MCFEncryption
from mcf_pls.evaluation import TapObservationConfig
