"""
Manage, preprocessing and data geeneration for the POD-CNF model.
"""

from .data_manage import DataPreprocessing
from .loader import LoadData, PODCNFdataset

__all__ = [
    "DataPreprocessing",
    "LoadData",
    "PODCNFdataset"
]

try:
    import fenics
    from .data_generation_stokes import ADR
    from .data_generation_linear_elasticity import FOMsampler
    
    __all__.extend([
        "ADR",
        "FOMsampler"
    ])
except ImportError:
    pass