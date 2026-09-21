"""
Visualization and plotting tools for Stokes flow and reduced order models.
"""

from .plotting_stokes import plot_stokes_solution, analyze_bases_variation_stokes
from .visualization import svdplot

__all__ = [
    "plot_stokes_solution",
    "analyze_bases_variation_stokes",
    "svdplot"
]