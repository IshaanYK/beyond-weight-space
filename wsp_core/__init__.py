"""
wsp_core: Subspace-Projected & Self-Verifying Weight-Space Programming Toolkit.
Provides mathematical tools for reading, analyzing, projecting, and verifying
neural network weight modifications without gradient descent.
"""

from .inspector import WeightInspector, SpectralProfile
from .projector import NullSpaceProjector, ProjectionResult
from .verifier import VerificationOracle, VerificationResult, VerificationDecision

__version__ = "0.1.0"
__all__ = [
    "WeightInspector",
    "SpectralProfile",
    "NullSpaceProjector",
    "ProjectionResult",
    "VerificationOracle",
    "VerificationResult",
    "VerificationDecision",
]

