"""What the hand does: the posture latent, movement recordings, synthetic and real (NinaPro) motion."""
from .latent import LATENT_NAMES, N_LATENT, REST, JointMap
from .recording import Motion, example_motion

__all__ = ["LATENT_NAMES", "N_LATENT", "REST", "JointMap", "Motion", "example_motion"]
