from .background_noise import BackgroundNoiseAugmentor
from .pitch import PitchAugmentor
from .reverb import ReverbAugmentor
from .speed import SpeedAugmentor
from .volume import VolumeAugmentor
from .telephone import TelephoneEncodingAugmentor

# from . import utils

from .__version__ import (
    
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)

SUPPORTED_AUGMENTORS = ['background_noise', 'pitch', 'speed', 'volume', 'reverb']

import logging.config
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.NullHandler",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "art": {"handlers": ["default"]},
        "tests": {"handlers": ["test"], "level": "INFO", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)