from .base import BaseDataset
from .registry import DatasetRegistry, register_dataset, get_dataset

# Import all datasets to register them
from .abc import ABC
from .anphy import ANPHY
from .apoe import APOE
from .apples import APPLES
from .bestair import BESTAIR
from .cap import CAP
from .cfs import CFS
from .cps import CPS
from .dcsm import DCSM
from .dodh import DODH
from .dodo import DODO
from .dreamt import DREAMT
from .fdcsr import FDCSR
from .hmc import HMC
from .homepap import HOMEPAP
from .isruc import ISRUC
from .mesa import MESA
from .mitbih import MITBIH
from .mnc import MNC
from .mros import MROS
from .msp import MSP
from .mwt import MWT
from .nchsdb import NCHSDB
from .physio2018 import Physio2018
from .shhs import SHHS
from .sleepedf2018 import SleepEDF2018
from .sleepbrl import SLEEPBRL
from .sof import SOF
from .stages  import STAGES
from .ucddb import UCDDB
from .wsc import WSC

# __all__ = [
#     "ABC",
#     "APOE",
#     "APPLES",
#     "BESTAIR",
#     "BaseDataset",
#     "CFS",
#     "CPS",
#     "DCSM",
#     "DODH",
#     "DODO",
#     "DREAMT",
#     "DatasetRegistry",
#     "FDCSR",
#     "HMC",
#     "HOMEPAP",
#     "ISRUC",
#     "MESA",
#     "MITBIH",
#     "MNC",
#     "MROS",
#     "MSP",
#     "Physio2018",
#     "SHHS",
#     "SleepEDF2018",
#     "SOF",
#     "WSC",
#     "get_",
#     "register_dataset",
# ]
