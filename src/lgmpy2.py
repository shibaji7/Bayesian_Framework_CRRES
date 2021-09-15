"""lgmpy2.py: Module is used to convert CDMAG to GSM coordinates using python2.7 """

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, Chakraborty"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import pandas as pd
import numpy as np
from lgmpy import magcoords
import aacgmv2

import os
import sys

def convert_cdmag_gsm(row, frm="CDMAG", to="GSM"):
    """
    All the data in CDMAG from the web database
    are converted GSM coordinates
    """
    a = [row["CDMAG_R"], row["CDMAG_MLAT"], row["CDMAG_MLON"]]
    x = magcoords.coordTrans(a, row["epoch"], frm, to)
    row["R"], row["MLAT"], row["MLON"] = x[0], x[1], x[2]
    row["MLT"] = aacgmv2.convert_mlt([x[2]], row["epoch"], m2a=False)[0]
    return row

f = sys.argv[1]
if os.path.exists(f):
    o = pd.read_csv(f, parse_dates=["epoch"])
    o = o.apply(convert_cdmag_gsm, axis=1)
    keys = ["epoch", "SAT", "L", "Lstar", "R", "MLAT", "MLON", "MLT", "CDMAG_R", "CDMAG_MLAT", "CDMAG_MLON", "CDMAG_MLT", "B(pT)"] 
    o[keys].to_csv(f, index=False, header=True, float_format="%.3f")
else: print(" File not exists - ", f)