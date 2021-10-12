"""models.py: Module is used to implement GPR and Keras-BNN with different kernels."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, Chakraborty"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import sys
sys.path.append("src/")
import get_data as gd
import datetime as dt
import argparse
from dateutil import parser as prs
from loguru import logger

class Model(object):
    """
    Generic model class that process data and holds other operations
    """
    
    def __init__(self, args):
        for k in vars(args).keys():
            setattr(self, p, vars(args)[p])
        self.load_data()
        self.input_keys = ["L", "Lstar", "AE", "MLAT", "MLT", "mod"]
        return

    def load_data(self):
        self.frame = pd.read_csv(self.fname, parse_dates=["epoch"])
        self.frame["mod"] = self.frame.epoch.apply(lambda x: x.minute + 60*x.hour)
        return
    
def modeling(args):
    logger.info(f"Start model for - {args.start}-{args.end}")
    logger.info(f"Data file - {args.fname}")
    if os.path.exists(args.fname): 
        if args.operation == "gpr": GPR(args)
        elif args.operation == "bnn": BNN(args)
        else: logger.error(f"Model does not exists - {args.operation}!")
    else: logger.error(f"Data file does not exists - {args.fname}!")
    return

if __name__ == "__main__":
    parser.add_argument("-o", "--operation", default="gpr", help="Model type")
    parser.add_argument("-s", "--start", default=dt.datetime(2012,10,1), help="Start date (default 2012-10-01)", 
            type=prs.parse)
    parser.add_argument("-e", "--end", default=dt.datetime(2012,10,31), help="End date (default 2012-10-31)", 
            type=prs.parse)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    args = parser.parse_args()
    logger.info(f"Simulation run using model.__main__")
    if args.verbose:
        logger.info("Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
    args.fname = "tmp/%s_%s.csv"%(args.start.strftime("%Y%m%d"), args.end.strftime("%Y%m%d"))
    modeling(args)