"""get_data.py: Module is used to implement download data and implement an object to hold the cdf object"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2021, Chakraborty"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import os
import cdflib
import requests
import datetime as dt
import shutil


import pandas as pd
from cdflib.epochs import CDFepoch

class CDFLoader(object):
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2", "ver": "1.4.5"}, baseUrl="http://emfisis.physics.uiowa.edu/Flight/", 
                 localDir="tmp/EMFISIS/", fname="rbsp-{sc}_WFR-spectral-matrix-diagonal_emfisis-{lev}_{date}_v{ver}.cdf"):
        """
        Download CDF files from the server.
        
        Parameters:
        -----------
        dates = List of datetime
        params = Data parameters
        baseUrl = Base URL to invoke
        """
        self.dates = dates
        self.params = params
        self.baseUrl = baseUrl
        self.files = []
        self.file_objects = []
        self.localDir = localDir
        self.fname_scm = fname
        return
    
    def fetch(self):
        """ Fetch data from remote and hold the files """
        for date in self.dates:
            _dir_ = self.localDir + date.strftime("%Y%m%d") + "/"
            if not os.path.exists(_dir_): os.makedirs(_dir_)
            fname = self.fname_scm.format(sc=self.params["sc"], lev=self.params["lev"], date=date.strftime("%Y%m%d"), 
                                          ver=self.params["ver"])
            floc = _dir_ + fname
            print(fname)
            url = self.baseUrl + "RBSP-{sc}/{lev}/{year}/{month}/{day}/".format(sc=self.params["sc"].upper(), lev=self.params["lev"],
                   year=date.year, month="%02d"%date.month, day="%02d"%date.day) + fname
            if not os.path.exists(floc):
                print(" URL -", url)
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(floc, "wb") as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                        self.files.append(floc)
                        self.file_objects.append(cdflib.CDF(floc))
            else: 
                print(" Loading from - ", floc)
                self.files.append(floc)
                self.file_objects.append(cdflib.CDF(floc))
        return self
    
    def get_dataset(self, keys=["BuBu", "BvBv", "BwBw", "EuEu", "EvEv", "EwEw"], WFR_file_id=0):
        """ Convert the raw data to dict format """
        self.epoch = None
        o = {"Epoch": []}
        for f in self.file_objects:
            dates = [dt.datetime(i[0], i[1], i[2], i[3], i[4], i[5]) for i in CDFepoch.breakdown(f.varget("Epoch"))]
            o["Epoch"].extend(dates)
            for key in keys:
                if key not in o.keys(): o[key] = f.varget(key)
                else: o[key] = np.concatenate(o[key], f.varget(key))
        o["WFR"] = self.get_WFR_info(WFR_file_id)
        self.epoch = o["Epoch"]
        return o
    
    def get_WFR_info(self, file_id=0):
        """ Get WFR bin and frequency informations """
        self.WFR = {}
        self.WFR["bins"] = self.file_objects[file_id].varget("WFR_bins").ravel()
        self.WFR["bandwidth"] = self.file_objects[file_id].varget("WFR_bandwidth").ravel()
        self.WFR["frequencies"] = self.file_objects[file_id].varget("WFR_frequencies").ravel()
        return self.WFR
    
if __name__ == "__main__":
    cl = CDFLoader([dt.datetime(2015,3,17)]).fetch()
    o = cl.get_dataset()
    print(cl.get_WFR_info())
    #print(cl.get_dataset()["BuBu"].shape)
    import plotlib
    fti = plotlib.FrequencyTimePlot(cl.epoch, cl.WFR, 3)
    fti.addParamPlot(o["BuBu"], title=r"$B_u^2$", xlabel="")
    fti.addParamPlot(o["BvBv"], title=r"$B_v^2$", xlabel="")
    fti.addParamPlot(o["BwBw"], title=r"$B_w^2$", xlabel="Time [UT]")
    fti.save("out.png")
    