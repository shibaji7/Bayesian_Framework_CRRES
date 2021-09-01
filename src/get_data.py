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
import h5py
import json
import numpy as np

import pandas as pd
from cdflib.epochs import CDFepoch
from bs4 import BeautifulSoup
import pickle

class CDFLoader(object):
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2"}, 
                 baseUrl="http://emfisis.physics.uiowa.edu/Flight/", 
                 localDir=None, file_kind=None):
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
        self.localDir = localDir
        self.file_kind = file_kind
        self.files = {"locations": [],
                      "fnames": [],
                      "urls": [],
                      "file_objects": []}
        for d in dates:
            fnames = self.get_local_fname(d, self.get_urls(d, base=True))
            self.files["fnames"].extend(fnames)
            self.files["urls"].extend(self.get_urls(d))
            self.files["locations"].extend([self.get_local_floc(d)]*len(fnames))
        return
    
    def get_local_fname(self, d, url):
        """ Create local file name """
        fnames = []
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            response.raw.decode_content = True
            soup = BeautifulSoup(response.raw, "lxml")
            tags = [t.text for t in soup.find_all(["a"])]
            for t in tags:
                if self.file_kind in t: fnames.append(t)
        return fnames
    
    def get_local_floc(self, d):
        """ Create local file location """
        floc = self.localDir + d.strftime("%Y%m%d") + "/"
        return floc
    
    def get_urls(self, d, base=False):
        """ Create URLs """
        url = self.baseUrl + "RBSP-{sc}/{lev}/{year}/{month}/{day}/".format(sc=self.params["sc"].upper(), lev=self.params["lev"],
                   year=d.year, month="%02d"%d.month, day="%02d"%d.day)
        if not base:
            files = self.get_local_fname(d, url)
            urls = []
            for f in self.get_local_fname(d, url):
                urls.append(url + f)
            url = urls
        return url
    
    def fetch(self):
        """ Fetch data from remote and hold the files """
        for loc, fname, url in zip(self.files["locations"], self.files["fnames"], self.files["urls"]):
            if not os.path.exists(loc): os.makedirs(loc)
            floc = loc + fname
            if not os.path.exists(floc):
                print(" URL -", url)
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(floc, "wb") as f:
                        shutil.copyfileobj(response.raw, f)
                        self.files["file_objects"].append(cdflib.CDF(floc))
            else: 
                print(" Loading from - ", floc)
                self.files["file_objects"].append(cdflib.CDF(floc))
        return self
    
    def get_dataset_raw(self, keys, WFR_file_id=0, verbose=False):
        """ Convert the raw data to dict format """
        self.epoch = None
        o = {"Epoch": []}
        for f in self.files["file_objects"]:
            dates = [dt.datetime(i[0], i[1], i[2], i[3], i[4], i[5]) for i in CDFepoch.breakdown(f.varget("Epoch"))]
            o["Epoch"].extend(dates)
            if verbose: print(f.cdf_info())
            for key in keys:
                if key not in o.keys(): o[key] = f.varget(key)[:]
                else: o[key] = np.concatenate(o[key], f.varget(key)[:])
        if WFR_file_id is not None: o["WFR"] = self.get_WFR_info(WFR_file_id)
        self.epoch = o["Epoch"]
        return o
    
    def clean(self):
        """ Remove all files form local """
        for d in self.files["locations"]:
            shutil.rmtree(d)
        return

class SpectralInfo(CDFLoader):
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2"}, localDir="tmp/EMFISIS/"):
        file_kind = "WFR-spectral-matrix-diagonal_emfisis"
        super().__init__(dates, params=params, localDir=localDir, file_kind=file_kind)
        return
    
    def get_dataset(self, keys=["BuBu", "BvBv", "BwBw", "EuEu", "EvEv", "EwEw"], WFR_file_id=0):
        return self.get_dataset_raw(keys, WFR_file_id, verbose=False)
    
    def get_WFR_info(self, file_id=0):
        """ Get WFR bin and frequency informations """
        self.WFR = {}
        self.WFR["bins"] = self.files["file_objects"][file_id].varget("WFR_bins").ravel()
        self.WFR["bandwidth"] = self.files["file_objects"][file_id].varget("WFR_bandwidth").ravel()
        self.WFR["frequencies"] = self.files["file_objects"][file_id].varget("WFR_frequencies").ravel()
        return self.WFR
    
class WaveformInfo(CDFLoader):
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2"}, localDir="tmp/EMFISIS/"):
        file_kind = "WFR-waveform_emfisis"
        super().__init__(dates, params=params, localDir=localDir, file_kind=file_kind)
        return
    
    def get_dataset(self, keys=["BuSamples"]):
        return self.get_dataset_raw(keys, WFR_file_id=None, verbose=True)
    
class LocationInfo(object):
    """ Extract MagEphem data and store """
    
    def __init__(self, dates, params={"sc":"a"}, baseUrl="http://emfisis.physics.uiowa.edu/Flight/RBSP-{sc}/LANL/MagEphem/{year}/", 
                 localDir="tmp/EMFISIS/", fname="rbspa_def_MagEphem_OP77Q_{date}_v3.0.0.h5"):
        """
        Download CDF files from the server.
        
        Parameters:
        -----------
        dates = List of datetime
        params = Data parameters
        baseUrl = Base URL to invoke
        """
        self.dates = dates
        self.url = baseUrl + fname
        self.fname = fname
        self.files = [localDir + "%s/"%d.strftime("%Y%m%d") + self.fname.format(date=d.strftime("%Y%m%d")) for d in dates]
        self.urls = [self.url.format(year=d.strftime("%Y"), date=d.strftime("%Y%m%d"), 
                                                  sc=params["sc"].upper()) for d in dates]
        self.localDir = localDir
        self.file_objects = []
        return
    
    def fetch(self):
        """ Fetch data from remote and hold the files """
        for date, url, tfname in zip(self.dates, self.urls, self.files):
            _dir_ = self.localDir + date.strftime("%Y%m%d") + "/"
            if not os.path.exists(_dir_): os.makedirs(_dir_)
            if not os.path.exists(tfname):
                print(" URL -", url)
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(tfname, "wb") as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                        self.file_objects.append(h5py.File(tfname, "r"))
            else:
                print(" Loading from - ", tfname)
                self.file_objects.append(h5py.File(tfname, "r"))
        return self
    
    def describe(self, idx=0, fname="config/ephem_desc.json"):
        """ Describe any h5 MagEphem file """
        f = self.file_objects[0]
        with open(fname, "r") as _f: o = json.loads("\n".join(_f.readlines()))
        st = ""
        for k in f.keys():
            name = f.get(k).name[1:]
            if name in o.keys():
                desc = o[name]["DESCRIPTION"]
                st += (" " + name + ": " + desc + "\n")
            else:
                st += (" " + name + ": " + "\n")
        print(" Describe location dataset-", "\n", st)
        return self
    
    def extract_data(self, keys=["L", "Lstar", "UTC"]):
        self.describe()
        params = {}
        for d, f in zip(self.dates, self.file_objects):
            for key in f.keys():
                name = f.get(key).name[1:]
                if name in keys:
                    if name == "UTC": 
                        val = [d + dt.timedelta(hours=h) for h in f.get(key)[:]]
                        val[-1] = d + dt.timedelta(1)
                    else: val = (f.get(key)[:]).tolist()
                    params[name] = params[name].extend(val) if name in params.keys() else val
        return params
    
    def clean(self):
        """ Remove all files form local """
        for d in self.files:
            d = "/".join(d.split("/")[:-1])
            shutil.rmtree(d)
        return
    
class DownloadEMFISIS(object):
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2"}, localDir="tmp/EMFISIS/", clean=True):
        self.dates = dates
        self.params = params
        self.localDir = localDir
        self.files = [localDir + "%s.pickle"%d.strftime("%Y%m%d") for d in dates]
        self.cln = clean
        self.outs = {}
        return
    
    def download(self):
        for d, f in zip(self.dates, self.files):
            if os.path.exists(f):
                print(" Loading from - ", f)
                self.outs[d] = pickle.load(open( f, "rb" ) )
            else:
                self.li = LocationInfo([d], self.params, localDir=self.localDir)
                self.si = SpectralInfo([d], self.params, localDir=self.localDir)
                a = {}
                a["LocationInfo"] = self.li.fetch().extract_data()
                a["SpectralData"] = self.si.fetch().get_dataset()
                with open(f, "wb") as h: pickle.dump(a, h, protocol=pickle.HIGHEST_PROTOCOL)
                self.outs[d] = a
                if self.cln: self.clean()
        return self
    
    def clean(self):
        self.li.clean()
        return
    
    def spectral_to_BField(self, d=None, flim=[100,2000]):
        if d == None: d = self.dates[0]
        loc = self.outs[d]["LocationInfo"]
        loc["L"], loc["Lstar"] = np.array(loc["L"]), np.array(loc["Lstar"])
        loc["L"][loc["L"] < 0], loc["Lstar"][loc["Lstar"] < 0] = np.nan, np.nan
        _l = pd.DataFrame()
        _l["L"], _l["Lstar"], _l["epoch"] = np.nanmedian(loc["L"], axis=1), np.nanmedian(loc["Lstar"], axis=1), loc["UTC"]
        _l = _l.set_index("epoch").resample("3s").interpolate().reset_index()
        spec = self.outs[d]["SpectralData"]
        b2_psd = spec["BuBu"] + spec["BvBv"] + spec["BwBw"]
        epoch = spec["Epoch"]
        freq = spec["WFR"]["frequencies"]
        o = pd.DataFrame()
        B = []
        unit = {"name": "pT", "value": 1e-12}
        for i in range(b2_psd.shape[0]):
            o["freq"] = np.copy(freq)
            o["psd"] = np.copy(b2_psd[i,:])
            o = o[(o.freq>=flim[0]) & (o.freq<=flim[1])]
            b = 1e3*np.sqrt(np.trapz(o.psd, x=o.freq))
            B.append(b)
            o = o[0:0]
        _l = _l[(_l.epoch>=epoch[0]) & (_l.epoch<=epoch[-1])]
        o = pd.DataFrame()
        o["B(pT)"], o["epoch"], o["L"], o["Lstar"] = B, epoch, np.copy(_l.L)[::2], np.copy(_l.Lstar)[::2]
        print(o.head())
        return o, unit
    
    def create_wave_data(self):
        epoch, Bw, _L, _Ls, L, Ls, utc = [], [], [], [], [], [], []
        for o in self.outs:
            _l, _ls = np.array(o["LocationInfo"]["L"]), np.array(o["LocationInfo"]["Lstar"])
            _l[_l<0], _ls[_ls<0] = np.nan, np.nan
            _l, _ls = np.nanmax(_l, axis=1), np.nanmax(_ls, axis=1)
            _L.extend(_l)
            _Ls.extend(_ls)
            print(np.array(o["WaveData"]["BwSamples"]).max(), np.array(o["WaveData"]["BwSamples"]).min(),
                  np.array(o["LocationInfo"]["L"]).shape)
            Bw.extend((1000*np.array(o["WaveData"]["BwSamples"]).max(axis=1)).tolist())
            epoch.extend([e.replace(second=0) for e in o["WaveData"]["Epoch"]])
            utc.extend(o["LocationInfo"]["UTC"])
        L = [_L[utc.index(e)] for e in epoch]
        Ls = [_Ls[utc.index(e)] for e in epoch]
        return (epoch, L, Ls, Bw)

    

if __name__ == "__main__":
    dates = [dt.datetime(2012,10,6)+dt.timedelta(i) for i in range(1)]
    dwl = DownloadEMFISIS(dates)
    dwl.download()
    #dwl.spectral_to_BField()
    #d = dwl.create_wave_data()
    #from plotlib import WaveTimePlot as WTP
    #wti = WTP(d[0], 1)
    #wti.addParamPlot(d[0], d[2], d[3], ylabel="L*")
    #wti.save("out.png")
    pass