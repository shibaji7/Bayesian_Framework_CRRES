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
from scipy import constants as C
from matplotlib.dates import date2num

import pandas as pd
from cdflib.epochs import CDFepoch
from bs4 import BeautifulSoup
import pickle

from paramiko import SSHClient
from scp import SCPClient


class Connection(object):
    
    def __init__(self, hostname="localhost", port = 1247, username="shibaji", password="virginia@1"):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.con = False
        self._conn_()
        return
    
    def _conn_(self):
        """ Create Conn """
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect(hostname=self.hostname, port = self.port, username=self.username, password=self.password)
        self.scp = SCPClient(self.ssh.get_transport())
        self.con = True
        return
    
    def _close_(self):
        """ Close connection """
        if self.con:
            self.scp.close()
            self.ssh.close()
        return

class CDFLoader(object):
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2"}, 
                 baseUrl="http://emfisis.physics.uiowa.edu/Flight/", 
                 localDir=None, file_kind=None, v=False):
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
        self.verbose = v
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
                if self.verbose: print(" URL -", url)
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(floc, "wb") as f:
                        shutil.copyfileobj(response.raw, f)
                        self.files["file_objects"].append(cdflib.CDF(floc))
            else: 
                if self.verbose: print(" Loading from - ", floc)
                self.files["file_objects"].append(cdflib.CDF(floc))
        return self
    
    def get_dataset_raw(self, keys, WFR_file_id=0):
        """ Convert the raw data to dict format """
        self.epoch = None
        o = {"Epoch": []}
        for f in self.files["file_objects"]:
            dates = [dt.datetime(i[0], i[1], i[2], i[3], i[4], i[5]) for i in CDFepoch.breakdown(f.varget("Epoch"))]
            o["Epoch"].extend(dates)
            if self.verbose: print(f.cdf_info())
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
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2"}, localDir="tmp/EMFISIS/", v=False):
        file_kind = "WFR-spectral-matrix-diagonal_emfisis"
        super().__init__(dates, params=params, localDir=localDir, file_kind=file_kind, v=v)
        return
    
    def get_dataset(self, keys=["BuBu", "BvBv", "BwBw"], WFR_file_id=0):
        return self.get_dataset_raw(keys, WFR_file_id)
    
    def get_WFR_info(self, file_id=0):
        """ Get WFR bin and frequency informations """
        self.WFR = {}
        self.WFR["bins"] = self.files["file_objects"][file_id].varget("WFR_bins").ravel()
        self.WFR["bandwidth"] = self.files["file_objects"][file_id].varget("WFR_bandwidth").ravel()
        self.WFR["frequencies"] = self.files["file_objects"][file_id].varget("WFR_frequencies").ravel()
        return self.WFR
    
class WaveformInfo(CDFLoader):
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2"}, localDir="tmp/EMFISIS/", v=False):
        file_kind = "WFR-waveform_emfisis"
        super().__init__(dates, params=params, localDir=localDir, file_kind=file_kind, v=v)
        return
    
    def get_dataset(self, keys=["BuSamples"]):
        return self.get_dataset_raw(keys, WFR_file_id=None)
    
class LocationInfo(object):
    """ Extract MagEphem data and store """
    
    def __init__(self, dates, params={"sc":"a"}, baseUrl="http://emfisis.physics.uiowa.edu/Flight/RBSP-{sc}/LANL/MagEphem/{year}/", 
                 localDir="tmp/EMFISIS/", fname="rbsp{scm}_def_MagEphem_OP77Q_{date}_v3.0.0.h5", v=False):
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
        self.files = [localDir + "%s/"%d.strftime("%Y%m%d") + self.fname.format(date=d.strftime("%Y%m%d"), scm=params["sc"])
                      for d in dates]
        self.urls = [self.url.format(year=d.strftime("%Y"), date=d.strftime("%Y%m%d"), scm=params["sc"], 
                                                  sc=params["sc"].upper()) for d in dates]
        self.localDir = localDir
        self.file_objects = []
        self.verbose = v
        return
    
    def fetch(self):
        """ Fetch data from remote and hold the files """
        for date, url, tfname in zip(self.dates, self.urls, self.files):
            _dir_ = self.localDir + date.strftime("%Y%m%d") + "/"
            if not os.path.exists(_dir_): os.makedirs(_dir_)
            if not os.path.exists(tfname):
                if self.verbose: print(" URL -", url)
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(tfname, "wb") as f:
                        response.raw.decode_content = True
                        shutil.copyfileobj(response.raw, f)
                        self.file_objects.append(h5py.File(tfname, "r"))
            else:
                if self.verbose: print(" Loading from - ", tfname)
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
        if self.verbose: print(" Describe location dataset-", "\n", st)
        return self
    
    def extract_data(self, keys=["L", "Lstar", "UTC", "Bmin_gsm", 
                                 "CDMAG_MLAT", "CDMAG_MLON", "CDMAG_MLT", 
                                 "CDMAG_R"], describe=False):
        if describe: self.describe()
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
    
class DownloadSC(object):
    
    def __init__(self, dates, params={"sc":"a", "lev":"L2"}, localDir="tmp/EMFISIS/", clean=True, v=False):
        self.dates = dates
        self.params = params
        self.localDir = localDir
        self.files = [localDir + "%s_%s.pickle"%(d.strftime("%Y%m%d"), params["sc"].upper()) for d in dates]
        self.cln = clean
        self.outs = {}
        self.units = [{"name": "pT", "value": 1e-12}]
        self.verbose = v
        return
    
    def reset_params(self, params):
        self.outs = {}
        self.params = params
        self.files = [self.localDir + "%s_%s.pickle"%(d.strftime("%Y%m%d"), params["sc"].upper()) for d in self.dates]
        return self
        
    def download(self):
        for d, f in zip(self.dates, self.files):
            if os.path.exists(f):
                if self.verbose: print(" Loading from - ", f)
                self.outs[d] = pickle.load(open( f, "rb" ) )
            else:
                self.li = LocationInfo([d], self.params, localDir=self.localDir)
                self.si = SpectralInfo([d], self.params, localDir=self.localDir)
                a = {}
                a["LocationInfo"] = self.li.fetch().extract_data()
                a["SpectralData"] = self.si.fetch().get_dataset()
                a["params"] = self.params
                with open(f, "wb") as h: pickle.dump(a, h, protocol=pickle.HIGHEST_PROTOCOL)
                self.outs[d] = a
                if self.cln: self.clean()
        return self
    
    def clean(self):
        self.li.clean()
        return
    
    def spectral_to_BField(self, flims=None):
        """
        For Hiss: flims = [{"max":2000, "min":100}]
        """
        def integrate_b(t, l, u):
            ox = t[(t.freq>=l) & (t.freq<=u)]
            m = 1e3*np.sqrt(np.trapz(ox.psd, x=ox.freq))
            return m
        
        con = Connection()
        for d in self.dates:
            fname = self.localDir + "%s_%s.csv"%(d.strftime("%Y%m%d"), self.params["sc"].upper())
            if os.path.exists(fname):
                if self.verbose: print(" Loading from - ", fname)
                o = pd.read_csv(fname, parse_dates=["epoch"])
            else:
                loc = self.outs[d]["LocationInfo"]
                fce = 1e-9*np.array(loc["Bmin_gsm"])[:, 3] * C.e / (2*C.pi * C.m_e)
                loc["L"], loc["Lstar"] = np.array(loc["L"]), np.array(loc["Lstar"])
                loc["L"][loc["L"] < 0], loc["Lstar"][loc["Lstar"] < 0] = np.nan, np.nan
                _l = pd.DataFrame()
                _l["L"], _l["Lstar"], _l["epoch"], _l["fce"] = np.nanmedian(loc["L"], axis=1),\
                                np.nanmedian(loc["Lstar"], axis=1), loc["UTC"], fce
                _l["CDMAG_MLAT"], _l["CDMAG_MLON"], _l["CDMAG_MLT"], _l["CDMAG_R"] = loc["CDMAG_MLAT"],\
                                loc["CDMAG_MLON"], loc["CDMAG_MLT"], loc["CDMAG_R"]
                _l = _l.set_index("epoch").resample("1s").interpolate().reset_index()
                spec = self.outs[d]["SpectralData"]
                b2_psd = spec["BuBu"] + spec["BvBv"] + spec["BwBw"]
                epoch = spec["Epoch"]
                freq = spec["WFR"]["frequencies"]
                o = pd.DataFrame()
                L, Lstar, CDMAG_MLAT, CDMAG_MLON, CDMAG_MLT, CDMAG_R, Fce = [], [], [], [], [], [], []
                Bl, Bu, B = [], [], []
                for i in range(b2_psd.shape[0]):
                    o["freq"] = np.copy(freq)
                    o["psd"] = np.copy(b2_psd[i,:])
                    b, bl, bu = 0., 0., 0.
                    if flims is None:
                        f = _l[_l.epoch==epoch[i]]
                        if len(f) > 0:
                            fce = f.fce.tolist()[0]
                            Fce.append(fce)
                            b += integrate_b(o, 0.1*fce, 0.9*fce)
                            bl += integrate_b(o, 0.1*fce, 0.5*fce)
                            bu += integrate_b(o, 0.5*fce, 0.9*fce)
                            L.append(f.L.tolist()[0])
                            Lstar.append(f.Lstar.tolist()[0])
                            CDMAG_MLAT.append(f.CDMAG_MLAT.tolist()[0])
                            CDMAG_MLON.append(f.CDMAG_MLON.tolist()[0])
                            CDMAG_MLT.append(f.CDMAG_MLT.tolist()[0])
                            CDMAG_R.append(f.CDMAG_R.tolist()[0])
                        else:
                            b, bl, bu = np.nan, np.nan, np.nan
                            Fce.append(np.nan)
                            L.append(np.nan)
                            Lstar.append(np.nan)
                            CDMAG_MLAT.append(np.nan)
                            CDMAG_MLON.append(np.nan)
                            CDMAG_MLT.append(np.nan)
                            CDMAG_R.append(np.nan)
                    else:
                        for flim in flims:
                            ox = o[(o.freq>=flim["min"]) & (o.freq<=flim["max"])]
                            b += 1e3*np.sqrt(np.trapz(ox.psd, x=ox.freq))
                            bl, bu = np.nan, np.nan
                    B.append(b)
                    Bl.append(bl)
                    Bu.append(bu)
                    o = o[0:0]
                o = pd.DataFrame()
                o["B(pT)"], o["Bl(pT)"], o["Bu(pT)"], o["epoch"], o["L"], o["Lstar"] = B, Bl, Bu, epoch, L, Lstar
                o["CDMAG_MLAT"], o["CDMAG_MLON"], o["CDMAG_MLT"], o["CDMAG_R"] = CDMAG_MLAT, CDMAG_MLON, CDMAG_MLT, CDMAG_R
                o["SAT"], o["Fce"] = self.params["sc"].upper(), Fce
                o.to_csv(fname, index=False, header=True)
                if self.verbose: print(" Local extraction done - ", d)
                # Run remote conversion in Python 2.7
                stdin, stdout, stderr = con.ssh.exec_command("cd CodeBase/Bayesian_Framework_CRRES/ "\
                                "\n python src/lgmpy2.py {f}".format(f=fname), get_pty=True)
                for line in iter(stdout.readline, ""):
                    if self.verbose: print(line, end="")
                o = pd.read_csv(fname, parse_dates=["epoch"])
            if self.verbose: print(o.head())        
        # End remote connections
        con._close_()
        return self
    
    def merge_satellites(self):
        sats = ["a", "b"]
        for d in self.dates:
            fname = self.localDir + "%s.csv"%(d.strftime("%Y%m%d"))
            u = pd.DataFrame()
            for sat in sats:
                f = self.localDir + "%s_%s.csv"%(d.strftime("%Y%m%d"), sat.upper())
                if os.path.exists(f): u = pd.concat([u, pd.read_csv(f)])
            u.to_csv(fname, index=False, header=True)
        return

def download_dataset(dates, localDir="tmp/EMFISIS/"):
    params = {"sc":"a", "lev":"L2"}
    d = DownloadSC(dates, params, localDir)
    d.download().spectral_to_BField().reset_params({"sc":"b", "lev":"L2"})
    d.download().spectral_to_BField()
    d.merge_satellites()
    return


class DataLoader(object):
    
    def __init__(self, dates, localDir="tmp/EMFISIS/", first_date_reset = True, v=False):
        self.dates = dates
        self.localDir = localDir
        self.verbose = v
        o = pd.DataFrame()
        for d in self.dates:
            f = self.localDir + d.strftime("%Y%m%d.csv")
            if os.path.exists(f): 
                if self.verbose: print(" Data file %s exists."%f)
                o = pd.concat([o, pd.read_csv(f, parse_dates=["epoch"])])
            else: 
                if self.verbose: print(" Data file %s does not exists."%f)
        if first_date_reset and o.epoch.tolist()[0] != dates[0]: 
            if self.verbose: print(" Reseting first date, row.")
            f = o.iloc[0]
            f["epoch"] = dates[0]
            f = pd.DataFrame([f.to_dict()])
            o = pd.concat([f, o]).reset_index(drop = True)
        o = o.reset_index()
        o.drop(columns=["index"], inplace=True)
        self.o = o.copy()
        return
    
    def _filter_(self, sc=None, dates=None, mlt=None, mlat=None):
        if sc is not None: self.o = self.o[self.o.SAT == sc]
        if (dates is not None) and (len(dates) == 2): self.o = self.o[(self.o.epoch >= dates[0]) & (self.o.epoch < dates[1])]
        return
    
if __name__ == "__main__":
    dates = [dt.datetime(2012,10,6) + dt.timedelta(i) for i in range(5)]
    download_dataset(dates)
    pass