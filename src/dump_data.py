#!/usr/bin/env python

"""dump_data.py: dump data downloads all the ascii type files from different FTP repositories."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2019, Space@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import requests
import numpy as np
import pandas as pd
import glob
import datetime as dt
from random import randint
np.random.seed(0)


##############################################################################################
## Download 1m resolution solar wind omni data from NASA GSFC ftp server
##############################################################################################
def download_omni_dataset(dates, tmpdir="tmp/EMFISIS/"):
    base_uri = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min/omni_min%d%02d.asc"
    base_storage = tmpdir + "omni/"
    if not os.path.exists(base_storage): os.system("mkdir -p " + base_storage)
    base_storage += "%d%02d.asc"
    header = ["DATE","ID_IMF","ID_SW","nIMF","nSW","POINT_RATIO","TIME_SHIFT(sec)","RMS_TIME_SHIFT","RMS_PF_NORMAL",
              "TIME_BTW_OBS","Bfa","Bx","By_GSE","Bz_GSE","By_GSM","Bz_GSM","B_RMS","Bfa_RMS","V","Vx_GSE","Vy_GSE",
              "Vz_GSE","n","T","P_DYN","E","BETA","MACH_A","X_GSE","Y_GSE","Z_GSE","BSN_Xgse","BSN_Ygse","BSN_Zgse",
              "AE","AL","AU","SYM-D","SYM-H","ASY-D","ASY-H","PC-N","MACH_M"]
    if dates is not None:
        fpaths = []
        for date in dates:
            fpath = base_storage%(date.year,date.month)
            if fpath not in fpaths:
                url = base_uri%(date.year,date.month)
                response = requests.get(url)
                with open(fpath,"w") as f: f.write(response.text)
                fpaths.append(fpath)
        base_storage = tmpdir + "omni/"
        if not os.path.exists(base_storage): os.system("mkdir -p " + base_storage)
        csv_base = base_storage + "%s.csv"
        for fname in fpaths:
            csv_fname = csv_base%(fname.split("/")[-1].replace(".asc",""))
            print(fname, "-to-", csv_fname)
            with open(fname, "r") as f: lines = f.readlines()
            linevalues = []
            for i, line in enumerate(lines):
                values = list(filter(None, line.replace("\n", "").split(" ")))
                timestamp = dt.datetime(int(values[0]),1,1,int(values[2]),int(values[3])) + dt.timedelta(days=int(values[1])-1)
                linevalues.append([timestamp,int(values[4]),int(values[5]),int(values[6]),int(values[7]),
                                   int(values[8]),int(values[9]),int(values[10]),float(values[11]),int(values[12]),
                                   float(values[13]),float(values[14]),float(values[15]),float(values[16]),
                                   float(values[17]),float(values[18]),float(values[19]),float(values[20]),float(values[21]),
                                   float(values[22]),float(values[23]),float(values[24]),float(values[25]),float(values[26]),
                                   float(values[27]),float(values[28]),float(values[29]),float(values[30]),
                                   float(values[31]),float(values[32]),float(values[33]),float(values[34]),
                                   float(values[35]),float(values[36]),float(values[37]),float(values[38]),float(values[39]),
                                   float(values[40]),float(values[41]),float(values[42]),float(values[43]),
                                   float(values[44]),float(values[45])])
            _o = pd.DataFrame(linevalues, columns=header)
            _o.to_csv(csv_fname, header=True, index=False)
            os.system("rm "+fname)
    return

def get_omni_dataset(dates, tmpdir="tmp/EMFISIS/"):
    base_storage = tmpdir + "omni/%d%02d.csv"
    files, o = [], pd.DataFrame()
    for date in dates:
        fpath = base_storage%(date.year,date.month)
        if fpath not in files: files.append(fpath)
    for file in files:
        if os.path.exists(file): o = pd.concat([o, pd.read_csv(file, parse_dates=["DATE"])])
    return o

def fetch_Kp_data(dates, tmpdir="tmp/EMFISIS/"):
    base_storage = tmpdir + "omni/Kp_raw.csv"
    fname = tmpdir + "omni/Kp.csv"
    url = "http://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt"
    o = []
    if not os.path.exists(base_storage): os.system(f"wget -O {base_storage} {url}")
    if not os.path.exists(fname):
        with open(base_storage, "r") as f: lines = f.readlines()[30:]
        header = []
        for i, l in enumerate(lines):
            l = list(filter(None, l.replace("#", "").replace("\n", "").split(" ")))
            x = dict(
                date = dt.datetime.strptime(l[0]+l[1]+l[2], "%Y%m%d") + dt.timedelta(hours=float(l[3])),
                date_m = dt.datetime.strptime(l[0]+l[1]+l[2], "%Y%m%d") + dt.timedelta(hours=float(l[4])),
                Kp = float(l[7])
            )
            o.append(x)
        o = pd.DataFrame.from_records(o)
        o.to_csv(fname, index=False, header=True)
    else: o = pd.read_csv(fname, parse_dates=["date", "date_m"])
    dx = pd.DataFrame()
    for d in dates:
        x = o[(o.date>=d) & (o.date<=d+dt.timedelta(1))]
        x = x.set_index("date").resample("6s").ffill().reset_index()
        x.drop(x.tail(1).index,inplace=True)
        dx = pd.concat([dx, x])
    dx = dx.reset_index().drop(columns=["index"])
    return dx
    