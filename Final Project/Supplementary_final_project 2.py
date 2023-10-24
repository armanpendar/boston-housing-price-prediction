# -*- coding: utf-8 -*-
"""
In The Name Of God


@author: Ali Pilehvar Meibody
"""


#baraye bazi doostan , dade haye boston nemiomad bala
#mitonid kode zir ro run bezanid

import pandas as pd
import numpy as np


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)


data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


#in code mire mostaghim data ro download mikone
#x haro mirize dar zarfe data
#y haro mirize dar zarfe target

#moafagh bashid....