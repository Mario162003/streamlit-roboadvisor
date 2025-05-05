import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from datetime import datetime
import requests
import io
from pathlib import Path
from scipy.stats.mstats import winsorize
import pandas_datareader.data as web
warnings.filterwarnings('ignore')
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



import statsmodels.api as sm
from scipy import optimize
from scipy.optimize import minimize
from scipy.stats import norm

import ipywidgets as widgets
from ipywidgets import interact, IntSlider

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import math
import scipy.optimize as sco
from pandas.io.formats.style import Styler

#from pypfopt import risk_models
#from pypfopt.efficient_frontier import EfficientFrontier
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from fredapi import Fred