import pandas as pd

def load(cfg):
    #metadata load from csv 
    '''this file loads meta data from a csv.
    Takes the input '''
    meta = pd.read_excel(cfg.)
    meta.keys()
    meta = meta[meta['AGE'].notna()]
    meta = meta[meta['SEX_ID (1=m, 2=f)'].notna()]

