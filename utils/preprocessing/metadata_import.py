import pandas as pd

class MetadataImport():
    def __init__(self, cfg) -> None:
        self.metapath = cfg.INPUT_PATHS.META_PATH
        self.cols =  cfg.INPUT_PATHS.META_COLS
        self.pat_col_name = cfg.INPUT_PATHS.META_COLS[0]
        self.model_name = cfg.MODEL.NAME
        
    def load_csv(self):
        #metadata load from csv 
        '''this file loads meta data from a csv, with only specified cols'''
        meta = pd.read_csv(self.metapath, usecols=self.cols)
        return meta
        
    def _get_array(self, meta_df, patid):
        pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == patid]
        return pat_meta_arr

    def unet_meta_bottleneck(self):
        new_arr = []
        return new_arr
    
    def model_specific_loader(self):
        new_arr = eval(self.model_name)
        return new_arr

    