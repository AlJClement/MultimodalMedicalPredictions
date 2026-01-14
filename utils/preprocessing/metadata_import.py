import pandas as pd
import numpy as np

class MetadataImport():
    def __init__(self, cfg) -> None:
        self.metapath = cfg.INPUT_PATHS.META_PATH
        self.cols_dict =  cfg.INPUT_PATHS.META_COLS        

        self.pat_col_name = cfg.INPUT_PATHS.ID_COL

        cfg.INPUT_PATHS.META_COLS_CLASSES.append(self.pat_col_name)
        self.class_ls = cfg.INPUT_PATHS.META_COLS_CLASSES

        self.model_name = cfg.MODEL.NAME
        self.model_features = cfg.MODEL.META_FEATURES
     
        
    def load_csv(self):
        #metadata load from csv 
        '''this file loads meta data from a csv, with only specified cols'''
        col_names = list([self.pat_col_name])
        for dic in self.cols_dict:
            col_name, value = list(dic.items())[0]
            col_names.append(col_name)

        meta = pd.read_csv(self.metapath,dtype=object, usecols=col_names)
        meta_class = pd.read_csv(self.metapath,dtype=object, usecols=self.class_ls)
        return meta, meta_class
        
    def _get_array(self, meta_df, patid):
        pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == patid.split('_')[0]]
        # if pat_meta_arr.empty:
        #     pat_id_rnoh=patid.split('_')[1]
        #     pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == pat_id_rnoh]
        if pat_meta_arr.empty:
            ##oai
            pat_id_oai = patid.split('-')[0]
            pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == pat_id_oai]
        if pat_meta_arr.empty:
            #retuve
            pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == patid]
        if pat_meta_arr.empty:
                raise ValueError('no meta data found for: ', patid)
        return pat_meta_arr
    
    def _get_class_arr(self, meta_df, patid):
        pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == patid.split('_')[0]]
        if pat_meta_arr.empty:
            pat_id_rnoh=patid.split('_')[1]
            pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == pat_id_rnoh]
        if pat_meta_arr.empty:
            pat_id_oai = patid.split('-')[0]
            pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == pat_id_oai]
        if pat_meta_arr.empty:
            #retuve
            pat_meta_arr = meta_df.loc[meta_df[self.pat_col_name] == patid]
        if pat_meta_arr.empty:
                raise ValueError('no meta data found for: ', patid)
        return pat_meta_arr.values
    
        
    def _duplicate_col(self, meta_data_col, num_cols):
        new_cols = meta_data_col.astype(float)*np.ones((num_cols,1))
        return new_cols
    
    def _hot_encode(self, meta_data_col, num_cols, col_name):
        unique, inverse = np.unique(meta_data_col, return_inverse=True)
        new_cols = np.eye(unique.shape[0])[inverse].transpose()
        if unique.shape[0]==2:
            num_cols_per_col = int(num_cols/2)
            _new_cols_0 = self._duplicate_col(new_cols[0],num_cols_per_col)
            _new_cols_1 = self._duplicate_col(new_cols[1],num_cols_per_col)
            new_cols = np.concatenate((_new_cols_0, _new_cols_1))
        elif unique.shape[0]==1:
            print('WARNING:CHECK HOT ENCODED VALUES FOR',col_name,', EXPECTED TWO BUT ONLY FOUND 1 UNIQUE COL VALUE')
            new_cols = self._duplicate_col(new_cols[0],num_cols)
        else:
            raise ValueError('FOUND ', unique.shape[0],' UNIQUE VALUES, EXPECTED 2 FOR HOT ENCODING')
            
        return new_cols

    def _tokenize(self, meta_data_col):
        new_cols = meta_data_col
        return new_cols

    def unet_meta_lastlayer(self, meta_data_arr):
        '''takes input features from data loader and restructures for how the network requires'''
        #for this we take the number of meta features and distribute across all values
        new_encoded_arr = np.array([])
        
        if self.model_features == []:
            return new_encoded_arr
        else:
            for i in range(len(self.cols_dict)): #assume first col is accession number/patient id so pass
                col_name, col_encodetype = list(self.cols_dict[i].items())[0]
                num = self.model_features[i] #amount of these features to add
                meta_data_col = meta_data_arr.transpose()[i]

                if col_encodetype == 'hot':
                    new_cols = self._hot_encode(meta_data_col, num, col_name)
                elif col_encodetype == 'continuous':
                    new_cols = self._duplicate_col(meta_data_col, num)
                elif col_encodetype == 'tokenize':
                    new_cols = self._tokenize(meta_data_col, num)
                else:
                    raise ValueError('check colum encoding types')
                
                if new_encoded_arr.shape[0]==0:
                    new_encoded_arr=new_cols
                else:
                    new_encoded_arr=np.concatenate((new_encoded_arr, new_cols))

            return new_encoded_arr.transpose()

    def unet_plus_plus(self, meta_data_arr):
        return self.unet_meta_lastlayer(meta_data_arr)

    def unet_plus_plus_meta(self, meta_data_arr):
        return self.unet_meta_lastlayer(meta_data_arr)
    
    def hrnet(self, meta_data_arr):
        return self.unet_meta_lastlayer(meta_data_arr)