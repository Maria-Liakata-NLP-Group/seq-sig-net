import numpy as np
import pandas as pd
import re
import torch

class PrepareData:
    """
    model_specifics: dict
        dictionary of specified model options
    time_column: list or str, optional
        contains the name of time features. default: 'time_encoding'
    zero_padding: boolean, optional
        if True, we pad by 0s, if False we pad by the last data point. default: True
    w_last: boolean, optional
        if True we pad the last w posts, if False we pad as much as the maximum timeline in the dataset. default: True
    """
    def __init__(self, model_specifics, time_column = 'time_encoding', zero_padding=True, w_last=True):
        self.pad_window = model_specifics["w"] if "w" in model_specifics.keys() else model_specifics["history_len"] if "history_len" in model_specifics.keys() else 10
        self.k = model_specifics["k"] if "k" in model_specifics.keys() else None
        self.n = model_specifics["n"] if "n" in model_specifics.keys() else None
        self.dim_reduction = model_specifics['dimensionality_reduction'] if "dimensionality_reduction" in model_specifics.keys() else None
        self.post_embedding_tp = model_specifics['post_embedding_tp'] if "post_embedding_tp" in model_specifics.keys() else None
        self.time_injection_post_tp = model_specifics['time_injection_post_tp'] if "time_injection_post_tp" in model_specifics.keys() else None
        self.zero_padding = zero_padding
        self.w_last = w_last
        self.time_column = time_column if ((isinstance(time_column, list)) | (time_column==None)) else [time_column]
        self.pad_with = model_specifics["pad_with"] if "pad_with" in model_specifics.keys() else 0
        

    def pad(self, df):

        #sort dataframe (in case it is not sorted already)
        df = df.sort_values(by=['timeline_id', 'datetime']).reset_index(drop=True)

        #keep reduced or original embeddings?
        if (self.dim_reduction==True) :
            emb_str = "^d\w*[0-9]"
        else:
            emb_str = "^e\w*[0-9]"

        id_counts = df.groupby(['timeline_id'])['timeline_id'].count()
        time_n = id_counts.max()
        if self.time_column == None:
            df_new = np.array(df[['timeline_id','label']+[c for c in df.columns if re.match(emb_str, c)]])
        else:
            df_new = np.array(df[['timeline_id','label']+ self.time_column +[c for c in df.columns if re.match(emb_str, c)]])
        
        #iterate to create slices
        start_i = 0
        end_i = 0
        dims = df_new.shape[1]
        zeros = np.concatenate(( np.array([100]), np.repeat(self.pad_with, dims-2) ),axis=0)
        sample_list = []

        for i in range(df.shape[0]):
            if (i==0):
                i_prev = 0
            else:
                i_prev = i-1
            if (df['timeline_id'][i]==df['timeline_id'][i_prev]):
                end_i +=1
                if ((self.w_last==True) & ((end_i - start_i) > self.pad_window)):
                    start_i = end_i - self.pad_window
            else: 
                start_i = i
                end_i = i+1

            #data point with history
            df_add = df_new[start_i:end_i, 1:][np.newaxis, :, :]
            #padding length
            if (self.w_last == True):
                padding_n = self.pad_window - (end_i- start_i) 
            else:
                padding_n = time_n - (end_i- start_i) 
            #create padding
            if self.zero_padding:
                zeros_tile = np.tile(zeros,(padding_n,1))[np.newaxis, :, :]
            else:
                zeros_tile = np.tile(df_new[end_i-1, 1:],(padding_n,1))[np.newaxis, :, :]
            #append zero padding
            df_padi = np.concatenate((df_add, zeros_tile) ,axis=1) 
            #append each sample to final list
            sample_list.append(df_padi)
        
        return df, np.concatenate(sample_list)
    
    def unit_input(self, df, df_padded, embeddings_lastdim=False):
        #torch conversion and removal of label and time dimensions
        exclude_ind = 1 + (0 if (self.time_column==None) else len(self.time_column))
        path = torch.from_numpy(df_padded[: , : , exclude_ind:].astype(float))
        
        #get time feature and standardise it (if defined)
        if (self.time_injection_post_tp == 'timestamp'):
            self.mean = df_padded[: , : , 1][df_padded[: , : , 1]!=0].mean()
            self.std = df_padded[: , : , 1][df_padded[: , : , 1]!=0].std()
            time_feature = (torch.from_numpy(df_padded[: , : , 1].astype(float)).unsqueeze(1) - self.mean) /self.std
        else:
            time_feature = None

        #get current post (if defined)
        if (self.post_embedding_tp == 'sentence'):
            bert_embeddings = torch.tensor(df[[c for c in df.columns if re.match("^e\w*[0-9]", c)]].values).unsqueeze(2).repeat(1, 1, self.pad_window)
        elif (self.post_embedding_tp == 'reduced'):
            bert_embeddings = torch.tensor(df[[c for c in df.columns if re.match("^d\w*[0-9]", c)]].values).unsqueeze(2).repeat(1, 1, self.pad_window)
        else:
            bert_embeddings = None

        #transpose dimensions of data to have: [sample size, embedding features ,window size(w)]
        x_data = torch.transpose(path, 1,2)
            
        #concatenate timestamp and current post (if needed)
        if (time_feature != None):
            x_data = torch.cat((x_data, time_feature), dim=1)
        if (bert_embeddings != None):
            x_data = torch.cat((x_data, bert_embeddings), dim=1)
        
        if embeddings_lastdim:
            x_data = torch.transpose(x_data, 1,2)

        return x_data
    
    def lstm_input(self, df, x_data):
        x_datam = []
        mask_m = []
        x_data_units = torch.clone(x_data.unsqueeze(3))

        for i in range(1,self.n):
            #boolean columns to determine if timeline change when we shift back in data points
            tl_i = i*self.k
            df['timeline_match'+ str(tl_i)] = df['timeline_id'].eq(df.timeline_id.shift(tl_i))
            #constract shifted matrices
            x_datam.append(torch.roll(x_data, tl_i, 0))
            #create mask and then assing 0s based on that mask
            mask_m.append(torch.zeros_like(x_data))
            mask_m[i-1][~torch.tensor(df['timeline_match'+ str(tl_i)].values)] = 2
            mask_m[i-1] = mask_m[i-1].ge(1)
            #zero out (pad) cases that the timeline_id has changed
            x_datam[i-1][mask_m[i-1]] = 0
            #concatenate all units together. Expected output: [sample size, embedding features ,window size(w), number of units(n)]
            x_data_units = torch.cat((x_data_units, x_datam[i-1].unsqueeze(3)), dim=3)

        return x_data_units