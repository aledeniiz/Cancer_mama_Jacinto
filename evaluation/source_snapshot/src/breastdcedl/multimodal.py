"""Compact multimodal fusion and JSON-only train-fitted clinical transforms."""
import numpy as np
import torch
from torch import nn
class FusionNet(nn.Module):
 def __init__(self, clinical_dim):
  super().__init__();self.clinical=nn.Sequential(nn.Linear(clinical_dim,8),nn.ReLU(),nn.Dropout(.2));self.fusion=nn.Sequential(nn.Linear(72,16),nn.ReLU(),nn.Dropout(.3),nn.Linear(16,1))
 def forward(self,image,clinical):return self.fusion(torch.cat((image,self.clinical(clinical)),1)).flatten()
def fit_clinical(frame,features):
 x=frame[features].apply(__import__('pandas').to_numeric,errors='coerce').to_numpy(float);x[x==-1]=np.nan
 med=np.nanmedian(x,0);med=np.where(np.isfinite(med),med,0);filled=np.where(np.isnan(x),med,x);mean=filled.mean(0);std=filled.std(0);std[std<1e-8]=1
 return dict(features=list(features),median=med.tolist(),mean=mean.tolist(),std=std.tolist())
def encode_clinical(records,state):
 import pandas as pd
 frame=pd.DataFrame([records] if isinstance(records,dict) else records);x=frame.reindex(columns=state['features']).apply(pd.to_numeric,errors='coerce').to_numpy(float);x[x==-1]=np.nan
 if np.isinf(x).any():raise ValueError('Finite clinical inputs required')
 
 for j,name in enumerate(state['features']):
  values=x[:,j][~np.isnan(x[:,j])]
  if name=='HR' and not np.isin(values,[0,1]).all():raise ValueError('HR must be 0, 1 or missing')
  if name=='age' and ((values<18)|(values>110)).any():raise ValueError('Age outside supported adult range')
 missing=np.isnan(x);x=np.where(missing,state['median'],x);return np.concatenate(((x-state['mean'])/state['std'],missing.astype(float)),1).astype('float32')
def predict_embedding(embedding64,clinicalrecord,bundle):
 net=FusionNet(len(bundle['clinical_transform']['features'])*2);net.load_state_dict(bundle['head_state']);net.eval()
 emb=np.asarray(embedding64,dtype='float32').reshape(1,64);emb=(emb-np.array(bundle['embedding_mean']))/np.array(bundle['embedding_std'])
 with torch.inference_mode():return float(net(torch.tensor(emb,dtype=torch.float32),torch.tensor(encode_clinical(clinicalrecord,bundle['clinical_transform']))).sigmoid().item())
