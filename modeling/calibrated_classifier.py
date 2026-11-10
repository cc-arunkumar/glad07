import numpy as np
from scipy.special import expit,logit
class PriorCalibratedClassifier:
 def __init__(self,base_model,log_odds_offset=0.0,feature_names=None):self.base_model=base_model;self.log_odds_offset=float(log_odds_offset);self.feature_names_in_=np.asarray(feature_names or [],dtype=object);self.classes_=np.array([0,1]);self.n_features_in_=len(self.feature_names_in_)
 def predict_proba(self,X):
  raw=np.clip(self.base_model.predict_proba(X)[:,1],1e-6,1-1e-6);p=expit(logit(raw)+self.log_odds_offset);return np.c_[1-p,p]
 def predict(self,X):return (self.predict_proba(X)[:,1]>=.5).astype(int)
