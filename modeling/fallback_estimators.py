import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin,ClassifierMixin
class AudienceRuleRegressor(BaseEstimator,RegressorMixin):
 def fit(self,X,y): self.feature_names_in_=np.asarray(getattr(X,'columns',[]),dtype=object);self.n_features_in_=len(self.feature_names_in_);return self
 def predict(self,X):
  import pandas as pd
  d=pd.DataFrame(X).copy();allv=pd.to_numeric(d.get('audience_all',0),errors='coerce').fillna(0);idx=pd.to_numeric(d.get('forecast_under18_index',40),errors='coerce').fillna(40);start=pd.to_numeric(d.get('start_minute',800),errors='coerce').fillna(800);day=d.get('school_holiday_flag','No').astype(str).str.lower().isin(['yes','true','1']).astype(float);return np.maximum(0,allv*(idx/100)*(1+0.08*day)*np.where(start<360,0.25,np.where(start>1320,0.55,1)))
class IssueKeywordClassifier(BaseEstimator,ClassifierMixin):
 def fit(self,X,y):
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity
  self.classes_=np.unique(np.asarray(y));self.vectorizer_=TfidfVectorizer(ngram_range=(1,2),stop_words='english',max_features=40000,min_df=1);mat=self.vectorizer_.fit_transform(np.asarray(X,dtype=str));self.centroids_=[]
  for c in self.classes_:self.centroids_.append(np.asarray(mat[np.asarray(y)==c].mean(axis=0)))
  self.centroids_=np.vstack(self.centroids_);return self
 def predict(self,X):
  from sklearn.metrics.pairwise import cosine_similarity
  scores=cosine_similarity(self.vectorizer_.transform(np.asarray(X,dtype=str)),self.centroids_);return self.classes_[scores.argmax(axis=1)]
 def predict_proba(self,X):
  from scipy.special import softmax
  from sklearn.metrics.pairwise import cosine_similarity
  scores=cosine_similarity(self.vectorizer_.transform(np.asarray(X,dtype=str)),self.centroids_)
  return softmax(scores,axis=1)
