import pandas as pd,numpy as np,re
from config import config

def _minute(value):
 try:
  if isinstance(value,(int,float,np.number)):return float(value)
  m=re.fullmatch(r'(\d{1,2}):(\d{2})',str(value).strip())
  if not m:return np.nan
  h,mi=map(int,m.groups());return h*60+mi
 except:return np.nan

def add_slot_minutes(frame):
 df=frame.copy()
 if config.START_MINUTE_COLUMN not in df:
  df[config.START_MINUTE_COLUMN]=df.get(config.START_TIME_COLUMN,pd.Series(index=df.index)).map(_minute)
 if config.DURATION_MINUTES_COLUMN not in df:
  start=df.get(config.START_TIME_COLUMN,pd.Series(index=df.index)).map(_minute);end=df.get(config.END_TIME_COLUMN,pd.Series(index=df.index)).map(_minute)
  df[config.DURATION_MINUTES_COLUMN]=(end-start).clip(lower=0)
 return df

def build_feature_frame(frame,numeric_columns=None,categorical_columns=None):
 if isinstance(numeric_columns,str) and categorical_columns is None:
  task=numeric_columns
  numeric_columns=config.EXPOSURE_NUMERIC_FEATURES if task in ('exposure','audience') else config.AMENDMENT_NUMERIC_FEATURES
  categorical_columns=config.EXPOSURE_CATEGORICAL_FEATURES if task in ('exposure','audience') else config.AMENDMENT_CATEGORICAL_FEATURES
 elif numeric_columns is None:
  numeric_columns=config.AMENDMENT_NUMERIC_FEATURES;categorical_columns=config.AMENDMENT_CATEGORICAL_FEATURES
 else:
  categorical_columns=categorical_columns or []
 df=frame.copy()
 if config.START_MINUTE_COLUMN in numeric_columns or config.DURATION_MINUTES_COLUMN in numeric_columns:df=add_slot_minutes(df)
 cols=list(numeric_columns)+list(categorical_columns)
 for c in numeric_columns:
  if c not in df:df[c]=np.nan
  df[c]=pd.to_numeric(df[c],errors='coerce').astype(float)
 for c in categorical_columns:
  if c not in df:df[c]='Unknown'
  df[c]=df[c].map(lambda value: np.nan if pd.isna(value) else str(value))
 return df[cols].copy()
