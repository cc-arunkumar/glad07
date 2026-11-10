import html
import re, math
import pandas as pd
from config import config

def clean_text(value):
 if pd.isna(value): return None
 s=html.unescape(str(value)).replace('\xa0',' ').strip()
 if s.lower() in [x.lower() for x in config.NULL_TOKENS]: return None
 s=re.sub(config.MARKUP_PATTERN,' ',s); s=re.sub(config.ENCODING_ARTIFACT_PATTERN,' ',s); s=re.sub(config.WHITESPACE_RUN_PATTERN,' ',s).strip()
 return s or None

def clean_mapped(value,mapping):
 s=clean_text(value)
 return mapping.get(s.lower(),s) if s else None

def parse_number(value,kind='plain'):
 s=clean_text(value)
 if not s:return None
 s=s.replace(',','').strip()
 if kind!='audience': s=re.sub(r'\s*(?:spots?|channels?)$', '', s, flags=re.I).strip()
 if kind=='audience':
  m=re.fullmatch(r'(-?\d+(?:\.\d+)?)\s*(thousand|k|m)?',s,re.I)
  if not m:return None
  n=float(m.group(1)); unit=(m.group(2) or '').lower()
  if unit=='m': n*=1000
  elif unit in ('k','thousand'): pass
  elif abs(n)>=1000: n/=1000
  return round(n,config.AUDIENCE_DECIMALS)
 s=s.replace('%','').strip();
 try:return float(s) if '.' in s else int(s)
 except:return None

def parse_spot_length(value):
 s=clean_text(value)
 if not s:return None
 m=re.fullmatch(r'(\d+)(?:"|s|\s*sec)?',s,re.I)
 if m:return int(m.group(1))
 m=re.fullmatch(r'(\d+):(\d{2})',s)
 if m:return int(m.group(1))*60+int(m.group(2))
 m=re.fullmatch(r'(\d{2}):(\d{2}):(\d{2})',s)
 if m:return int(m.group(1))*3600+int(m.group(2))*60+int(m.group(3))
 return None

def parse_clock_time(value, system=None, day=None):
 s=clean_text(value)
 if not s:return None
 m=re.fullmatch(r'(\d{1,2}):(\d{2})',s)
 if not m:return None
 h,mi=map(int,m.groups()); mins=h*60+mi
 if str(system).strip().lower()!='txline' and h<6: mins+=1440
 return mins

def normalize_table(df, value_mappings=None, whole_patterns=None, decimal_patterns=None, range_limits=None):
 out=df.copy()
 for c in out.columns:
  if out[c].dtype=='object': out[c]=out[c].map(clean_text)
 for c,m in (value_mappings or {}).items():
  if c in out: out[c]=out[c].map(lambda x:clean_mapped(x,m))
 for c in (whole_patterns or {}):
  if c in out: out[c]=out[c].map(lambda x:parse_number(x))
 for c in (decimal_patterns or {}):
  if c in out: out[c]=out[c].map(lambda x:parse_number(x))
 for c,(lo,hi) in (range_limits or {}).items():
  if c in out:
   out[c]=out[c].map(lambda x: None if x is None or pd.isna(x) else (None if isinstance(x,(int,float)) and (x<lo or x>hi) else x))
 return out
