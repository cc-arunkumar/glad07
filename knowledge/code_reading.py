import re
from knowledge.vector_index import read_document_chunks
from config import config

def _one_value(pattern,text,group=1,flags=re.I):
 values=[]
 for match in re.finditer(pattern,text,flags):
  value=match.group(group)
  if value not in values:
   values.append(value)
 if len(values)>1:
  raise ValueError('Conflicting rules in category code')
 return values[0] if values else None

def read_category_code(collection,category):
 if category not in config.CATEGORY_CODE_DOCUMENTS:raise ValueError('Unknown product category')
 doc=config.CATEGORY_CODE_DOCUMENTS[category]
 text=' '.join(read_document_chunks(collection,doc))
 hour_matches=list(re.finditer(
  r'may not be transmitted between\s+(?P<start>\d{1,2}:\d{2})\s+and\s+(?P<end>\d{1,2}:\d{2})',
  text,re.I))
 hours_values={(m.group('start'),m.group('end')) for m in hour_matches}
 if len(hours_values)>1:
  raise ValueError('Conflicting restricted hours in category code')
 hours=({'start':next(iter(hours_values))[0],'end':next(iter(hours_values))[1]}
        if hours_values else None)
 unrestricted=bool(re.search(config.ANY_TIME_PATTERN,text,re.I))
 if unrestricted and hours:
  raise ValueError('Category code states both restricted hours and any time')
 if not unrestricted and not hours:
  raise ValueError('Category code does not state its transmission hours')

 sport_matches=list(re.finditer(
  r'begins more than\s+(?P<after>\d+)\s+minutes after.*?more than\s+'
  r'(?P<before>\d+)\s+minutes before',
  text,re.I|re.S))
 sport_values={(int(m.group('after')),int(m.group('before'))) for m in sport_matches}
 if len(sport_values)>1:
  raise ValueError('Conflicting live sport exceptions in category code')
 sport=(dict(zip(('minutes_after_start','minutes_before_end'),next(iter(sport_values))))
        if sport_values else None)
 if sport and not hours:
  raise ValueError('Live sport exception has no restricted hours')

 children_values=set()
 if re.search(r'not be transmitted in or adjacent to a programme made for children',text,re.I):
  children_values.add(config.CHILDREN_IN_OR_ADJACENT)
 if re.search(r'not be transmitted in a programme made for children',text,re.I):
  children_values.add(config.CHILDREN_IN_ONLY)
 if len(children_values)>1:
  raise ValueError('Conflicting children rules in category code')
 child=next(iter(children_values)) if children_values else None

 index=_one_value(r'under-18\s+audience index is\s+(\d+)\s+or more',text)
 cap=_one_value(r'No more than\s+(\d+)\s+spots of one campaign',text)
 separation=_one_value(r'at least\s+(\d+)\s+minutes apart',text)
 rules={
  'restricted_hours':config.BREACH_WINDOW,
  'live_sport_exception':config.BREACH_WINDOW,
  'children':config.BREACH_CHILDREN,
  'under18_index':config.BREACH_INDEX,
  'daily_cap':config.BREACH_FREQUENCY,
  'separation':config.BREACH_SEPARATION,
 }
 return {'category':category,'document_name':doc,'restricted_hours':hours,
         'any_time':unrestricted,'live_sport_exception':sport,'children_rule':child,
         'index_threshold':int(index) if index is not None else None,
         'under18_index_threshold':int(index) if index is not None else None,
         'daily_cap':int(cap) if cap is not None else None,
         'separation_minutes':int(separation) if separation is not None else None,
         'minimum_separation_minutes':int(separation) if separation is not None else None,
         'rule_codes':rules,'source_text':text}
