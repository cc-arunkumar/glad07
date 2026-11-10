import re

def parse_labelled_response(response, labels=None, agent_name=None):
 text=re.sub(r'<think>.*?</think>','',str(response),flags=re.I|re.S).strip()
 if labels is None:
  return text

 labels=list(labels)
 normalized_lines=[]
 for line in text.splitlines():
  line=re.sub(r'^\s*#+\s*','',line)
  line=re.sub(r'^\s*>\s?','',line)
  normalized_lines.append(line.strip())

 pattern=re.compile(
  r'^\s*(?:\*\*)?(?P<label>'+'|'.join(re.escape(label) for label in
      sorted(labels,key=len,reverse=True))+r')(?:\*\*)?\s*:\s*(?P<value>.*)$',
  re.I)
 positions=[]
 for index,line in enumerate(normalized_lines):
  match=pattern.match(line)
  if match:
   label=next(label for label in labels if label.lower()==match.group('label').lower())
   positions.append((index,label,match.group('value')))
 parsed={}
 for number,(line_index,label,first_value) in enumerate(positions):
  stop=positions[number+1][0] if number+1<len(positions) else len(normalized_lines)
  value_lines=([first_value] if first_value else [])+normalized_lines[line_index+1:stop]
  value='\n'.join(part for part in value_lines if part.strip()).strip()
  value=re.sub(r'(?m)^\s*>\s?','',value)
  value=value.strip("*#`_~ \t-–—•>:[]")
  if not value:
   raise ValueError(f'Empty response for {label} from {agent_name or "agent"}')
  if label in parsed:
   parsed[label]+='\n'+value
  else:
   parsed[label]=value
 missing=[label for label in labels if label not in parsed]
 if missing:
  raise ValueError('Missing response labels: '+', '.join(missing))
 return parsed

def call_agent(client,prompt,agent_name=None):
 if client is None:raise ValueError('LLM client not configured')
 result=client.models.generate_content(model='gemini-2.5-flash',contents=prompt);return parse_labelled_response(getattr(result,'text',result))
