import re,hashlib
from config import config
def split_documents_into_sections(documents):
 sections=[]
 for d in documents:
  txt=d['document_text']
  headers=list(re.finditer(config.DOCUMENT_HEADER_PATTERN,txt,flags=re.I))
  if len(headers)!=1: raise ValueError('Document must contain exactly one metadata header')
  txt=txt[headers[0].end():]
  matches=list(re.finditer(config.SECTION_HEADING_PATTERN,txt))
  if len(matches)<2:
   # PDFs often flatten line boundaries: split by numbered headings
   matches=list(re.finditer(r'(?<!\d)(\d{1,2})\.\s+([A-Z])',txt))
  if not matches or not re.match(r'\s*1\. [A-Z]',txt): raise ValueError('Document sections must start at 1')
  for i,m in enumerate(matches):
   part=txt[m.start():matches[i+1].start() if i+1<len(matches) else len(txt)].strip()
   if part:sections.append({'document_name':d['document_name'],'section_text':part})
 return sections
def build_chunk_records(sections):
 out=[];size=config.CHUNK_SIZE_CHARACTERS;overlap=config.CHUNK_OVERLAP_CHARACTERS
 for sec in sections:
  text=sec['section_text']; doc=sec['document_name']; start=0
  while start<len(text):
   end=min(len(text),start+size)
   if end<len(text):
    boundary=text.rfind(' ',start+size//2,end)
    if boundary>start:end=boundary
   chunk=text[start:end].strip()
   chunk=re.sub(r'\b\d{1,2}:$','',chunk).strip()
   if chunk:
    ident=hashlib.sha1((doc+str(len(out))+chunk).encode()).hexdigest()[:20];out.append({'chunk_id':ident,'chunk_text':chunk,'document_name':doc})
   if end>=len(text):break
   start=max(start+1,end-overlap)
   while start<len(text) and not text[start-1:start].isspace(): start+=1
 return out
