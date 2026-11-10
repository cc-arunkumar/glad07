from pathlib import Path
from pypdf import PdfReader
import re
from config import config
def extract_text_from_directory(directory):
 result=[]
 for path in sorted(Path(directory).glob('*.pdf')):
  reader=PdfReader(str(path)); pages=[]
  for page in reader.pages:
   text=page.extract_text() or ''
   if not text.strip(): raise ValueError(f'Page in {path.name} contains no extractable text')
   pages.append(text)
  content='\n'.join(pages);content=re.sub(config.CONTROL_CHARACTER_PATTERN,' ',content);content=re.sub(r'[ \t]+',' ',content);content=re.sub(r'\n{2,}','\n',content).strip()
  result.append({'document_name':path.name,'document_text':content})
 return result
