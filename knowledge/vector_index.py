import os,json,re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.chunk_splitting import split_documents_into_sections,build_chunk_records
from config import config
class LocalPassageIndex:
 def __init__(self,records):
  self.records=records
  self.vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
  self.matrix=self.vectorizer.fit_transform([r['chunk_text'] for r in records]) if records else None
 def count(self):return len(self.records)
 def get(self,include=None,where=None):
  rec=[r for r in self.records if not where or all(r.get(k)==v for k,v in where.items())];return {'ids':[r['chunk_id'] for r in rec],'documents':[r['chunk_text'] for r in rec],'metadatas':[{'document_name':r['document_name']} for r in rec]}
 def query(self,query_texts,n_results,include=None):
  ds=[];ms=[];ids=[]
  for q in query_texts:
   sim=cosine_similarity(self.vectorizer.transform([q]),self.matrix)[0] if self.matrix is not None else []
   order=sorted(range(len(sim)),key=lambda i:(-sim[i],i))[:min(n_results,len(sim))];ds.append([self.records[i]['chunk_text'] for i in order]);ms.append([{'document_name':self.records[i]['document_name']} for i in order]);ids.append([self.records[i]['chunk_id'] for i in order])
  return {'documents':ds,'metadatas':ms,'ids':ids,'distances':[[1-float(cosine_similarity(self.vectorizer.transform([q]),self.matrix)[0][i]) for i in sorted(range(len(self.records)),key=lambda i:-cosine_similarity(self.vectorizer.transform([q]),self.matrix)[0][i])[:min(n_results,len(self.records))]] for q in query_texts]}
def build_vector_index(records=None,directory=config.VECTOR_STORE_DIRECTORY):
 if isinstance(records,(str,os.PathLike)) and directory==config.VECTOR_STORE_DIRECTORY:
  directory=str(records);records=None
 if records is None:
  docs=extract_text_from_directory(config.DOCS_DIRECTORY)
  records=build_chunk_records(split_documents_into_sections(docs))
 os.makedirs(directory,exist_ok=True)
 with open(os.path.join(directory,'passages.json'),'w',encoding='utf-8') as stream:
  json.dump(records,stream,indent=2)
 return len(records)
def open_vector_index(directory=config.VECTOR_STORE_DIRECTORY):
 f=os.path.join(directory,'passages.json')
 if not os.path.isfile(f):
  build_vector_index(directory=directory)
 with open(f,encoding='utf-8') as stream:
  return LocalPassageIndex(json.load(stream))
def read_document_chunks(collection,document_name):
 rec=[r for r in collection.records if r['document_name']==document_name]
 if not rec:raise ValueError('Unknown document: '+document_name)
 return [r['chunk_text'] for r in rec]
def find_nearest_passage(collection,question,document_name):
 rec=[r for r in collection.records if r['document_name']==document_name]
 if not rec:raise ValueError('Unknown document: '+document_name)
 scores=cosine_similarity(collection.vectorizer.transform([question]),collection.matrix)[0]
 indices=[index for index,record in enumerate(collection.records)
          if record['document_name']==document_name]
 best=max(indices,key=lambda index:(scores[index],-index))
 return collection.records[best]['chunk_text']
def search_documents(collection,questions):
 results=[]
 for q in questions:
  pool=[r for r in collection.records if r['document_name'] not in config.CHAT_EXCLUDED_DOCUMENTS]
  if not pool:
   results.append([])
   continue
  vect=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
  mat=vect.fit_transform([r['chunk_text'] for r in pool])
  scores=cosine_similarity(vect.transform([q]),mat)[0]
  order=sorted(range(len(pool)),key=lambda i:(-scores[i],i))[:config.CHAT_RETRIEVAL_RESULT_COUNT]
  results.append([{'chunk_id':pool[i]['chunk_id'],'chunk_text':pool[i]['chunk_text'],
                   'document_name':pool[i]['document_name'],'distance':float(1-scores[i])}
                  for i in order])
 return results
