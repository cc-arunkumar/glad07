from knowledge.vector_index import find_nearest_passage,open_vector_index,search_documents,read_document_chunks
from config import config

def read_executive_checks(collection,issue):
 doc=config.GUIDE_DOCUMENT_BY_ISSUE.get(issue)
 if not doc:raise ValueError('Unknown issue')
 return find_nearest_passage(collection,config.EXECUTIVE_CHECKS_QUESTION,doc)

def read_rebooking_steps(collection,category):
 doc=config.CATEGORY_CODE_DOCUMENTS.get(category)
 if not doc:
  raise ValueError('Unknown product category')
 return find_nearest_passage(collection,config.REBOOKING_QUESTION,doc)

def read_standing_restrictions(collection,category=None):
 doc=(config.CATEGORY_CODE_DOCUMENTS.get(category) if category is not None
      else 'Advertising-Clearance-Policy.pdf')
 if not doc:
  raise ValueError('Unknown product category')
 question=config.STANDING_RESTRICTIONS_QUESTION
 return find_nearest_passage(collection,question,doc)
