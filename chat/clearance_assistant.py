from config import config
def answer_question(question,collection=None,history=None):
 if len(str(question).strip())<config.CHAT_QUESTION_MINIMUM_CHARACTERS:return config.CHAT_REFUSAL_MESSAGE
 from knowledge.vector_index import open_vector_index,search_documents
 col=collection or open_vector_index();passages=search_documents(col,[question])[0];kept=[p for p in passages if p['distance']<=config.CHAT_MAX_RELEVANCE_DISTANCE]
 if not kept:return config.CHAT_REFUSAL_MESSAGE
 return '\n\n'.join(f"[{p['document_name']}] {p['chunk_text']}" for p in kept)
