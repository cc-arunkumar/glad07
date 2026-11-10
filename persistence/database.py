import os
from sqlalchemy import create_engine,inspect,text
from sqlalchemy.orm import sessionmaker
from config import config
os.makedirs(config.DATABASE_DIRECTORY,exist_ok=True)
engine=create_engine(config.APPLICATION_DATABASE_URL,connect_args={'check_same_thread':False})
session_factory=sessionmaker(bind=engine,expire_on_commit=False)
def initialize_database():
 from persistence.models import Base
 Base.metadata.create_all(engine)
 inspector=inspect(engine)
 additions={
  'user_accounts':{'password_salt':"VARCHAR DEFAULT ''"},
  'agent_runs':{'thread_id':"VARCHAR DEFAULT ''",'status_final':'VARCHAR'},
  'chat_turns':{
   'user_role':'VARCHAR','question_text':"TEXT DEFAULT ''",'question_type':"VARCHAR DEFAULT ''",
   'campaign_reference':'VARCHAR','answer_text':"TEXT DEFAULT ''",
   'source_documents':"TEXT DEFAULT '[]'",'closest_distance':'FLOAT',
   'was_refused':'BOOLEAN DEFAULT 0',
  },
 }
 with engine.begin() as connection:
  for table,table_additions in additions.items():
   if table not in inspector.get_table_names():
    continue
   columns={column['name'] for column in inspector.get_columns(table)}
   for column,definition in table_additions.items():
    if column not in columns:
     connection.execute(text(
      f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
     ))
  if 'user_accounts' in inspector.get_table_names():
   rows=connection.execute(text(
    "SELECT username, password_hash FROM user_accounts WHERE password_hash LIKE '%:%'"
   )).fetchall()
   for username,stored in rows:
    salt,digest=stored.split(':',1)
    connection.execute(text(
     "UPDATE user_accounts SET password_hash=:digest, password_salt=:salt WHERE username=:username"
    ),{'digest':digest,'salt':salt,'username':username})
