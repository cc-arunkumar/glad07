import json
import secrets
from persistence.database import initialize_database,session_factory
from persistence.models import UserAccount
from access.password_security import hash_password
from config import config
def main():
 initialize_database()
 with session_factory() as s:
  for account in config.SEED_USER_ACCOUNTS:
   user=s.get(UserAccount,account['username'])
   salt=secrets.token_hex(config.PASSWORD_SALT_BYTES)
   values={'display_name':account['display_name'],'role':account['role'],
           'password_hash':hash_password(account['password'],salt),'password_salt':salt,
           'assigned_agencies':json.dumps(account.get('assigned_agencies',[]))}
   if user is None:
    s.add(UserAccount(username=account['username'],**values))
   else:
    for name,value in values.items():
     setattr(user,name,value)
  s.commit()
if __name__=='__main__':main()
