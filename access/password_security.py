import hashlib,secrets
from config import config
def hash_password(password,salt=None):
 salt=salt or secrets.token_hex(config.PASSWORD_SALT_BYTES)
 salt_bytes=bytes.fromhex(salt) if isinstance(salt,str) else salt
 return hashlib.pbkdf2_hmac('sha256',str(password).encode(),salt_bytes,
                            config.PASSWORD_HASH_ROUNDS).hex()

def verify_password(password,stored,salt=None):
 try:
  if salt is None and ':' in stored:
   legacy_salt,stored=stored.split(':',1)
   salt=legacy_salt
  if not salt:return False
  actual=hash_password(password,salt)
  return secrets.compare_digest(actual,stored)
 except (TypeError,ValueError):
  return False
