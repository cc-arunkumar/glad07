from pydantic import BaseModel
class SignInRequest(BaseModel):username:str;password:str
class ChatRequest(BaseModel):question:str
class DecisionRequest(BaseModel):decision:str;reason:str
