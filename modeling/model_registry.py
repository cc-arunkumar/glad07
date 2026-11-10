import joblib,json,os
def save_model(model,path): os.makedirs(os.path.dirname(path) or '.',exist_ok=True);joblib.dump(model,path)
def load_model(path): return joblib.load(path)
def save_feature_schema(schema,path): os.makedirs(os.path.dirname(path) or '.',exist_ok=True);json.dump(list(schema),open(path,'w'),indent=2)
def load_feature_schema(path): return json.load(open(path))
