from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
def create_amendment_model(numeric_features,categorical_features):
 prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),numeric_features),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),categorical_features)])
 return Pipeline([('prep',prep),('model',ExtraTreesClassifier(n_estimators=220,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1))])
