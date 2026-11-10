from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor,ExtraTreesRegressor
def create_exposure_model(numeric_features,categorical_features):
 prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),numeric_features),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),categorical_features)])
 return Pipeline([('prep',prep),('model',ExtraTreesRegressor(n_estimators=160,min_samples_leaf=2,random_state=42,n_jobs=-1))])
