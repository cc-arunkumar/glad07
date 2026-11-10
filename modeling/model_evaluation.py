from sklearn.metrics import roc_auc_score,r2_score,f1_score,classification_report

def evaluate_classifier(model,X_train,y_train,X_eval=None,y_eval=None):
 def prob(x):return model.predict_proba(x)[:,1] if hasattr(model,'predict_proba') else model.decision_function(x)
 tr=prob(X_train);auc_tr=float(roc_auc_score(y_train,tr))
 if X_eval is None:return {'train_roc_auc':auc_tr,'roc_auc':auc_tr}
 ev=prob(X_eval);auc_ev=float(roc_auc_score(y_eval,ev));return {'train_roc_auc':auc_tr,'held_out_roc_auc':auc_ev,'generalisation_gap':auc_tr-auc_ev,'roc_auc':auc_ev,'scores':ev}
def evaluate_regressor(model,X_train,y_train,X_eval=None,y_eval=None):
 tr=model.predict(X_train);r2tr=float(r2_score(y_train,tr))
 if X_eval is None:return {'train_r2':r2tr,'r2':r2tr}
 ev=model.predict(X_eval);r2ev=float(r2_score(y_eval,ev));import numpy as np;mae=float(np.mean(np.abs(np.asarray(y_eval)-ev)));baseline=float(np.mean(np.abs(np.asarray(y_eval)-np.mean(y_train))));return {'train_r2':r2tr,'held_out_r2':r2ev,'generalisation_gap':r2tr-r2ev,'r2':r2ev,'predictions':ev,'mean_absolute_error':mae,'average_estimate_mean_absolute_error':baseline,'lowest_prediction':float(np.min(ev))}
def evaluate_text_classifier(model,X_train,y_train,X_eval=None,y_eval=None):
 tr=model.predict(X_train);ftr=float(f1_score(y_train,tr,average='macro',zero_division=0))
 if X_eval is None:return {'train_macro_f1':ftr,'macro_f1':ftr}
 ev=model.predict(X_eval);fev=float(f1_score(y_eval,ev,average='macro',zero_division=0));return {'train_macro_f1':ftr,'held_out_macro_f1':fev,'generalisation_gap':ftr-fev,'macro_f1':fev,'predictions':ev,'per_class_f1':{str(label):float(f1_score(y_eval,ev,labels=[label],average='macro',zero_division=0)) for label in sorted(set(y_eval))}}
