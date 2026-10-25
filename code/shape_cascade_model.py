"""One public predictor: shape-Fail screening followed by the original appearance model."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

SEED=20261024
SHAPE_GRID=np.r_[np.arange(.5,1.0,.025),1.01]
CONF_GRID=np.arange(.5,.951,.025)
GAP=.2

def shape_input(df):
    b=df.before_shape.to_numpy(int);a=df.after_shape.to_numpy(int)
    return pd.DataFrame({'before':b.astype(str),'after':a.astype(str),'pair':[f'{x}|{y}' for x,y in zip(b,a)],'changed':(b!=a).astype(int),'valid':((b>0)&(a>0)).astype(int)})

def new_shape_model():
    transform=ColumnTransformer([('categories',OneHotEncoder(handle_unknown='ignore'),['before','after','pair']),('flags','passthrough',['changed','valid'])])
    return make_pipeline(transform,LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED))

def new_appearance_model():
    return LogisticRegression(C=1,class_weight='balanced',max_iter=1000,random_state=SEED)

def appearance_decision(p,confidence):
    top=np.maximum(p,1-p);gap=np.abs(2*p-1)
    return np.where((top>=confidence)&(gap>=GAP),(p>=.5).astype(int),2)

def shape_mask(p,df,threshold):
    return (p>=threshold)&(df.before_shape.to_numpy()>0)&(df.after_shape.to_numpy()>0)&(df.before_shape.to_numpy()!=df.after_shape.to_numpy())

def cascade_decision(cp,ap,df,shape_threshold,confidence):
    result=appearance_decision(ap,confidence)
    result[shape_mask(cp,df,shape_threshold)]=1
    return result

def metrics(y,pred):
    y=np.asarray(y);pred=np.asarray(pred);cm=confusion_matrix(y,pred,labels=[0,1,2])[:2]
    n0,n1=(y==0).sum(),(y==1).sum();tp=cm[1,1];fp=cm[0,1];dec=pred!=2
    divide=lambda a,b:float(a/b) if b else None
    return {'n':len(y),'accuracy_all':float((pred==y).mean()),'accuracy_decided':divide(((pred==y)&dec).sum(),dec.sum()),'coverage':float(dec.mean()),'abstention_rate':float((~dec).mean()),'pass_recall':divide(cm[0,0],n0),'fail_recall_all':divide(tp,n1),'fail_precision':divide(tp,tp+fp),'fail_recall_decided':divide(tp,cm[1,0]+tp),'fail_case_coverage':divide(cm[1,0]+tp,n1),'pass_false_fail_rate':divide(fp,n0),'fail_f1':divide(2*tp,2*tp+fp+(n1-tp)),'balanced_recall_all':float(.5*(cm[0,0]/n0+tp/n1)),'confusion_matrix':cm.tolist()}

def tune(y,cp,ap,df,cascade):
    candidates=[]
    for ct in SHAPE_GRID if cascade else [1.01]:
        screen=shape_mask(cp,df,ct);n=int(screen.sum());precision=float(np.mean(np.asarray(y)[screen]==1)) if n else None
        # A direct-Fail override needs >=5 OOF decisions and >=95% precision.
        if n and (n<5 or precision<.95):continue
        for confidence in CONF_GRID:
            pred=cascade_decision(cp,ap,df,float(ct),float(confidence));m=metrics(y,pred)
            violation=max(0,m['pass_false_fail_rate']-.10)+max(0,.60-m['coverage'])
            rank=(violation<=1e-12,-violation,m['fail_f1'] or 0,m['balanced_recall_all'],m['accuracy_all'],m['coverage'],float(ct))
            candidates.append((rank,{'shape_threshold':float(ct),'appearance_confidence':float(confidence),'gap_threshold':GAP,'oof_shape_direct_fail_count':n,'oof_shape_precision':precision,'oof_metrics':m}))
    return max(candidates,key=lambda item:item[0])[1]

class ShapeAppearanceCascade:
    """fit_select receives training data only; predict returns final labels and routing."""
    def fit_select(self,X,df,groups):
        df=df.reset_index(drop=True);y=df.consistency.to_numpy(int);ci=shape_input(df)
        cp=np.full(len(y),np.nan);ap=np.full(len(y),np.nan);folds=np.full(len(y),-1)
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
        for fold,(fit,valid) in enumerate(cv.split(X,y,groups)):
            assert not set(np.asarray(groups)[fit])&set(np.asarray(groups)[valid])
            assert len(np.unique(y[fit]))==2
            shape=new_shape_model().fit(ci.iloc[fit],y[fit]);appearance=new_appearance_model().fit(X[fit],y[fit])
            cp[valid]=shape.predict_proba(ci.iloc[valid])[:,1];ap[valid]=appearance.predict_proba(X[valid])[:,1];folds[valid]=fold
        assert np.isfinite(cp).all() and np.isfinite(ap).all() and (folds>=0).all()
        self.config_=tune(y,cp,ap,df,True)
        self.baseline_config_=tune(y,cp,ap,df,False)
        self.shape_model_=new_shape_model().fit(ci,y)
        self.appearance_model_=new_appearance_model().fit(X,y)
        self.oof_=df[['ID','consistency']].copy();self.oof_['fold']=folds;self.oof_['task_group']=groups
        self.oof_['shape_fail_probability']=cp;self.oof_['appearance_fail_probability']=ap
        self.oof_['cascade_prediction']=cascade_decision(cp,ap,df,self.config_['shape_threshold'],self.config_['appearance_confidence'])
        self.oof_['baseline_cv_prediction']=appearance_decision(ap,self.baseline_config_['appearance_confidence'])
        self.oof_['original_07_prediction']=appearance_decision(ap,.7)
        return self

    def predict_details(self,X,df):
        cp=self.shape_model_.predict_proba(shape_input(df))[:,1]
        screened=shape_mask(cp,df,self.config_['shape_threshold'])
        # Actual cascade: the appearance model sees only rows not screened Fail.
        ap=np.full(len(df),np.nan);result=np.ones(len(df),dtype=int)
        if (~screened).any():
            ap[~screened]=self.appearance_model_.predict_proba(X[~screened])[:,1]
            result[~screened]=appearance_decision(ap[~screened],self.config_['appearance_confidence'])
        details=df[['ID','consistency','before_shape','after_shape']].copy().reset_index(drop=True)
        details['shape_fail_probability']=cp;details['shape_stage']=np.where(screened,'Fail','Uncertain')
        details['appearance_fail_probability']=ap;details['final_prediction']=result
        details['route']=np.where(screened,'shape_direct_fail','appearance_model')
        return details

    def predict(self,X,df):
        return self.predict_details(X,df).final_prediction.to_numpy()
