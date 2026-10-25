"""Fixed color/shape equality gate followed by the original appearance classifier."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from shape_cascade_model import new_appearance_model, appearance_decision, metrics, CONF_GRID, GAP, SEED
ATTRIBUTES=['before_color','after_color','before_shape','after_shape']

def direct_fail_mask(df):
    a=df[ATTRIBUTES].to_numpy(float)
    if not (np.isfinite(a).all() and (a>0).all() and (a==np.floor(a)).all()):
        raise ValueError('联合相等规则要求四项均为有效正整数类别；缺失/特殊状态不能当成真实颜色或形状。')
    return (a[:,0]!=a[:,1])|(a[:,2]!=a[:,3])

def tune(y,ap,df,rule):
    gate=direct_fail_mask(df) if rule else np.zeros(len(df),dtype=bool)
    best=None
    for threshold in CONF_GRID:
        pred=appearance_decision(ap,float(threshold));pred[gate]=1
        m=metrics(y,pred)
        violation=max(0,m['pass_false_fail_rate']-.10)+max(0,.60-m['coverage'])
        rank=(violation<=1e-12,-violation,m['fail_f1'] or 0,m['balanced_recall_all'],m['accuracy_all'],m['coverage'])
        if best is None or rank>best[0]:best=(rank,{'appearance_confidence':float(threshold),'gap_threshold':GAP,'oof_metrics':m})
    return best[1]

class ColorShapeRuleCascade:
    def fit_select(self,X,df,groups):
        df=df.reset_index(drop=True);y=df.consistency.to_numpy(int)
        gate=direct_fail_mask(df);ap=np.full(len(df),np.nan);folds=np.full(len(df),-1)
        for fold,(fit,valid) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y,groups)):
            assert not set(np.asarray(groups)[fit])&set(np.asarray(groups)[valid])
            model=new_appearance_model().fit(X[fit],y[fit])
            ap[valid]=model.predict_proba(X[valid])[:,1];folds[valid]=fold
        assert np.isfinite(ap).all() and (folds>=0).all()
        self.config_=tune(y,ap,df,True);self.baseline_config_=tune(y,ap,df,False)
        self.appearance_model_=new_appearance_model().fit(X,y)
        self.oof_=df[['ID','consistency',*ATTRIBUTES]].copy()
        self.oof_['fold']=folds;self.oof_['task_group']=groups
        self.oof_['direct_fail']=gate;self.oof_['appearance_fail_probability']=ap
        pred=appearance_decision(ap,self.config_['appearance_confidence']);pred[gate]=1
        self.oof_['cascade_prediction']=pred
        self.oof_['baseline_cv_prediction']=appearance_decision(ap,self.baseline_config_['appearance_confidence'])
        self.oof_['original_07_prediction']=appearance_decision(ap,.7)
        return self

    def predict_details(self,X,df):
        gate=direct_fail_mask(df);pred=np.ones(len(df),dtype=int);ap=np.full(len(df),np.nan)
        if (~gate).any():
            ap[~gate]=self.appearance_model_.predict_proba(X[~gate])[:,1]
            pred[~gate]=appearance_decision(ap[~gate],self.config_['appearance_confidence'])
        result=df[['ID','consistency',*ATTRIBUTES]].copy().reset_index(drop=True)
        result['color_same']=result.before_color==result.after_color
        result['shape_same']=result.before_shape==result.after_shape
        result['rule_stage']=np.where(gate,'Fail','Uncertain')
        result['route']=np.where(gate,'rule_direct_fail','appearance_model')
        result['appearance_fail_probability']=ap;result['final_prediction']=pred
        return result

    def predict(self,X,df):return self.predict_details(X,df).final_prediction.to_numpy()
