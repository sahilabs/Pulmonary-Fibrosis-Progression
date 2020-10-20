temp_1=train.copy()
temp_2=sub.copy()
data=pd.concat([temp_1,temp_2],axis=0)
temp_1[features_numerical]=(temp_1[features_numerical]-data[features_numerical].mean())/data[features_numerical].std()
temp_2[features_numerical]=(temp_2[features_numerical]-data[features_numerical].mean())/data[features_numerical].std()
X_train=temp_1[features].values#((temp-temp.mean())/temp.std()).values
y_train=temp_1[target].values
X_test = temp_2[features].values
train_preds = np.zeros((X_train.shape[0],3))
test_preds = np.zeros((X_test.shape[0],3))
y_train=y_train.astype(np.float64)

from scipy.optimize import minimize

def score(sigma,FVC_true,FVC_pred):
    delta=np.where(np.abs(FVC_true-FVC_pred)>1000,1000,np.abs(FVC_true-FVC_pred))
    sigma_clipped=np.where(sigma>70,sigma,70)
    sqrt_2=np.sqrt(2)
    return sqrt_2*delta/sigma_clipped+np.log(sqrt_2*sigma_clipped)

def pediction_error(data,pred):
    predict_error=[]
    error_ID=[]
    b_exp_=[]
    for ID in data.Patient.unique():
        temp=data[data.Patient==ID].copy()
        predict=pred[temp.index]
        #predict=predict*(temp['STD_FVC'].values)/100
        temp['FVC_pred']=predict
        base_week,base_fvc,fvc_pred=temp.loc[temp.Weeks==temp.Base_Weeks.values[0]][['Base_Weeks','Base_FVC','FVC_pred']].values[0]
        if(fvc_pred!=0):
            error=int(base_fvc)/int(fvc_pred)
            predict=predict*error
        else:
            error=base_fvc-fvc_pred
            predict=predict+error
            print('zero__')
        
        error_exp=base_fvc-fvc_pred
        Weeks=temp.Weeks.values
        b=pred[data.Patient==ID]
        exp=np.exp(-np.abs(base_week-Weeks)/(np.abs(base_week)+1)*(np.abs(base_fvc-b)/base_fvc))
        exp_error=exp*error_exp
        b_exp=b+exp_error #pred_error[train.Patient==ID]
        
        b_exp_.append(b_exp)
        predict_error.append(predict)
        error_ID.append([ID,error])
    b_exp_=np.hstack(b_exp_)
    predict_error=np.hstack(predict_error)
    return predict_error,b_exp_,error_ID
def prediction_df(data,pred):
    actual=data['FVC'].values
    score_min=[]
    for p,a in zip(pred,actual):
        m=minimize(score,70, args=(a,p))
        score_min.append([m.x[0],m.fun])
    score_min=np.array(score_min)
    prediction_df=pd.DataFrame({'FVC':actual,'FVC_pred':pred,'Diff':np.abs(actual-pred),'Confidence':score_min[:,0],'score':score_min[:,1]})
    prediction_df['FVC']=actual
    prediction_df['FVC_pred']=pred
    prediction_df['Diff']=np.abs(actual-pred)
    prediction_df['Confidence']=score_min[:,0]
    prediction_df['score']=score_min[:,1]
    #print(prediction_df.score.mean())
    return prediction_df

  
  
  #####
  for i in range(RUN):
    #group wise K-Fold 'Patient_ID'
    callbacks=[]
    LOGGING=True
    gkf=GroupKFold(n_splits=FOLDS)
    groups=train.Patient.values
    fold=0
    OOF_val_score=[]
    for train_idx,val_idx in gkf.split(X_train,y_train,groups=groups):
        fold+=1
        print(f"FOLD{fold}")
        callbacks_lr = [get_lr_callback(BATCH_SIZE)]
        reduce_lr_loss=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=Factor,patience=patience,verbose = 1,epsilon = 1e-4,mode = 'min',min_lr = 0.00001)
        #reduce_lr_loss if the model reached to some plateau or loss is not decreasing then LR is reduced so that it won't overshoot#0.23889#50
        callbacks=[reduce_lr_loss]
        if(LOGGING==True):
            callbacks +=  [get_checkpoint_saver_callback(fold),                   
                         CustomCallback(),callbacks_lr]


        model_neural = model(Drop_1,Drop_2,Drop_3,_lambda,Quantile_1,Quantile_3,lr,'Adam')
        history = model_neural.fit(X_train[train_idx], y_train[train_idx], 
                  batch_size = BATCH_SIZE, 
                  epochs = EPOCHS,
                  validation_data = (X_train[val_idx], y_train[val_idx]),
                  callbacks = callbacks,verbose = 0)

        
        model_neural.load_weights('fold-%i.h5'%fold)
        print("Train:", model_neural.evaluate(X_train[train_idx], y_train[train_idx], verbose = 0, batch_size = BATCH_SIZE, return_dict = True))
        print("Val:", model_neural.evaluate(X_train[val_idx], y_train[val_idx], verbose = 0, batch_size = BATCH_SIZE, return_dict = True))
        print('Abs_error',np.abs(model_neural.predict(X_train[val_idx])[:,1]-y_train[val_idx]).mean())
        train_preds[val_idx]=train_preds[val_idx]+ model_neural.predict(X_train[val_idx],
                                             batch_size = BATCH_SIZE,
                                             verbose = 0)
        
        OOF_val_score.append(model_neural.evaluate(X_train[val_idx], y_train[val_idx], verbose = 0, batch_size = BATCH_SIZE, return_dict = True)['metric'])
        print("Predicting Test...") 
        plot_history(history)
        test_preds += model_neural.predict(X_test, batch_size = BATCH_SIZE, verbose = 0)/FOLDS
    print(np.mean(OOF_val_score))
    ############
    abs_mean=np.abs(train_preds[:,1]/(i+1)-train['FVC'].values).mean()
    sigma=(train_preds[:,-1]-train_preds[:,0])/(i+1)
    FVC_pred=train_preds[:,1]/(i+1)
    FVC_actual=train['FVC'].values
    score_mean=score(sigma,FVC_actual,FVC_pred).mean()
    ##################
    sigma_error=(train_preds[:,-1]-train_preds[:,0])*0.707/(i+1)
    pred_error,_,error= pediction_error(train,train_preds[:,1]/(i+1))
    FVC_pred=pred_error
    FVC_actual=train['FVC'].values
    abs_mean_error=np.abs(FVC_pred-train['FVC'].values).mean()
    score_error=score(sigma,FVC_actual,FVC_pred).mean()
    sigma_error_min_max=sigma.min(),sigma.max()
    pred_error_min_max=pred_error.min(),pred_error.max()
    array=[i,np.mean(OOF_val_score),abs_mean,score_mean,abs_mean_error,score_error,sigma_error_min_max,pred_error_min_max]
    final_score.append(array)
train_preds=train_preds/RUN
test_preds=test_preds/RUN



