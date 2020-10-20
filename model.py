features=np.array(['Weeks','Ex-smoker','Never smoked','Age','STD_FVC','Sex_Label','skew_l1','skew_l2','skew_r1','skew_r2','mean_l1','mean_l2','mean_r1','mean_r2','kurt_l1','kurt_l2','kurt_r1','kurt_r2','Base_Percent','Base_Weeks','Base_FVC','Area_l1','Area_l2','Area_r1','Area_r2','Area'])
target=['FVC']
features_cat=['Ex-smoker','Never smoked']
features_numerical=np.array(['Sex_Label','Weeks','Age','STD_FVC','skew_l1','skew_l2','skew_r1','skew_r2','mean_l1','mean_l2','mean_r1','mean_r2','kurt_l1','kurt_l2','kurt_r1','kurt_r2','Base_Percent','Base_Weeks','Base_FVC','Area_l1','Area_l2','Area_r1','Area_r2','Area'])#Tunned Parameters
##Hyperparameter
EPOCHS = 2000
BATCH_SIZE=252
Drop_1=0.395458
Drop_2=0.200271
Drop_3= 0.2#463196
FOLDS=6
Factor=0.25#54209
_lambda=0.748637
lr=0.00320555
patience=50#patience=99
#mean_OOF_val_score=7.72143
Quantile_1 =0.275231
Quantile_2=0.5 #fixed
Quantile_3=0.826674

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


def plot_history(history):
    score = history.history['metric']
    val_score = history.history['val_metric']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    # create subplots
    plt.figure(figsize = (20,5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, score, label = 'Training Score')
    plt.plot(epochs_range, val_score, label = 'Validation Score')
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Score')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label = 'Training Loss')
    plt.plot(epochs_range, val_loss, label = 'Validation Loss')
    # limit y-values for beter zoom-scale
    #plt.ylim(0.3 * np.mean(val_loss), 1.8 * np.mean(val_loss))

    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def get_lr_callback(batch_size=64,plot=False):
    LR_START=0.01
    LR_MAX= 0.0001*batch_size
    LR_MIN=0.0001
    ######
    LR_RAMP_EP=EPOCHS*0.3
    LR_SUS_EP  = 0
    LR_DECAY   = 0.993

    def lr_scheduler(epoch):
        if(epoch<LR_RAMP_EP):
            lr=(LR_MAX-LR_START)/LR_RAMP_EP*epoch+LR_START
        elif(epoch<LR_RAMP_EP+LR_SUS_EP):
            lr=LR_MAX
        else:
            lr=(LR_MAX-LR_MIN)*LR_DECAY**(epoch-LR_RAMP_EP-LR_SUS_EP)+LR_MIN
        return lr
    if(plot==False):
        lr_callback=tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=False)
        return lr_callback
    else:
        return lr_scheduler

#custom callbacks
class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_score=[]
        
    def on_epoch_end(self, epoch, logs=None):
        self.val_score.append(logs['val_metric'])
        if epoch % 250 == 0 or epoch == (EPOCHS -1 ):
            print(f"The average val-loss for epoch {epoch} is {logs['val_loss']:.2f}"
                  f" and the val-score is {logs['val_metric']}")

    def on_train_end(self, logs=None):
        best_epoch = np.argmin(self.val_score)
        # get score in best epoch
        best_score = self.val_score[best_epoch]
        print(f"Stop training, best model was found and saved in epoch {best_epoch + 1} with val-score: {best_score}."
              f" Final results in this fold (last epoch):") 

def get_checkpoint_saver_callback(fold):
    checkpt_saver = tf.keras.callbacks.ModelCheckpoint(
        'fold-%i.h5'%fold,
        monitor = 'val_loss',
        verbose = 0,
        save_best_only = True,
        save_weights_only = True,
        mode = 'min',
        save_freq = 'epoch')
    
    return checkpt_saver

#######################################
def metric(y_true,y_pred):
    sigma_min,max_error_FVC=tf.constant(70,dtype='float32'),tf.constant(1000,dtype='float32')
    tf.dtypes.cast(y_true,tf.float32)
    tf.dtypes.cast(y_pred,tf.float32)
    sigma=y_pred[:,-1]-y_pred[:,0]
    fvc_pred=y_pred[:,1]
    sigma_clipped=tf.maximum(sigma,sigma_min)
    delta=tf.abs(fvc_pred-y_true)
    delta=tf.minimum(delta,max_error_FVC)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype = tf.float32))
    metric_ = (delta / sigma_clipped) * sq2 + tf.math.log(sigma_clipped * sq2)
    return K.mean(metric_)


def qloss(y_true,y_pred,Quantile_1,Quantile_3):
    qs=[Quantile_1,0.5,Quantile_3]
    qs=tf.constant(np.array(qs),dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(qs* e, (qs-1) * e)
    threshold=200
    v=tf.math.abs(v)
    if(K.mean(v)<threshold):
        v=K.mean(v)
    else:
        v=tf.math.square(v)
        v=K.mean(v)
        v=K.sqrt(v)
    return v

def mloss(_lambda,Quantile_1,Quantile_3):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred,Quantile_1,Quantile_3) + (1 - _lambda) * metric(y_true, y_pred)
    return loss

import tensorflow_addons as tfa
def model(Drop_1,Drop_2,Drop_3,_lambda,Quantile_1,Quantile_3,lr=0.01,optimizer='Adam'):
    Input=layers.Input((len(features),))
    x=layers.BatchNormalization()(Input)
    x=tfa.layers.WeightNormalization(layers.Dense(256,activation='elu'))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.4)(x)
    x=tfa.layers.WeightNormalization(layers.Dense(128,activation='elu'))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.2)(x)
    x=tfa.layers.WeightNormalization(layers.Dense(64,activation='elu'))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.2)(x)
    x=tfa.layers.WeightNormalization(layers.Dense(32,activation='elu'))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.1)(x)
    x=tfa.layers.WeightNormalization(layers.Dense(32,activation='elu'))(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.1)(x)
    q1=layers.Dense(3,activation='relu')(x)
    q2=layers.Dense(3,activation='relu')(x)#this will make sure the prediction is in order x<y<z
    pred=layers.Lambda(lambda x:x[0]+tf.cumsum(x[1],axis=1))([q1,q2])
    model_neural=Models.Model(Input,pred)
    
    optimizer=tf.keras.optimizers.Adam(lr=lr) if optimizer=='Adam' else tf.keras.optimizers.SGD()
    _lamda=tf.constant(_lambda,dtype=tf.float32)
    model_neural.compile(loss=mloss(_lamda,Quantile_1,Quantile_3),optimizer=optimizer,metrics=[metric])
    return model_neural
