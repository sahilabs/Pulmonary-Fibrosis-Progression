def splt_array(array):
    r,c=array.shape
    L1=array[0:r//2,0:c//2]
    L2=array[r//2:r,0:c//2]
    R1=array[0:r//2,c//2:c]
    R2=array[r//2:r,c//2:c]
    return [L1,L2,R1,R2]


def stats_inf(mask,data,on='train'):
    ID_stats=[]
    p=[]
    mean_ID=[]
    stastical_inf=[]
    Area=[]
    for m in tqdm(mask):
        ID=m[0]
        inst_num=np.array(m[1])[:,0]
        masked_image=np.array(m[1])[:,1]
        inst_num=inst_num.astype(str)
        inst_num=[i+'.dcm' for i in inst_num]
        data_ID=data[data.Patient==ID]
        temp_=[]
        AREA_=[]
        #################
        for inst,mask_img in zip(inst_num,masked_image):
            path='/kaggle/input/osic-pulmonary-fibrosis-progression/'+on+'/'+ ID+'/'+inst
            Slice=pydicom.read_file(path)
            array=Slice.pixel_array
            if(array.shape[0]!=array.shape[1]):
                #resize the array
                a,b=mask_img.shape
                x=(array.shape[0]-a)//2
                y=(array.shape[1]-b)//2
                array=array[x:x+a,y:y+b]
            AREA_.append(splt_array(mask_img))
            slope=Slice.RescaleSlope
            intercept=Slice.RescaleIntercept
            array=array*slope+intercept
            masked=array*mask_img
            L1,L2,R1,R2=splt_array(masked)
            L1,L2,R1,R2=[spt[(spt<0)] for spt in [L1,L2,R1,R2] ]
            temp=masked[masked<0]#[(masked<-Slice.WindowCenter) &((masked>-1500))])
            if(len(temp)>0):
                temp_.append([L1,L2,R1,R2])
        ###
        prod=np.prod(Slice.PixelSpacing)
        AREA_=np.sum(np.array(AREA_),axis=0)
        AREA_=[np.sum(ar[ar>0])*prod for ar in AREA_]
        ###
        temp_=np.array(temp_)
        temp_=[np.hstack(t) for t in temp_.T]
        #features
        Mode=[stats.mode(t)[0][0] for t in temp_]
        Median=[np.median(t) for t in temp_]
        Mean=[np.mean(t) for t in temp_]
        Skew=[stats.skew(t) for t in temp_]
        Kurt=[stats.kurtosis(t) for t in temp_]
        stastical_inf.append(np.hstack([ID,Mode,Median,Mean,Skew,Kurt,AREA_]))
    return stastical_inf
  
  
stastical_inf_train=stats_inf(mask_train,data=train,on='train')
columns=['Patient','mode_l1','mode_l2','mode_r1','mode_r2','median_l1','median_l2','median_r1','median_r2','mean_l1','mean_l2','mean_r1','mean_r2','skew_l1','skew_l2','skew_r1','skew_r2','kurt_l1','kurt_l2','kurt_r1','kurt_r2','Area_l1','Area_l2','Area_r1','Area_r2']
stats_train=pd.DataFrame(stastical_inf_train,columns=columns)
stats_train[columns[1:]]=stats_train[columns[1:]].astype(float)
print(stats_train.shape)
stats_train.head()

