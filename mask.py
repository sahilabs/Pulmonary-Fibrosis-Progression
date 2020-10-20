Mask_Image=[] # ID,file_name,Masked_Image
#Segmented_Image=[] ## ID,file_name,Segmented_Image

for ID in tqdm(Patient_ID):#DP_ID:
    path='/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+ID
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    slices=np.array(slices)
    loc=[s.InstanceNumber for s in slices]
    slices=slices[np.argsort(loc)]
    filename=[s for s in os.listdir(path)]
    pixel_array=[]
    for s in slices:
        try:
            pixel_array.append(s.pixel_array)
        except:
            pass;
    if(len(pixel_array)==0):
        continue;
    ################################################
    if(slices.shape[0]>50):
        change=[]
        for i in range(len(pixel_array)-1):
            curr=pixel_array[i]
            nxt=pixel_array[i+1]
            tmp=np.abs(np.subtract(curr,nxt))
            curr=np.abs(curr)+1
            tmp=np.divide(tmp,curr)
            tmp=np.sum(tmp)
            change.append([i,tmp])
        change=np.array(change)
        change[:,1]=change[:,1]/(nxt.shape[0]*nxt.shape[1])
        pixel_array=np.array(pixel_array)
        ind_=sorted(np.argsort(np.gradient(change[:,1]*100))[::-1][0:50])#
        pixel_array=pixel_array[ind_]
        slices=slices[ind_]
        del ind_,change,curr,nxt,tmp
    else:
        pass;
    #####################################################
    #there are ct images having double padding and common thing in them is they don't have equal number row and col
    if(pixel_array[0].shape[0]!=pixel_array[0].shape[1]):
        tmp=[]
        for p,img in enumerate(pixel_array):
            start=img[0][0]
            ind_x=[]
            ind_y=[]
            for i,r in enumerate(img):
                if(sum(r==start)/img.shape[1]!=1):
                    ind_x.append(i)
            for i,c in enumerate(img.T):
                if(sum(c==start)/img.shape[0]!=1):
                    ind_y.append(i)
            img=img[ind_x[0]:ind_x[-1],ind_y[0]:ind_y[-1]]
            tmp.append(img)
        pixel_array=tmp
    ##########################################################
    winCenter=slices[0].WindowCenter
    winWidth=slices[0].WindowWidth
    try:
        RI=slices[0].RescaleIntercept
    except:
        RI=0;
    try:
        RS=min(slices[0].RescaleSlope,1)
    except:
        RS=1
    #changing the each value of pixel to Hu takes lots resource
    yMin = (winCenter - 0.5 * np.abs(winWidth))
    yMax = (winCenter + 0.5 * np.abs(winWidth))
    yMin = (yMin-RI)/RS
    yMax = (yMax-RI)/RS
    t=np.array(pixel_array).ravel()
    t=t[(t>=yMin)&(t<=yMax)]
    kmeans=KMeans(n_clusters=2, random_state=0).fit(t.reshape((-1,1)))
    ################################################################
    Mask=[]
    for i,p in enumerate(pixel_array):
        pred=kmeans.predict(np.array(p).reshape(-1,1))
        pred=pred.reshape(np.array(p).shape)
        start=pred[0][0]
        pred=np.where(pred==start,1,0)
        #plt.imshow(p)
        #plt.show()
        #plt.imshow(pred)
        #plt.show()
        ################
        binary=pred
        lungs = median(clear_border(binary))
        lungs = morphology.binary_closing(lungs, selem=morphology.disk(7))
        lungs = binary_fill_holes(lungs)
        lungs = morphology.dilation(lungs,np.ones([5,5]))
        #plt.imshow(lungs,cmap='gray')
        #plt.show()
        labels = measure.label(lungs,connectivity=2) # Different labels are displayed in different colorslabel_vals = np.unique(labels)
        #plt.imshow(labels)
        #plt.show()
        labels=np.array(labels)
        lbl_exclude=np.unique([labels[0:60,:],labels[-60:,:]])#there is some ct images that contains table ct scan, so its removed
        label=np.unique(labels)
        #################
        #mean_label=[]
        mask=np.zeros(np.array(p).shape)
        for l in label:
            msk=np.where(labels==l,1,0)
            if(( l in lbl_exclude) | (np.sum(msk)<100)|(np.abs(np.mean(msk*p))<1)):
                continue;
            mask=np.add(mask,np.where(labels==l,1,0))
            #print('label'+' '+str(l)+' ' +str(np.mean(msk*p))+' '+str(np.std(msk*p))+' '+str(np.sum(msk)))
            #mean_label.append([l,np.mean(msk*p)])
        
        mask=mask.astype(np.uint8)
        #print(filename[i])
        #plt.imshow(mask)
        #plt.show()
        inst_num=slices[i].InstanceNumber
        Mask.append([inst_num,mask])
        #Segmented_Image.append([ID,inst_num,mask*p])
    Mask_Image.append([ID,Mask])
    del Mask
