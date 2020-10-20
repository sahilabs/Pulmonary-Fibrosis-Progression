def Sampler(n_size=50):
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
