# Pulmonary-Fibrosis-Progression

## Table of Contentss
* [Pulmonary-Fibrosis-Progression](#Pulmonary-Fibrosis-Progression)
* [Symptoms](#Symptoms)
* [Motivation](#Motivation)
* [Library](#Library)
* [DATA](#DATA)
  * [CT_Scan](#CT_Scan)
  * [TABULAR_DATA](#TABULAR_DATA)
   * [Attributes](#Attributes)
* [AIM](#AIM)
* [Requirement](#Requirement)
 * [Lung_Segmentation](#Lung_Segmentation)
 * [Feature_Extraction].(#Feature_Extraction)
* [Sampler](#Sampler)
* [Feature_Engineering](#Feature_Engineering)
* [Metric](#Metric)
* [Confidence](#confidence)
* [Model](#Model)
* [Loss](#Loss)
* [Overview](#overview)

## Pulmonary-Fibrosis-Progression
Pulmonary, meaning lung, and fibrosis, meaning scar tissue.Pulmonary fibrosis is a condition in which the lungs become scarred over time and The scarring of lung tissue makes it thick and stiff. As the lung tissue thickens, it becomes increasingly difficult for the body to transfer oxygen from the lungs into the bloodstream. As a result, the brain and other organs may not receive enough oxygen. Scarring may also increase the risk of lung cancer.
## Symptoms 
Symptoms include shortness of breath, a dry cough, feeling tired,weight loss,and nail clubbing.Complications may include pulmonary hypertension, respiratory failure, pneumothorax, and lung cancer.

## Motivation
To diagnose your condition, your doctor may review your medical and family history, discuss your signs and symptoms, review any exposure you've had to dusts, gases and chemicals,
and conduct a physical exam. During the physical exam, your doctor will use a stethoscope to listen carefully to your lungs while you breathe. He or she may also suggest one or more of the following tests such as Chest X-ray,Computerized tomography (CT) scan,Echocardiogram.but doctors aren’t easily able to tell where an individual may fall on that spectrum.**Till Now there is no Direct Diagnosis for the  Pulmonary Fibrosis** .

## DATA
 ## CT_SCAN
  Baseline chest CT scan of patients is provided. CT Scan is in Dicom file(Digital Imaging and Communications in Medicine).
Inser ct scan image,meta data
 ## TABULAR_DATA  (Measuring_FVC)
  A patient has an image acquired at time Week = 0(CT_scan) and has numerous follow up visits over the course of approximately 1-2 years, at which time their FVC is measured.
  Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (FVC), i.e. the volume of air exhaled.
  ### Attributes
   **Patient**- a unique Id for each patient (also the name of the patient's DICOM folder).<br/>
   **Weeks**- the relative number of weeks pre/post the baseline CT (may be negative).<br/>
   **FVC** - the recorded lung capacity in ml (through Spirometer).<br/>
   **Percent**- a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics.<br/>
   **Age** - Age of the Patient<br/>.
   **Sex** - (Male/Female)<br/>
   **SmokingStatus** - In these three type of labels :- Currently_Smoking,Ex_Smokers, Never_smoked.<br/>

show how the FVC is decay wrt to week

# AIM
Predict Lungs Function Decline based on using  output of a spirometer and CT scan.
code to import dicom file.....
# Lung_Segmentation
## Step by Step
```python
ID='ID00210637202257228694086' #Unique Patient ID
path='/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+ID
slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
slices=np.array(slices)
loc=[s.InstanceNumber for s in slices]
slices=slices[np.argsort(loc)]
filename=[s for s in os.listdir(path)]
pixel_array=[s.pixel_array for s in slices]
#Take some random CT Images for demonstration
for p in pixel_array[[8,11,12,14]]:
    plt.imshow(p,cmap='gray')
    plt.show()
```
## These are some Random CT Scan  
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/1.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/2.png" width="200" /> 
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/3.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/4.png" width="200" />
</p>
add image hu value...</br>

```python
#Convert image(pixel_array) to HU(Hounsfield Unit) 
winCenter=slices[0].WindowCenter 
winWidth=slices[0].WindowWidth
RI=slices[0].RescaleIntercept
RS=slices[0].RescaleSlope
yMin = (winCenter - 0.5 * np.abs(winWidth))
yMax = (winCenter + 0.5 * np.abs(winWidth))
#converting all ct image to HU values take a lot computational, so choose work in reverse manner
yMin = (yMin-RI)/RS
yMax = (yMax-RI)/RS
t=np.array(pixel_array).ravel()  
t=t[(t>=yMin)&(t<=yMax)]
#there are range of HU Values see above Table 
#K-means seprates different range of HU value
kmeans=KMeans(n_clusters=2, random_state=0).fit(t.reshape((-1,1)))
for i,p in enumerate(pixel_array[[[8,11,12,14]]]):
    pred=kmeans.predict(np.array(p).reshape(-1,1))#kmeans predicts for two label[0,1]
    pred=pred.reshape(np.array(p).shape)
    start=pred[0][0]
    binary=np.where(pred==start,1,0)
    plt.imshow(binary)
    plt.show()
 ```
## Binary Images
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/1_Binary.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/2_Binary.png" width="200" /> 
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/3_Binary.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/4_Binary.png" width="200" />
</p>

```python
#Median Blur with clear border(Clear objects connected to the label image border)
lungs = median(clear_border(binary))
```
## Median Blur
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/1_Median_Blur.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/2_Median_Blur.png" width="200" /> 
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/3_Median_Blur.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/4_Median_Blur.png" width="200" />
</p>

```python
#Closing is a mathematical morphology operation that consists in the succession of a dilation and an erosion of the input with the same structuring element.
#Closing therefore fills holes smaller than the structuring element.
lungs = morphology.binary_closing(lungs, selem=morphology.disk(7))
```
## Binary Closing
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/1_Binary_Closing.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/2_Binary_Closing.png" width="200" /> 
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/3_Binary_Closing.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/4_Binary_Closing.png" width="200" />
</p>

```python
lungs = scipy.ndimage.binary_fill_holes(lungs)#binary_fill_holes function, which uses mathematical morphology to fill the holes.
lungs = morphology.dilation(lungs,np.ones([5,5]))
```
## Binary_Fill_Holes and Dilation 
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/1_Binary_Fill_HoleDilation.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/2_Binary_Fill_HoleDilation.png" width="200" /> 
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/3_Binary_Fill_HoleDilation.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/4_Binary_Fill_HoleDilation.png" width="200" />
</p>

```python
labels = np.array(measure.label(lungs,connectivity=2))#Measure image regions to filter small objects
lbl_exclude=np.unique([labels[0:60,:],labels[-60:,:]])#there is some ct images that contains table ct scan, so its removed. Top and bottom 60 pixels are removed
label=np.unique(labels)
mask=np.zeros(np.array(p).shape)
for l in label:
    msk=np.where(labels==l,1,0)
    if(( l in lbl_exclude) | (np.sum(msk)<100)|(np.abs(np.mean(msk*p))<1)):
        continue;
    mask=np.add(mask,np.where(labels==l,1,0))
```
## Label Connectivity
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/1_Label_Connectivity.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/2_Label_Connectivity.png" width="200" /> 
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/3_Label_Connectivity.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/4_Label_Connectivity.png" width="200" />
</p>
<br/>
  ## FinallySegmentedImage
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/1_segment.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/2_segment.png" width="200" /> 
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/3_segment.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/4_segment.png" width="200" />
</p>
<br/>
# Feature_Extraction
## Split the Segmentated Images
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/l1.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/r1.png" width="200" /> 
</p>
<br/>
<p float="left">
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/l2.png" width="200" />
  <img src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/r2.png" width="200" />
</p>
<br/>
**Area** :<br/>
**Mean** :<br/>
**Skew** :<br/>
**Kurt** :<br/>

# Sampler
<br/>
<p float="left">
<img align="left" width="200" height="200" src="https://github.com/sahilabs/Pulmonary-Fibrosis-Progression/blob/main/Image/No_ctscan_PerPatient.png" />
</p>
<br/>
***Above the figure shows CT images Per Patient***<br/>
**Why Sampler ?:**  By observing on Average CT Image of Patient  is around 400 which is huge to compute and many CT Scan Image are Almost Similar and it's due to slice length is  very small which makes the layer of image to be Similar. So it's required to creat Sampler which can extract different type of Images.</br>

22221211
