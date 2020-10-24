# Pulmonary-Fibrosis-Progression

## Table of Content
* [Pulmonary-Fibrosis-Progression](#Pulmonary-Fibrosis-Progression)
* [Symptoms](#Symptoms)
* [Motivation](#Motivation)
* [Library](#Library)
* [DATA](#DATA)
  * [CT_Scan](#CT_Scan)
  * [TABULAR_DATA](#TABULAR_DATA)
   * [Attributes](#Attributes)
* [AIM](#AIM)
 requirement
 library installation
 * [Lung_Segmentation](#Lung_Segmentation)
 sampler
 feature extraction
 feature engineering
 metric
 confidence
 model
 Loss
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
```python
ID='ID00210637202257228694086'
path='/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+ID
slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
slices=np.array(slices)
loc=[s.InstanceNumber for s in slices]
slices=slices[np.argsort(loc)]
filename=[s for s in os.listdir(path)]
pixel_array=[s.pixel_array for s in slices]
```


