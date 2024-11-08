This repo is forked from https://github.com/dactylogram/ECG_peak_detection.git

# ECG_peak_detection
Robust R-peak detection in an electrocardiogram with stationary wavelet transformation and separable convolution

## How to install

- clone repo
- create virtual environment
- activate venv

- ```` pip install -r requirements.txt ````

## How to use
- update config (instructions at the bottom)
- activate venv
- ```` python detector.py <DATABASE> <MODEL><optional arguments> ````

The detector automatically detects whether to use SWT preprocessing from the model name.

### Alternative use (input: numpy array)
Use in another script:
````
from utils.evaluator import Evaluator

detector = Evaluator(numpy_array)
detector.load(numpy_array)
peaks = detector.find_peaks()
````
### Arguments:
```
Flag           Note                                                                                                  Default
-h, --help     show this help message and exit                
--database     A path to the directory containing VAF files or a string denoting the desired dataset option.            -      
--visualise    optional tag to generate plots of examples with less than 50% sensitivity                              False                          
```
### Example usage:
```
python detector.py MIT_BIH_ST self_trained_model.pt -v 
```
All data classes are equipped with a visualisation method, useful for examining both model input and output. This method can be called from the detector.py script using
```
peak_detector.db_loading.visualise(peak_detector.set_dict, idx)
```
where ````idx```` is the index of the sample to be visualised. The figure is saved as a png and html file in the structure outlined below


```
repository
├─ ...
├─ figures
   ├─ edf
   ├─ publicDBs
      ├─ html
         ├─ idx.html
      ├─ png
         ├─ idx.png
   ├─ training
├─ ...


```

### Output
A dataframe containing the inputs and outputs from the model are saved in "set_dict.csv". 
The "ecg" column contains the raw ECG data, the "label" column contains the ground truth index of beat positions and the
"pred_position" contains the index of beat positions output by the model.

An overview of the model performance is saved in "table_summary.csv".

## Prepare databases
Relevant databases can be downloaded here:
* MIT_BIH: https://physionet.org/content/mitdb/1.0.0/
* INCART: https://physionet.org/content/incartdb/1.0.0/
* QT(DB): https://physionet.org/content/qtdb/1.0.0/
* MIT_BIH_ST: https://physionet.org/content/stdb/1.0.0/
* European_ST_T: https://www.physionet.org/content/edb/1.0.0/
* TELE: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QTG0EP

Your data directory must be ordered like below.
```
database
├─ MIT_BIH
   ├─ l00.atr
   ├─ l00.dat
   ├─ l00.hea
   ├─ ...
   └─ 234.xws
├─ INCART
   ├─ I01.atr
   ├─ ...
   └─ I75.hea
├─ ...
└─ TELE
```

## Config
The project configuration is stored in the ````config.yaml````file. All model parameters and training parameters are defined here. The datapath config MUST be updated to run the ```detector.py```script without updating all data and evaluation classes individually. E.g.

````
#repo config
path_to_data: /PATH/TO/YOUR/DATABASE/FOLDER/
path_to_repository: /PATH/TO/YOUR/LOCAL/REPOSITORY/
````


## Peak detection
You can see peak detection codes in 'detector.py' and 'detector.ipynb' files. Note that the model was trained by MIT_BIH, INCART, and QT databases and you can see cross-database performance when you test MIT_BIH_ST, European_ST_T, and TELE databases.
