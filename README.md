This repo is forked from https://github.com/dactylogram/ECG_peak_detection.git

# ECG_peak_detection
Robust R-peak detection in an electrocardiogram with stationary wavelet transformation and separable convolution

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

## Peak detection
You can see peak detection codes in 'detector.py' and 'detector.ipynb' files. Note that the model was trained by MIT_BIH, INCART, and QT databases and you can see cross-database performance when you test MIT_BIH_ST, European_ST_T, and TELE databases.
