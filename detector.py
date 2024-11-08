import numpy as np
import matplotlib.pyplot as plt
import time
from utils.evaluator import Evaluator
from tqdm import tqdm
import sys, errno
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
                        prog="detector.py",
                        description="find the positions of R-peaks in a collection of ECG samples"
)
parser.add_argument("database", help="path to data or selected database name (MIT-BIH, INCART, QTDB, etc.)")
parser.add_argument("model", help="trained model weights to use, stored in .pt file")
parser.add_argument("-v", "--visualise", help="visualise examples of poor performance", action='store_true')
args = parser.parse_args()

database = args.database
model_name = args.model

if "no_swt" in model_name:
     use_swt=False 
else: 
    use_swt=True

'''
The current model was developed by training MIT_BIH, INCART, and QT databases.
If you test these databases, you will see the performance in the training set.
Cross-database testing is available when you test MIT_BIH_ST, European_ST_T, and TELE databases.
'''

peak_detector = Evaluator(data=database, model_name=model_name)

### Run peak detection pipeline
print('Database ... {0}'.format(database))
start = time.time()
peak_detector.load(database, use_swt=use_swt)
peak_detector.find_peaks()
end = time.time()
elapsed = end-start
average_cost = elapsed/len(peak_detector.db_loading.metadata_patient)
print('Average elapsed time : {0:.2f}'.format(average_cost))


### Summary of model performance
table_summary = peak_detector.report_all()
table_summary.loc[table_summary.shape[0],:] = peak_detector.report_summary()
#table_summary.index = peak_detector.db_loading.metadata_patient + ['Total']
table_summary = table_summary.round(decimals=4)

print('Summary of model performance')
print(table_summary)

table_summary.to_csv("table_summary.csv")
dict = pd.DataFrame.from_dict(peak_detector.set_dict)
dict.to_csv("set_dict.csv")

### Visualize a specific ECGs
# t_idx = 0
# t_patient = table_summary.index[t_idx]
# t_ecg = peak_detector.set_dict['ecg'][t_idx]
# t_label = peak_detector.set_dict['label'][t_idx]
# t_pred_TP = peak_detector.set_dict['pred_TP'][t_idx]
# t_pred_FP = peak_detector.set_dict['pred_FP'][t_idx]
# t_pred_FN = peak_detector.set_dict['pred_FN'][t_idx]
# t_xtick = np.arange(t_ecg.shape[0])/360

# plt.plot(t_xtick, t_ecg, color='black')
# plt.plot(t_xtick[t_pred_TP], [t_ecg[x] for x in t_pred_TP], 'o', color='green')
# plt.plot(t_xtick[t_pred_FP], [t_ecg[x] for x in t_pred_FP], '*', color='red')
# if len(t_pred_FN) > 0:
#     plt.plot(t_xtick[t_pred_FN], [t_ecg[x] for x in t_pred_FN], '*', color='blue')
# plt.title('Database {}, Patient {}'.format(test_database, t_patient))
# plt.xlabel('Time (s)')
# plt.ylabel('Voltage (mV)')
# plt.savefig(path_to_data + '/ECG-peak-detection/results.png')



# plot the bad examples
if args.visualise:
    for idx in tqdm(table_summary[table_summary["sensitivity"] < 0.5].index):
        if idx == len(peak_detector.set_dict["ecg"]):
            continue
        peak_detector.db_loading.visualise(peak_detector.set_dict, idx)

