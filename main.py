##### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# I found that the eeg_recording28 doesn't have enough data,the date for drowsy is only 76800.
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV  # For hyperparameter tuning
from sklearn.feature_selection import SelectKBest, f_classif  # For feature selection
from imblearn.over_sampling import SMOTE
import seaborn as sns
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.io
import matplotlib.pyplot as plt
import os
from scipy import signal

from scipy.fft import fft, fftshift

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

file_names = []
for dirname, _, filenames in os.walk('EEGdata'):
    for filename in filenames:
        file_names.append(os.path.join(dirname, filename))
        # print(os.path.join(dirname, filename))

# print(file_names)
# each trial is about 54 mins
# build 5 order high pass filter
from scipy.signal import butter, lfilter, freqz
# ----- ----- ----- -----
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    x = signal.filtfilt(b, a, data)
    y = signal.filtfilt(b, a, x)
    return y
#starting from Channel 3,/0,1,2,3
# The introduction of the data set in kaggle doesn't provide enough information.
#And we don't know which data are useful I have to plot all the channels.
#There are 14 channels, but the experimenter modified the headset.
#after removing the DC, the EEG amplitude should be around (-100,100).
#According to the plot, I think ['F7','F3','P7','O1','O2','P8','AF4'] are useful channels.
fig, ax = plt.subplots(14,1)
fig.set_figwidth(30)
fig.set_figheight(500)
#fig.set_size_inches(10,10)
mat = scipy.io.loadmat(file_names[0])
data = mat['o']['data'][0,0]
FS = mat['o']['sampFreq'][0][0][0][0]
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
for i in range(14):
    data_ave = data[5000:15000,i+3]-np.mean(data[5000:15000,i+3])
    ax[i].plot(data_ave)
    ax[i].set_title(channels[i])
    ax[i].set_ylim(-100,100)
plt.show()
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#F7,F3,P7,O1,O2,P8,AF4
useful_channels=[4,5,8,9,10,11,16]
useful_channels_names=['F7','F3','P7','O1','O2','P8','AF4']
fig,ax = plt.subplots(7)
fig.set_size_inches(20,40)
j=0
for i in useful_channels:
    data_ave = data[5000:15000,i]-np.mean(data[5000:15000,i])
    ax[j].plot(data_ave)
    ax[j].set_ylim(-200,200)
    ax[j].set_title(channels[i-3])
    j=j+1
plt.show()
marker=128*60*10
#delete file #28 because it doesnot have enough data
useful_file_index = [3,4,5,6,7,10,11,12,13,14,17,18,19,20,21,24,25,26,27,31,32,33,34]
#useful_file_index = np.arange(1,35)
chan_num=7
trail_names=[]
data_focus={}
data_unfocus={}
data_drowsy={}
focus={}
unfocus={}
drowsy={}
#for i in useful_file_index:
i=1
for index,filename in enumerate(filenames):
    if int(filename.split('d')[1].split('.')[0]) in useful_file_index:
        mat = scipy.io.loadmat(file_names[index])
        trail_names.append(filename.split('.')[0])
        data_focus[trail_names[-1]]=mat['o']['data'][0,0][0:marker,useful_channels].copy()
        data_unfocus[trail_names[-1]]=mat['o']['data'][0,0][marker:2*marker,useful_channels].copy()
        data_drowsy[trail_names[-1]]=mat['o']['data'][0,0][2*marker:3*marker,useful_channels].copy()
        focus[trail_names[-1]]=mat['o']['data'][0,0][0:marker,useful_channels].copy()
        unfocus[trail_names[-1]]=mat['o']['data'][0,0][marker:2*marker,useful_channels].copy()
        drowsy[trail_names[-1]]=mat['o']['data'][0,0][2*marker:3*marker,useful_channels].copy()
data_focus.keys()
data_focus['eeg_record3']

# High Pass 0.16HZ
row, col = data_focus['eeg_record3'].shape
for name in trail_names:
    for i in range(col):
        data_focus[name][:,i]=butter_highpass_filter(data_focus[name][:,i], 0.16, 128, 5)
        data_unfocus[name][:,i]=butter_highpass_filter(data_unfocus[name][:,i], 0.16, 128, 5)
        data_drowsy[name][:,i]=butter_highpass_filter(data_drowsy[name][:,i], 0.16, 128, 5)
        #print(name,data_drowsy[name][:,i].shape)
feature_names = []
freq_range=np.arange(0.5,18.5,0.5)
symb='_'
#useful_channels_names=['F7','F3','P7','O1','O2','P8','AF4']
for index,channel in enumerate(useful_channels_names):
    for freq in freq_range:
        feature_names.append(channel+symb+str(freq))
feature_names

#validate the high pass filter
fig,ax = plt.subplots(1)
fig.set_size_inches(60,10)
color = 'tab:red'
ax.set_xlabel('X-axis')
ax.set_ylabel('Y1-axis', color = color)
ax.plot(unfocus['eeg_record3'][0:500,0], color = color,label='raw signal')
ax.tick_params(axis ='y', labelcolor = color)
ax.set_ylim(3500,4100)
# Adding Twin Axes to plot using dataset_2
ax0 = ax.twinx()
color = 'tab:green'
ax0.plot(data_unfocus['eeg_record3'][0:500,0],color=color,label='filtered signal')
ax0.set_ylim(-150,150)
fig.legend()
plt.show()

#validate the high pass filter
fig,ax = plt.subplots(1)
fig.set_size_inches(60,10)
color = 'tab:red'
ax.set_xlabel('X-axis')
ax.set_ylabel('Y1-axis', color = color)
ax.plot(unfocus['eeg_record33'][0:500,0], color = color,label='raw signal')
ax.tick_params(axis ='y', labelcolor = color)
ax.set_ylim(3800,4200)
# Adding Twin Axes to plot using dataset_2
ax0 = ax.twinx()
color = 'tab:green'
ax0.plot(data_unfocus['eeg_record33'][0:500,0],color=color,label='filtered signal')
ax0.set_ylim(-100,100)
fig.legend()
plt.show()

# STFT was then calculated at a time step of 1 s producing a set of time-varying DFT
# amplitudes X STFT (t,ω) at 1s intervals within each input EEG channel.
t_win = np.arange(0, 128)
M = 128
window_blackman = 0.42 - 0.5 * np.cos((2 * np.pi * t_win) / (M - 1)) + 0.08 * np.cos(
    (4 * np.pi * t_win) / (M - 1))  # window_blackman = signal.windows.blackmanharris(128)

# col is 7
power_focus = {}
for name in trail_names:
    power_focus[name] = np.zeros([col, 513, 601])

power_unfocus = {}
for name in trail_names:
    power_unfocus[name] = np.zeros([col, 513, 601])

power_drowsy = {}
for name in trail_names:
    power_drowsy[name] = np.zeros([col, 513, 601])

# the output of the stft is 513*601,1 second data will produce 1 column of data,there are 601
for name in trail_names:
    for i in range(col):
        f, t, y1 = scipy.signal.stft(data_focus[name][:, i], fs=128, window=window_blackman, nperseg=128,
                                     noverlap=0, nfft=1024, detrend=False, return_onesided=True, boundary='zeros',
                                     padded=True)
        f, t, y2 = scipy.signal.stft(data_unfocus[name][:, i], fs=128, window=window_blackman, nperseg=128,
                                     noverlap=0, nfft=1024, detrend=False, return_onesided=True, boundary='zeros',
                                     padded=True)
        f, t, y3 = scipy.signal.stft(data_drowsy[name][:, i], fs=128, window=window_blackman, nperseg=128,
                                     noverlap=0, nfft=1024, detrend=False, return_onesided=True, boundary='zeros',
                                     padded=True)
        power_focus[name][i, :, :] = (np.abs(y1)) ** 2
        power_unfocus[name][i, :, :] = (np.abs(y2)) ** 2
        power_drowsy[name][i, :, :] = (np.abs(y3)) ** 2

#validate the power value
fig,ax = plt.subplots(col,1)
fig.set_size_inches(40,100)
for i in range(col):
    ax[i].pcolormesh(t, f, power_focus['eeg_record18'][i,:,:],vmin=0, vmax=2*np.sqrt(2), shading='gouraud')
plt.show()
fig,ax = plt.subplots(col,1)
fig.set_size_inches(20,50)
for i in range(col):
    ax[i].pcolormesh(t, f, power_focus['eeg_record33'][i,:,:],vmin=0, vmax=2*np.sqrt(2), shading='gouraud')
plt.show()
plt.plot(power_focus['eeg_record18'][2,1,:])
plt.show()
plt.plot(power_focus['eeg_record33'][2,1,:])
plt.show()
# combine bins into 0.5HZ, and keep 0-18 HZ.

num = []

power_focus_bin = {}
for name in trail_names:
    power_focus_bin[name] = np.zeros([7, 36, 601])

power_unfocus_bin = {}
for name in trail_names:
    power_unfocus_bin[name] = np.zeros([7, 36, 601])

power_drowsy_bin = {}
for name in trail_names:
    power_drowsy_bin[name] = np.zeros([7, 36, 601])

for name in trail_names:
    for chn in range(col):
        j = 0
        for i in range(1, 144, 4):
            power_focus_bin[name][chn, j, :] = np.average(power_focus[name][chn, i:i + 4, :], axis=0)
            power_unfocus_bin[name][chn, j, :] = np.average(power_unfocus[name][chn, i:i + 4, :], axis=0)
            power_drowsy_bin[name][chn, j, :] = np.average(power_drowsy[name][chn, i:i + 4, :], axis=0)
            # print(np.average(power_drowsy[name][chn,i:i+4,:],axis=0).shape)
            # if name=='eeg_record3':
            #    if chn==0:
            #        num.append((f[i:i+4]))
            #    print(j)
            j = j + 1

# print(num)
# print(len(num))
# avarage over 15 seconds running window.

power_focus_ave = {}
for name in trail_names:
    power_focus_ave[name] = np.zeros([7, 36, 585])

power_unfocus_ave = {}
for name in trail_names:
    power_unfocus_ave[name] = np.zeros([7, 36, 585])

power_drowsy_ave = {}
for name in trail_names:
    power_drowsy_ave[name] = np.zeros([7, 36, 585])

for name in trail_names:
    for chn in range(col):
        j = 0
        for k in range(0, 585):
            power_focus_ave[name][chn, :, j] = np.average(power_focus_bin[name][chn, :, k:k + 15], axis=1)
            power_unfocus_ave[name][chn, :, j] = np.average(power_unfocus_bin[name][chn, :, k:k + 15], axis=1)
            power_drowsy_ave[name][chn, :, j] = np.average(power_drowsy_bin[name][chn, :, k:k + 15], axis=1)
            # print(np.average(power_drowsy_bin[name][chn,:,k:k+15],axis=1).shape)
            j = j + 1

fig,ax = plt.subplots(col,1)
t1=np.arange(0,585)
f1=np.arange(0,18,0.5)
fig.set_size_inches(10,20)
for i in range(col):
    ax[i].pcolormesh(t1, f1, power_focus_ave['eeg_record18'][i,:,:],vmin=0, vmax=2*np.sqrt(2), shading='gouraud')
plt.show()

# Turn the data into a vector
# [252,585]

svm_focus = {}
for name in trail_names:
    svm_focus[name] = np.zeros([252, 585])

svm_unfocus = {}
for name in trail_names:
    svm_unfocus[name] = np.zeros([252, 585])

svm_drowsy = {}
for name in trail_names:
    svm_drowsy[name] = np.zeros([252, 585])

for name in trail_names:
    for j in range(585):
        svm_focus[name][:, j] = power_focus_ave[name][:, :, j].reshape(1, -1)
        svm_unfocus[name][:, j] = power_unfocus_ave[name][:, :, j].reshape(1, -1)
        svm_drowsy[name][:, j] = power_drowsy_ave[name][:, :, j].reshape(1, -1)
    svm_focus[name] = 10 * np.log(svm_focus[name])
    svm_unfocus[name] = 10 * np.log(svm_unfocus[name])
    svm_drowsy[name] = 10 * np.log(svm_drowsy[name])
# now, we get the svm vector 252*585 252 rows
svm_focus['eeg_record18'].shape
# along the horizontal direction ,it shows one channel at specific frequecy changes with time.
# each channel has 36 rows of data,
# the following shows 7 channels, each with 36 slots of frequency.
fig,ax = plt.subplots(5,1)
fig.set_size_inches(10,150)
for i in range(5):
    ax[i].plot(svm_focus['eeg_record18'][i,:],label='focus')
    ax[i].plot(svm_drowsy['eeg_record18'][i,:],c='r',label='drowsy')
    ax[i].plot(svm_unfocus['eeg_record18'][i,:],c='green',label='unfocus')
    ax[i].legend()
    ax[i].set_title("The first channel with Frequency at: "+ str((i+1)*0.5))

plt.show()

# along the horizontal direction ,it shows one channel at specific frequecy changes with time.
# each channel has 5 rows of data,
# the following shows 7 channels, each with 5 slots of frequency.
fig,ax = plt.subplots(5,1)
fig.set_size_inches(10,150)
for i in range(5):
    ax[i].plot(svm_focus['eeg_record5'][i,:],label='focus')
    ax[i].plot(svm_drowsy['eeg_record5'][i,:],c='r',label='drowsy')
    ax[i].plot(svm_unfocus['eeg_record5'][i,:],c='green',label='unfocus')
    ax[i].legend()
    ax[i].set_title("The first channel with Frequency at: "+ str((i+1)*0.5) )

plt.show()

#at a specific time, each channel changes with frequency
for i in range(0,25,5):
    plt.plot(svm_focus[trail_names[i]][:,1])
# the peak is at the lowest frequency.
#Each peak is the start of the data.
plt.title("The first channel with Frequency at: "+ str((21+1)*0.5))
plt.show()
#--------0
label_focus = [0]*585
#--------1
label_unfocus = [1]*585
#--------2
label_drowsy = [2]*585

#subject is the variable for all participants

subj1_files={'eeg_record3','eeg_record4','eeg_record5','eeg_record6','eeg_record7'}
subj2_files={'eeg_record10','eeg_record11','eeg_record12','eeg_record13','eeg_record14'}
subj3_files={'eeg_record17','eeg_record18','eeg_record19','eeg_record20','eeg_record21'}
subj4_files={'eeg_record24','eeg_record25','eeg_record26','eeg_record27'}
subj5_files={'eeg_record31','eeg_record32','eeg_record33','eeg_record34'}
# I will try to use the data from all participants to train the model
target=[]
subj=np.array([]).reshape(252,0).copy()
for name in trail_names:
    subj=np.concatenate((subj,svm_focus[name]), axis=1)
    subj=np.concatenate((subj,svm_unfocus[name]), axis=1)
    subj=np.concatenate((subj,svm_drowsy[name]), axis=1)
    target = target+label_focus+label_unfocus+label_drowsy
subj=subj.T
target = np.array(target)
subj.shape
print('length of the target:',len(target))
print('the shape of the data from the subject:', subj.shape)
target1=[]
subj1=np.array([]).reshape(252,0).copy()
for name in subj1_files:
    subj1=np.concatenate((subj1,svm_focus[name]), axis=1)
    subj1=np.concatenate((subj1,svm_unfocus[name]), axis=1)
    subj1=np.concatenate((subj1,svm_drowsy[name]), axis=1)
    target1 = target1+label_focus+label_unfocus+label_drowsy
subj1=subj1.T
target1 = np.array(target1)
print('length of the target1:',len(target1))
print('the shape of the data from the subject1:', subj1.shape)
target2=[]
subj2=np.array([]).reshape(252,0).copy()
for name in subj2_files:
    subj2=np.concatenate((subj2,svm_focus[name]), axis=1)
    subj2=np.concatenate((subj2,svm_unfocus[name]), axis=1)
    subj2=np.concatenate((subj2,svm_drowsy[name]), axis=1)
    target2 = target2+label_focus+label_unfocus+label_drowsy
subj2=subj2.T
target2 = np.array(target2)
print('length of the target2:',len(target2))
print('the shape of the data from the subject2:', subj2.shape)
target3=[]
subj3=np.array([]).reshape(252,0).copy()
for name in subj3_files:
    subj3=np.concatenate((subj3,svm_focus[name]), axis=1)
    subj3=np.concatenate((subj3,svm_unfocus[name]), axis=1)
    subj3=np.concatenate((subj3,svm_drowsy[name]), axis=1)
    target3 = target3+label_focus+label_unfocus+label_drowsy
subj3=subj3.T
target3 = np.array(target3)
print('length of the target3:',len(target3))
print('the shape of the data from the subject3:', subj3.shape)
target4=[]
subj4=np.array([]).reshape(252,0).copy()
for name in subj4_files:
    subj4=np.concatenate((subj4,svm_focus[name]), axis=1)
    subj4=np.concatenate((subj4,svm_unfocus[name]), axis=1)
    subj4=np.concatenate((subj4,svm_drowsy[name]), axis=1)
    target4 = target4+label_focus+label_unfocus+label_drowsy
subj4=subj4.T
target4 = np.array(target4)
print('length of the target4:',len(target4))
print('the shape of the data from the subject4:', subj4.shape)
target5=[]
subj5=np.array([]).reshape(252,0).copy()
for name in subj5_files:
    subj5=np.concatenate((subj5,svm_focus[name]), axis=1)
    subj5=np.concatenate((subj5,svm_unfocus[name]), axis=1)
    subj5=np.concatenate((subj5,svm_drowsy[name]), axis=1)
    target5 = target5+label_focus+label_unfocus+label_drowsy
subj5=subj5.T
target5 = np.array(target5)
print('length of the target5:',len(target5))
print('the shape of the data from the subject5:', subj5.shape)

data_train, data_test, data_train_target, data_test_target = train_test_split(subj, target, test_size=0.8, random_state=0)
scaler = preprocessing.StandardScaler().fit(data_train)
X_train_scaled = scaler.transform(data_train)
X_test_scaled = scaler.transform(data_test)
data_train1, data_test1, data_train_target1, data_test_target1 = train_test_split(subj1, target1, test_size=0.8, random_state=0)
scaler = preprocessing.StandardScaler().fit(data_train1)
X_train_scaled1 = scaler.transform(data_train1)
X_test_scaled1 = scaler.transform(data_test1)
data_train2, data_test2, data_train_target2, data_test_target2 = train_test_split(subj2, target2, test_size=0.8, random_state=0)
scaler = preprocessing.StandardScaler().fit(data_train2)
X_train_scaled2 = scaler.transform(data_train2)
X_test_scaled2 = scaler.transform(data_test2)
data_train3, data_test3, data_train_target3, data_test_target3 = train_test_split(subj3, target3, test_size=0.8, random_state=0)
scaler = preprocessing.StandardScaler().fit(data_train3)
X_train_scaled3 = scaler.transform(data_train3)
X_test_scaled3 = scaler.transform(data_test3)
data_train4, data_test4, data_train_target4, data_test_target4 = train_test_split(subj4, target4, test_size=0.8, random_state=0)
scaler = preprocessing.StandardScaler().fit(data_train4)
X_train_scaled4 = scaler.transform(data_train4)
X_test_scaled4 = scaler.transform(data_test4)
data_train5, data_test5, data_train_target5, data_test_target5 = train_test_split(subj5, target5, test_size=0.8, random_state=0)
scaler = preprocessing.StandardScaler().fit(data_train5)
X_train_scaled5 = scaler.transform(data_train5)
X_test_scaled5 = scaler.transform(data_test5)
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled,data_train_target)
print(f'The Score for Training data with SVM Model for subject:',svm.score(X_train_scaled,data_train_target))
print(f'Score of For Test data with SVM Model for subject : {svm.score(X_test_scaled,data_test_target)}')
print(f'The Score for Training data with SVM Model for subject1:',svm.score(X_train_scaled1,data_train_target1))
print(f'Score of For Test data with SVM Model for subject1 : {svm.score(X_test_scaled1,data_test_target1)}')
print(f'The Score for Training data with SVM Model for subject2:',svm.score(X_train_scaled2,data_train_target2))
print(f'Score of For Test data with SVM Model for subject2 : {svm.score(X_test_scaled2,data_test_target2)}')
print(f'The Score for Training data with SVM Model for subject3:',svm.score(X_train_scaled3,data_train_target3))
print(f'Score of For Test data with SVM Model for subject3 : {svm.score(X_test_scaled3,data_test_target3)}')
print(f'The Score for Training data with SVM Model for subject4:',svm.score(X_train_scaled4,data_train_target4))
print(f'Score of For Test data with SVM Model for subject4 : {svm.score(X_test_scaled4,data_test_target4)}')
print(f'The Score for Training data with SVM Model for subject5:',svm.score(X_train_scaled5,data_train_target5))
print(f'Score of For Test data with SVM Model for subject5 : {svm.score(X_test_scaled5,data_test_target5)}')
# Use 60% data for test
# acuracy, precision, recall and F1 score
svm.fit(X_train_scaled, data_train_target)

# Evaluate the model on all subjects (looping through test data)
for subject_data, subject_target, test_data, test_target in [
    (subj, target, X_test_scaled, data_test_target),
    (subj1, target1, X_test_scaled1, data_test_target1),
    (subj2, target2, X_test_scaled2, data_test_target2),
    (subj3, target3, X_test_scaled3, data_test_target3),
    (subj4, target4, X_test_scaled4, data_test_target4),
    (subj5, target5, X_test_scaled5, data_test_target5),
]:
  # Make predictions on the test data
  predictions = svm.predict(test_data)

  # Print classification report
  print(f"\nClassification Report for Subject Data: {subject_data.shape}")
  print(classification_report(test_target, predictions))

  results = {
      "All Subjects": {"accuracy": 0.90, "precision": 0.9, "recall": 0.91, "f1": 0.90},
      "Subject 1": {"accuracy": 0.82, "precision": 0.80, "recall": 0.85, "f1": 0.82},
      "Subject 2": {"accuracy": 0.84, "precision": 0.84, "recall": 0.87, "f1": 0.86},
      "Subject 3": {"accuracy": 0.84, "precision": 0.82, "recall": 0.89, "f1": 0.85},
      "Subject 4": {"accuracy": 0.69, "precision": 0.69, "recall": 0.85, "f1": 0.76},
      "Subject 5": {"accuracy": 0.78, "precision": 0.73, "recall": 0.83, "f1": 0.78},
  }

  # Extract metric names and subject names
  metric_names = list(results["Subject 1"].keys())
  subject_names = list(results.keys())

  # Create a subplot for each metric
  fig, axes = plt.subplots(2, 2, figsize=(10, 6))

  # Loop through metrics and subjects to plot bars
  metric_index = 0
  for row in range(2):
      for col in range(2):
          metric_name = metric_names[metric_index]
          metric_data = [results[subject][metric_name] for subject in subject_names]
          axes[row, col].bar(subject_names, metric_data)
          axes[row, col].set_title(metric_name)
          axes[row, col].set_xlabel("Subject")
          axes[row, col].set_ylabel(metric_name)
          metric_index += 1

  # Adjust layout and show plot
  fig.suptitle("Classification Performance Metrics")
  plt.tight_layout()
  plt.show()

  #end


data_train, data_test, data_train_target, data_test_target = train_test_split(subj, target, test_size=0.6, random_state=0)
scaler = preprocessing.StandardScaler().fit(data_train)
X_train_scaled = scaler.transform(data_train)
X_test_scaled = scaler.transform(data_test)
#PCA should be used on scaled data
pca = PCA()
pca.fit(X_train_scaled)
X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
var = pca.explained_variance_/pca.explained_variance_.sum()
var[0:20].sum()
X_train_scaled.shape
X_train_pca.shape
data_train_target.shape
data_corr = pd.DataFrame(X_train_scaled).corr()
sns.heatmap(data_corr)
plt.show()
data_corr1 = pd.DataFrame(X_test_scaled).corr()
sns.heatmap(data_corr1)
plt.show()
pca_map=pd.DataFrame(pca.components_,columns=feature_names,index=np.arange(1,253))
#pca.components_.shape
pca_map
plt.plot(pca.components_[0,:])
x=np.where(pca.components_[0,:]>0.085)
x=np.array(x)
for i in x[0]:
    plt.scatter(x,pca.components_[0,x],c='r',label=feature_names[i])
#plt.legend()
plt.show()
plt.plot(pca.components_[2,:])
plt.plot(pca.components_[1,:])
x=np.where(pca.components_[1,:]>0.085)
x=np.array(x)
for i in x[0]:
    plt.scatter(x,pca.components_[1,x],c='r',label=feature_names[i])
#plt.legend()
plt.show()
#USE SVM linear model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
svm = SVC(kernel='linear')
svm.fit(X_train_scaled,data_train_target)
print(f'The Score for Training data with SVM Linear Model for all subjects:',svm.score(X_train_scaled,data_train_target))
print(f'Score of For Test data with SVM Linear Model for all subjects : {svm.score(X_test_scaled,data_test_target)}')


# Evaluate the model on all subjects (looping through test data)
for subject_data, subject_target, test_data, test_target in [
    (subj, target, X_test_scaled, data_test_target),
]:
  # Make predictions on the test data
  predictionL = svm.predict(test_data)

  # Print classification report
  print(f"\nClassification Report for Subject Data with SVM Linear Model: {subject_data.shape}")
  print(classification_report(test_target, predictionL))
#Use RBF model
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled,data_train_target)
print(f'The Score for Training data with SVM RBF Model for all subjects:',svm.score(X_train_scaled,data_train_target))
print(f'Score of For Test datawith SVM RBF Model for all subjects : {svm.score(X_test_scaled,data_test_target)}')
for subject_data, subject_target, test_data, test_target in [
    (subj, target, X_test_scaled, data_test_target),
]:
  # Make predictions on the test data
  predictionRBF = svm.predict(test_data)

  # Print classification report
  print(f"\nClassification Report for Subject Data with SVM RBF Model: {subject_data.shape}")
  print(classification_report(test_target, predictionRBF))
svm = SVC(kernel='rbf')
svm.fit(X_train_pca[:,0:30],data_train_target)
print(f'The Score for Training data with SVM RBF Model for all subjects:',svm.score(X_train_pca[:,0:30],data_train_target))
print(f'Score of For Test data with SVM RBF Model for all subjects : {svm.score(X_test_pca[:,0:30],data_test_target)}')
#Try KNN Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
neighbor = KNeighborsClassifier(n_neighbors=3)
neighbor.fit(X_train_scaled,data_train_target)
print("the score for training with KNN Model from 5 participants:",neighbor.score(X_train_scaled,data_train_target))
print("the score for test data with KNN Model from 5 participants:",neighbor.score(X_test_scaled,data_test_target))
for subject_data, subject_target, test_data, test_target in [
    (subj, target, X_test_scaled, data_test_target),
]:
  # Make predictions on the test data
  predictionN = neighbor.predict(test_data)

  # Print classification report
  print(f"\nClassification Report for Subject Data with KNN Model: {subject_data.shape}")
  print(classification_report(test_target, predictionN))

neighbor.fit(X_train_pca[:,0:30],data_train_target)
print("the score for training with data from 5 participants:",neighbor.score(X_train_pca[:,0:30],data_train_target))
print("the score for test data from 5 participants:",neighbor.score(X_test_pca[:,0:30],data_test_target))

# decision trees
from sklearn.tree import DecisionTreeClassifier                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
dt = DecisionTreeClassifier(max_depth=16)
dt.fit(X_train_scaled,data_train_target)
print(f"the score for training with decision trees data from 5 participants:{dt.score(X_train_scaled,data_train_target)}")
print("the score for test decision trees data from 5 participants:",dt.score(X_test_scaled,data_test_target))

for subject_data, subject_target, test_data, test_target in [
    (subj, target, X_test_scaled, data_test_target),
]:
  # Make predictions on the test data
  predictionD = dt.predict(test_data)

  # Print classification report
  print(f"\nClassification Report for Subject Data with decision trees Model: {subject_data.shape}")
  print(classification_report(test_target, predictionD))

  resultsM = {
      "SVM Model": {"accuracy": 0.90, "precision": 0.9, "recall": 0.91, "f1": 0.90},
      "Linear Model": {"accuracy": 0.73, "precision": 0.76, "recall": 0.79, "f1": 0.77},
      "RBF Model": {"accuracy": 0.93, "precision": 0.93, "recall": 0.94, "f1": 0.93},
      "KNN Model": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99, "f1": 0.99},
      "Decision Trees": {"accuracy": 0.85, "precision": 0.86, "recall": 0.85, "f1": 0.85},

  }

  # Extract metric names and subject names
  metric_names = list(resultsM["SVM Model"].keys())
  subject_names = list(resultsM.keys())

  # Create a subplot for each metric
  fig, axes = plt.subplots(2, 2, figsize=(10, 6))

  # Loop through metrics and subjects to plot bars
  metric_index = 0
  for row in range(2):
      for col in range(2):
          metric_name = metric_names[metric_index]
          metric_data = [resultsM[subject][metric_name] for subject in subject_names]
          axes[row, col].bar(subject_names, metric_data)
          axes[row, col].set_title(metric_name)
          axes[row, col].set_xlabel("Subject")
          axes[row, col].set_ylabel(metric_name)
          metric_index += 1

  # Adjust layout and show plot
  fig.suptitle("accuracy outcomes of various models")
  plt.tight_layout()
  plt.show()

from sklearn.ensemble import RandomForestClassifier

train_scores = []
test_scores = []
for depth in range(1, 35):
    dt_reg = RandomForestClassifier(max_depth=depth, random_state=0)
    dt_reg.fit(X_train_scaled, data_train_target)
    train_scores = train_scores + [dt_reg.score(X_train_scaled, data_train_target)]
    test_scores = test_scores + [dt_reg.score(X_test_scaled, data_test_target)]

x = list(range(1, 35))
plt.plot(x, train_scores, c='r', label='train')
plt.plot(x, test_scores, c='b', label='test')
plt.xlabel('depth')
plt.ylabel('accuracy')
plt.title('Depth vs. Accuracy for Random Forest Classifier')
plt.show()
from sklearn.model_selection import KFold
import tensorflow as tf

tf.random.set_seed(0)
kfold = KFold(n_splits=5, shuffle=True)

data_train_target = data_train_target

fold_no = 1

score_tr = []
score_cv = []

svm = SVC(kernel='rbf')

for train, cv in kfold.split(X_train_scaled, data_train_target):
    X_tr = X_train_scaled[train]
    Y_tr = data_train_target[train]
    X_cv = X_train_scaled[cv]
    Y_cv = data_train_target[cv]

    tf.random.set_seed(0)

    svm.fit(X_tr, Y_tr)
    score_tr.append(svm.score(X_tr, Y_tr))
    score_cv.append(svm.score(X_cv, Y_cv))

    print(f'Score for {fold_no} Fold Training: {score_tr[-1]:.3f}')
    print(f'Score for {fold_no} Fold cv    : {score_cv[-1]:.3f}')
    print('----------------------------------')
    fold_no = fold_no + 1

print(f'Score of Average For Training: {np.mean(score_tr):.3f}')
print(f'Score of Average For CV.     : {np.mean(score_cv):.3f}')
