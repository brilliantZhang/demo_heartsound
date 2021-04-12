import os
import numpy as np
import librosa
import librosa.display
from scipy.fftpack import next_fast_len
from scipy.signal import spectrogram
from python_speech_features import mfcc
from scipy import signal

def getMFCCMap(y, sr=1000):
    mfcc0 = mfcc(y, sr, numcep=13)
    mf1 = librosa.feature.delta(mfcc0) #一阶差分
    mf2 = librosa.feature.delta(mfcc0, order=2) #二阶差分
    mfcc_all=np.hstack((mfcc0,mf1,mf2))
    mfcc_all=mfcc_all.reshape(1,299,39,1) #合并图谱
    return mfcc_all

def get_drop_list():
    drop_list=[]
    data_dir='CinC2016/total/'
    for file in ['RECORDS-problem']:
        path='CinC2016/total/'+file
        f = open(path)
        tmp=f.read()
        f.close()
        filelist=list(tmp.split())  
        drop_list.extend(filelist)

    return drop_list

def get_file_list(path,label,drop_list):
    data_dir='CinC2016/total/'
    f = open(path)
    tmp=f.read()
    f.close()
    filelist=list(tmp.split())
    filelist=[data_dir+file+'.wav' for file in filelist if file not in drop_list]
    print(len(filelist))
    return filelist
def band_pass_filter(original_signal, order, fc1,fc2, fs):
    b, a = signal.butter(N=order, Wn=[2*fc1/fs,2*fc2/fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal

def get_test(file_folder, class_name,step=1):
    
    
    drop_list=get_drop_list()
    all_files = get_file_list(file_folder,class_name,drop_list)
    index = 0
    count=0
    if class_name==0:
      pre='neg'
    else:
      pre='pos'
    for i,name in enumerate(all_files):
        if i%5!=0:
            continue

        dataset = np.zeros((1,299,39,1))
        sig, sr = librosa.load(name, sr=1000, offset=0.0, duration=None) # load heart sound data
        sig = band_pass_filter(sig, 5, 25, 400, 1000)
        dura=librosa.get_duration(sig, sr=1000)
        print(name,dura)
        
        for idx in range(0,len(sig)-3000,step*1000):
            mf=getMFCCMap(sig[idx:idx+3000])
            dataset = np.vstack((dataset,mf))  # concat the dataset
        dataset = np.delete(dataset, 0, 0)

        print(dataset.shape)   
        dataset=dataset.astype('float16')
        np.save('/content/drive/MyDrive/mfcc_test/{}_{}.npy'.format(pre,i),dataset)
              
        print(idx,class_name)
        
        # remove the first one of the dataset, due to initialization
    
    return dataset

def get_bi_spectrum(file_folder, class_name,step=1):
    dataset = np.zeros((1,299,52,1))
    if class_name==0:
      pre='neg'
    else:
      pre='pos'
    drop_list=get_drop_list()
    all_files = get_file_list(file_folder,class_name,drop_list)
    index = 0
    count=0
    for i,name in enumerate(all_files):
        if i%5==0:
            continue
        print(name)
        
        sig, sr = librosa.load(name, sr=1000, offset=0.0, duration=None) # load heart sound data
        sig = band_pass_filter(sig, 3, 25, 400, 1000)
        dura=librosa.get_duration(sig, sr=1000)
        print(dura)
        for idx in range(0,len(sig)-3000,step*1000):
            #librosa.display.waveplot(sig, sr=1000, x_axis='time', offset=0.0, ax=None)
            #plt.show()
            mf = getMFCCMap(sig[idx:idx+3000]) 
            dataset = np.vstack((dataset, mf))  # concat the dataset

            if index==1000:
              index=0
              dataset = np.delete(dataset, 0, 0)
              dataset=dataset.astype('float16')
              np.save('/content/{}_dataset{}.npy'.format(pre,count),dataset)
              dataset = np.zeros((1,299,52,1))
              count+=1
            index+=1
        
        print(idx,class_name,'num:',index)
        
        # remove the first one of the dataset, due to initialization
    dataset = np.delete(dataset, 0, 0)
    return dataset



if __name__ =="__main__":
    #file_folder = '../../project/databases/database_1000'
    #class_name = ['N','AS','MS','MR','MVP']
    pos_path='CinC2016/total/RECORDS-abnormal'
    neg_path='CinC2016/total/RECORDS-normal'
    pos_dataset = get_bi_spectrum(pos_path, 1,1)
    neg_dataset = get_bi_spectrum(neg_path, 0,2)
    #get_test(pos_path, 1,step=1)
    #get_test(neg_path, 0,step=2)