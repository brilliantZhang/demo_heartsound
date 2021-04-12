from extract_bispectrum import polycoherence
from extract_mfcc import getMFCCMap
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt

from models.pcrnn import build_model
import pandas as pd
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow.keras as keras

def get_feat(sig):
	dataset1 = np.zeros((1,299,150,1))
	dataset2 = np.zeros((1,299,39,1))
	#sig, sr = librosa.load(name, sr=1000, offset=0.0, duration=None) # load heart sound data
	dura=librosa.get_duration(sig, sr=1000)
	
	for idx in range(0,len(sig)-3000,1*3000):
				
		freq1, freq2, bi_spectrum = polycoherence(sig[idx:idx+3000],nfft=1195, fs = 1000, norm=None,noverlap = 100, nperseg=256)
		bi_spectrum = np.array(abs(bi_spectrum))
		bi_spectrum = 255 * (bi_spectrum - np.min(bi_spectrum)) / (np.max(bi_spectrum) - np.min(bi_spectrum))

		mf=getMFCCMap(sig[idx:idx+3000])
		dataset2 = np.vstack((dataset2,mf)) 

		#plot
		if idx==0:
			df1 = freq1[1] - freq1[0]
			df2 = freq2[1] - freq2[0]
			freq1 = np.append(freq1, freq1[-1] + df1) - 0.5 * df1
			freq2 = np.append(freq2, freq2[-1] + df2) - 0.5 * df2
			fig, ax = plt.subplots()
			ax.pcolormesh(freq2, freq1, np.abs(bi_spectrum),cmap=plt.cm.jet)
			ax.axis('on')
			st.pyplot(fig)

			mfcc=pd.DataFrame(mf.reshape(299,39)[:,:13],columns=[i for i in range(13)])
			
			st.line_chart(mfcc)

		bi_spectrum=[bi_spectrum[i][:i+1] for i in range(len(bi_spectrum)-1)]
		reduce=[]
		n=len(bi_spectrum)
		for i in range(n//2+1):
			if i==n-i-1:
				reduce.append(bi_spectrum[i])
				continue
			reduce.append(np.concatenate((bi_spectrum[i],bi_spectrum[n-i-1])))    
		bi_spectrum=np.array(reduce)

		dataset1 = np.vstack((dataset1, bi_spectrum.reshape(1,299, 150, 1))) 
			   
	dataset1 = np.delete(dataset1, 0, 0)           
	dataset1=dataset1.astype('float16')
	dataset2 = np.delete(dataset2, 0, 0)           
	
	return dataset1,dataset2

def cal_acc(label,prediction):
	num=0
	N = len(label)
	for i in range(len(label)):
		pred = np.argmax(prediction[i])
		if label[i]==pred:
			num+=1
	return num/N

def predict(bi,mfcc):
	#bi,mfcc=get_feat(sig)
	@st.cache(allow_output_mutation=True)
	def load_network(model,weights_path=''):
		model.load_weights('model_fold{}.h5'.format(0))
		return model
	model = build_model()
	model = load_network(model)
	y = model.predict([mfcc,bi])
	if sum(np.argmax(y,axis=1))/len(y)>=0.5:
		return True
	else:
		return False

