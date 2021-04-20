import librosa
from scipy import signal
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def band_pass_filter(original_signal, order, fc1,fc2, fs):
    b, a = signal.butter(N=order, Wn=[2*fc1/fs,2*fc2/fs], btype='bandpass')
    new_signal = signal.lfilter(b, a, original_signal)
    return new_signal


def plot(freq1, freq2,bi_spectrum,mf):
    df1 = freq1[1] - freq1[0]
    df2 = freq2[1] - freq2[0]
    freq1 = np.append(freq1, freq1[-1] + df1) - 0.5 * df1
    freq2 = np.append(freq2, freq2[-1] + df2) - 0.5 * df2
    fig, ax = plt.subplots()
    ax.pcolormesh(freq2, freq1, np.abs(bi_spectrum),cmap=plt.cm.jet)
    ax.axis('on')
    st.pyplot(fig)
            
    mfcc=pd.DataFrame(mf[:,:,:,0].reshape(299,39)[:,:6],columns=[i for i in range(6)])      
    st.line_chart(mfcc)