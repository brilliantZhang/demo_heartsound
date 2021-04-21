from extract_bispectrum import polycoherence
from extract_mfcc import getMFCCMap
import numpy as np
import librosa
import streamlit as st

from models.pcrnn import build_model

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow.python.platform import gfile
from utils import plot

def get_feat(raw,sig):
    dataset1 = np.zeros((1,256,256,1))
    dataset2 = np.zeros((1,299,39,3))
    dura=librosa.get_duration(sig, sr=1000)
    
    for idx in range(0,len(sig)-3000,1*3000):
                
        freq1, freq2, bi_spectrum = polycoherence(raw[idx:idx+3000],nfft=1024, fs = 1000, norm=None,noverlap = 100, nperseg=256)
        bi_spectrum = np.array(abs(bi_spectrum))
        bi_spectrum = 255 * (bi_spectrum - np.min(bi_spectrum)) / (np.max(bi_spectrum) - np.min(bi_spectrum))
        dataset1 = np.vstack((dataset1, bi_spectrum.reshape(1,256, 256, 1))) 
        
        mf=getMFCCMap(sig[idx:idx+3000])
        dataset2 = np.vstack((dataset2,mf)) 

        if idx==0:
            plot(freq1, freq2,bi_spectrum,mf)
               
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
    @st.cache(allow_output_mutation=True)
    def load_network():
        sess = tf.Session()
        with gfile.FastGFile('./pb_models/model_combine.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
    
        sess.run(tf.global_variables_initializer())
         
        # 输出
        ops = [sess.graph.get_tensor_by_name('softmax_{}/Softmax:0'.format(i+1)) for i in range(10)]
        return sess,ops

    sess,ops = load_network()
    feed_dict={}
    for i in range(3,23,2):#3，23
        feed_dict[sess.graph.get_tensor_by_name('input_{}:0'.format(i))]=mfcc
        feed_dict[sess.graph.get_tensor_by_name('input_{}:0'.format(i+1))]=bi

    res = sess.run(ops, feed_dict=feed_dict)
    st.write(tmp)
    y = [np.argmax(r) for r in res]
    st.write(y)
    tmp = [[np.argmax(r),max(r)] for r in res]
    st.write(tmp)
    for i in range(len(tmp)):
        if tmp[i][0]==1:
            st.write("模型{}预测结果：异常，置信度：{}".format(i+1,tmp[i][1]))
        else:
            st.write("模型{}预测结果：正常，置信度：{}".format(i+1,tmp[i][1]))
    if sum(y)/len(y)>=0.5:
        return True
    else:
        return False

