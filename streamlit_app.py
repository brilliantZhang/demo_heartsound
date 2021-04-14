# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

import librosa
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
from streamlit.server.server import Server

from extract_bispectrum import polycoherence
from test import get_feat,predict
from utils import band_pass_filter

@st.cache(show_spinner=False)
def load_local_audio(uploaded_file):
    bytes_data = uploaded_file.getvalue()  
    image = BytesIO(bytes_data)
    return image


def run_the_app():
    st.sidebar.markdown("### 第一步：选择样例文件或上传本地的心音音频文件(wav)")
    
    choice = st.sidebar.selectbox('选择使用服务端样例音频文件或上传本地文件', ["样例1","样例2","本地上传"])
    if choice=="本地上传":
        file = st.sidebar.file_uploader(" ")
    elif choice=="样例1":
        st.audio('a0009.wav', format='audio/wav')
        audio_bytes = 'a0009.wav'
    else:
        st.audio('a0010.wav', format='audio/wav')
        audio_bytes = 'a0010.wav'

    st.sidebar.markdown("### 第二步：点击诊断查看结果")
    left_column,middle_column, right_column = st.sidebar.beta_columns(3)
    
    if middle_column.button('诊断'):
        if choice=="本地上传":
            audio_bytes = load_local_audio(file)
            st.audio(audio_bytes, format='audio/wav')
        st.markdown('---')
        st.markdown('## 音频数据可视化:')
        sig, sr = librosa.load(audio_bytes, sr=1000, offset=0.0, duration=None)

        st.markdown('---')
        st.markdown('### 原始音频数据波形图:')
        fig, ax = plt.subplots()
        ax.plot(sig)
        st.pyplot(fig)

        st.markdown('---')
        st.markdown('### 滤波后波形图:')
        sig = band_pass_filter(sig, 5, 25, 400, 1000)
        fig, ax = plt.subplots()
        ax.plot(sig)
        st.pyplot(fig)

        st.markdown('---')
        st.markdown('### 音频二阶谱特征以及MFCC特征图:')
        bi,mfcc=get_feat(sig)

        st.markdown('---')
        st.markdown('## 诊断结果:')
        if predict(bi,mfcc):
            st.success('心音正常')
        else:
            st.error('心音异常')
        
@st.cache(show_spinner=False)
def read_markdown(path):
    with open(path, "r",encoding = 'utf-8') as f:  # 打开文件
        data = f.read()  # 读取文件
    return data


# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # 1 初始化界面
    # Render the readme as markdown using st.markdown.
    image = Image.open('./images/icon.jpg')
    st.image(image,use_column_width=False)
    readme_text = st.markdown(read_markdown("instructions.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    #st.sidebar.title("图像检测参数调节器")   # 侧边栏
    app_mode = st.sidebar.selectbox("切换页面模式:",
        ["Run the app","Show instructions", "Show the source code"])
    
    # 展示栏目三
    if app_mode == "Run the app": 
        run_the_app() # 运行内容

    # 展示栏目一
    elif app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    # 展示栏目二
    elif app_mode == "Show the source code":
        readme_text.empty()     # 刷新页面
        st.code(read_markdown("streamlit_app.py"))

def login():
    image = Image.open('./images/icon.jpg')
    st.image(image,use_column_width=False)

    name = st.text_input('用户名', '')
    pawd = st.text_input('密码', '')
    
    if st.button('登录'):
        if name=='admin' and pawd=='admin':
            st.success('登录成功!')
            return True
        else:
            st.error('用户名或密码错误')
            return False 

if __name__ == "__main__":
    main()  
