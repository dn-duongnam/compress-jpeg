import streamlit as st
import os
import rawpy
import numpy as np
from io import BytesIO
import plotly.express as px
from scipy.fftpack import dct,idct
import plotly.figure_factory as ff
import plotly.graph_objects as go
import cv2
from scipy.ndimage import gaussian_filter
from skimage import io, metrics
import pandas as pd

def add_noise_periodic(img, a, b):
    x = np.linspace(0, 1,512) 
    noise = np.array([a*np.cos(b*np.pi*x) * 255])
    noisy_image = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_noise_periodic2d(img, a, b):
    x = np.linspace(0, 1,512) 
    y = np.linspace(0, 1,512) 
    X,Y = np.meshgrid(x, y)
    noise = np.array([a*np.sin(b*np.pi*(X + Y)) * 255])
    noisy_image = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_image[0]


def denoising_periodic(args):
    st.title('Denoising Periodic Display App')
    # upload and read image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "arw", "cr2", "png"])


    a = st.text_input('Input a: ', '0.25')
    a = float(a.strip())
    b = st.text_input("Input b: ", "30")
    b = float(b.strip())
    st.write("Type noise 1D: ")
    st.latex(r'''
             a \times cos(b \times x)
             ''')
    st.write("Type noise 2D: ")
    st.latex(r'''
             a \times sin(b \times (x + y))
             ''')
    typeNoisy = st.selectbox("Type noise Periodic <1D - 2D>: ", ["1D", "2D"])
    kernel_size = st.text_input("Kernel Size: ", "5")
    kernel_size = int(kernel_size.strip())
    alpha = st.text_input("Alpha: ", "21.5")
    alpha = float(alpha.strip())
    if uploaded_file is None: # catch error
        return ""
    file_bytes = uploaded_file.getvalue()
    nparr = np.frombuffer(file_bytes, np.uint8)

    if uploaded_file.type in ['image/jpeg', 'image/jpg']:
        img_raw = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif uploaded_file.type == 'image/png':
        img_raw = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        with rawpy.imread(uploaded_file) as raw:
            img_raw = raw.postprocess()
    
    height, width, _ = img_raw.shape
    if(height > 512 and width > 512):
        start_y = (height - 512) // 2
        start_x = (width - 512) // 2
        new_height, new_width = 512, 512
        image = img_raw[start_y:start_y + new_height, start_x:start_x + new_width, :]
    else:
        image = img_raw
        
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if typeNoisy == '1D':
        noisy_image = add_noise_periodic(image,a,b)
    else:
        noisy_image = add_noise_periodic2d(image,a,b)
    
    w = image.shape[1] #modify it if you want, maximal 8 due to default quantization table is 8*8
    h = w
    xLen = image.shape[1]//w
    yLen = image.shape[0]//h
    
    dct_blocks = dct(dct(noisy_image.T, norm='ortho').T, norm='ortho')

    J = dct_blocks
    AJ = np.abs(dct_blocks)
    
    # fig_1 = px.imshow(J)
    # st.subheader('DCT Block')
    # st.plotly_chart(fig_1)

    median_dct = cv2.medianBlur(AJ.astype(np.float32), kernel_size)
    
    mask_noisy = AJ > alpha * median_dct
    # fig_1 = px.imshow(mask_noisy)
    # st.subheader('Mask Noisy')
    # st.plotly_chart(fig_1)
    # st.write(mask_noisy)
    
    median_dct_noabs = cv2.medianBlur(J.astype(np.float32), kernel_size)
    dct_blocks_filtered = dct_blocks.copy()
    apply_mask = np.where(mask_noisy == True, median_dct_noabs, J)
    # fig_1 = px.imshow(median_dct_noabs)
    # st.subheader('Median DCT')
    # st.plotly_chart(fig_1)
    # st.write(median_dct_noabs)
    
    
    dct_blocks_filtered = apply_mask
    # fig_1 = px.imshow(dct_blocks_filtered)
    # st.subheader('DCT')
    # st.plotly_chart(fig_1)
    # st.write(dct_blocks_filtered)
    filtered_blocks = idct(idct(dct_blocks_filtered.T, norm='ortho').T, norm='ortho')
    norm_image = np.clip(filtered_blocks, 0, 255).astype(np.uint8)
    
    fig_1 = px.imshow(noisy_image, binary_string=True)
    fig_1.update_layout(title='Image Noisy')

    
    fig_2 = px.imshow(norm_image, binary_string=True)
    fig_2.update_layout(title='Image Denoising')
    
    st.write("### Compare")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_1, use_container_width=True)
    col2.plotly_chart(fig_2, use_container_width=True)
    
    
    psnr_value_noise = metrics.peak_signal_noise_ratio(noisy_image, image)
    mse_value_noise = metrics.mean_squared_error(noisy_image, image)
    ssim_value_noise = metrics.structural_similarity(noisy_image, image, win_size=3)
        
    # st.write(f"### Image Quality Metrics  (Size Block={size_block}, Noise={noise}, Sigma={sigma}, Threshold={threshold}, Stride={stride})")
    # table_data = {"PSNR": [psnr_value], "MSE": [mse_value], "SSIM": [ssim_value]}
    # table = st.table(table_data)
    
    psnr_value = metrics.peak_signal_noise_ratio(image, norm_image)
    mse_value = metrics.mean_squared_error(image, norm_image)
    ssim_value = metrics.structural_similarity(image, norm_image, win_size=3)
        
    st.write(f"### Image Quality Metrics")
    table_data = {"Type Image":["Denoising Image", "Noisy Image"],"PSNR": [psnr_value, psnr_value_noise], "MSE": [mse_value, mse_value_noise], "SSIM": [ssim_value,ssim_value_noise]}
    table = st.table(table_data)
    
    
    