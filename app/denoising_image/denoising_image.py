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


DCTbasis3x3 = np.array([
    [   0.5773502588272094726562500000000000000000,
        0.5773502588272094726562500000000000000000,
        0.5773502588272094726562500000000000000000,     ],
    [  0.7071067690849304199218750000000000000000,
      0.0000000000000000000000000000000000000000,
      -0.7071067690849304199218750000000000000000, ],
    [
        0.4082483053207397460937500000000000000000,
        -0.8164966106414794921875000000000000000000,
        0.4082483053207397460937500000000000000000      ]
], dtype=np.float32)

def ColorTransform(img, DCTbasis3x3, flag=1):
    img_cp = img.copy().astype(np.float32)
    image_flat = img_cp.reshape(-1, 3)
    if flag == 1:
        transformed_image_flat = np.dot(image_flat, DCTbasis3x3.T)
    else:
        transformed_image_flat = np.dot(image_flat, np.linalg.inv(DCTbasis3x3.T))
    transformed_image_flat = transformed_image_flat.reshape(img.shape)
    return transformed_image_flat

def image2Patches(img, size_patch, stride):
    h, w = img.shape[0:2]
    w_h, w_w = size_patch,size_patch
    s_h, s_w = stride, stride
    starting_points = [(x, y)  for x in set( list(range(0, h - w_h, s_h)) + [h - w_h] ) 
                                for y in set( list(range(0, w - w_w, s_w)) + [w - w_w] )]
    patches = np.empty((2, w_h, w_w, 3), dtype='float64')
    for i, (x, y) in enumerate(starting_points[:2]):
        patches[i] = img[x:x + w_h, y:y + w_w, :]
    return patches

def sliding_window_denoising(img, size_patch, stride, threshold):
    h, w = img.shape[0:2]
    w_h, w_w = size_patch,size_patch
    s_h, s_w = stride, stride
    result = np.zeros((*img.shape[:-1], 3), dtype='float64')
    overlap = np.zeros((*img.shape[:-1], 3), dtype='float64')
    starting_points = [(x, y)  for x in set( list(range(0, h - w_h, s_h)) + [h - w_h] ) 
                                for y in set( list(range(0, w - w_w, s_w)) + [w - w_w] )]
    patches = np.empty((len(starting_points), w_h, w_w, 3), dtype='float64')
    for i, (x, y) in enumerate(starting_points):
        patches[i] = img[x:x + w_h, y:y + w_w, :]

    patches_dct = dct(dct(patches, axis=1, norm='ortho'), axis=2, norm='ortho')
    patches_th = np.where(np.abs(patches_dct) < threshold, 0, patches_dct)
    patches_th = idct(idct(patches_th, axis=1, norm='ortho'), axis=2, norm='ortho')
        
    for i in range(len(patches)):
        x, y = starting_points[i]
        result[x:x + w_h, y:y + w_w, :] += patches_th[i]
        overlap[x:x + w_h, y:y + w_w, :] += 1

    assert np.sum(overlap == 0.) == 0, "Sliding window does not cover all volume"

    return np.clip((result / overlap), 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    
    salt_mask = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_mask] = 255
    
    pepper_mask = np.random.rand(*image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0
    
    return noisy_image

def add_poisson_noise(image, scale_factor):
    noisy_image = np.random.poisson(image * scale_factor) / scale_factor
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def addSpeckleNoise(img, intensity=0.5):
    image = img.copy()
    # Generate speckle noise
    row, col, c = image.shape
    speckle = intensity * np.random.randn(row, col, c)
    
    # Add the noise to the image
    noisy = image + image * speckle
    
    # Clip the values to be in the valid range [0, 255]
    noisy = np.clip(noisy, 0, 255)
    
    # Convert back to uint8 (if needed)
    noisy = noisy.astype(np.uint8)
    
    return noisy

def denoising_image(args):
    st.title('Image Color Display App')
    # upload and read image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "arw", "cr2", "png", "bmp"])
    size_block = st.selectbox("Size block:", [8, 16, 32, 64])
    
    noise = st.selectbox("Noise:", ["Gaussian", "Salt and Pepper", "Poisson","Speckle" , "None"])
    sigma = st.text_input('Sigma: ', '10')
    sigma = float(sigma.strip())
    threshold = st.text_input("Threshold: ", "30")
    threshold = float(threshold.strip())
    stride = st.text_input("Stride: ", "1")
    stride = int(stride.strip())
    if uploaded_file is None: # catch error
        return ""
    file_bytes = uploaded_file.getvalue()
    nparr = np.frombuffer(file_bytes, np.uint8)

    if uploaded_file.type in ['image/jpeg', 'image/jpg', 'image/bmp']:
        img_raw = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif uploaded_file.type == 'image/png':
        img_raw = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        with rawpy.imread(uploaded_file) as raw:
            img_raw = raw.postprocess()
    
    height, width, _ = img_raw.shape
    if(height > 1024 and width > 1024):
        start_y = (height - 1024) // 2
        start_x = (width - 1024) // 2
        new_height, new_width = 1024, 1024
        img = img_raw[start_y:start_y + new_height, start_x:start_x + new_width, :]
    else:
        img = img_raw
        
    if noise == "Gaussian":
        rows, cols, channels = img.shape
        mean = 0
        noise_val = np.random.normal(mean, sigma, (rows, cols, channels))
        noisy_image = np.clip(img + noise_val, 0, 255).astype(np.uint8)
    elif noise == "Salt and Pepper":
        salt_prob, pepper_prob = sigma / 1000, sigma / 1000
        noisy_image = add_salt_and_pepper_noise(img, salt_prob, pepper_prob)
    elif noise == "Poisson":
        noisy_image = add_poisson_noise(img, sigma)
    
    elif noise == "Speckle":
        noisy_image = addSpeckleNoise(img)
    else:
        noisy_image = img

    if noise == "None":
        fig_original = px.imshow(noisy_image)
        st.subheader('Original Noise Image')
        st.plotly_chart(fig_original)
        
        image_patch = image2Patches(noisy_image, size_block, stride)
        patches_dct_ori = dct(dct(image_patch, axis=1, norm='ortho'), axis=2, norm='ortho')
        fig_original = px.imshow(patches_dct_ori[0,:, :, 0])
        fig_original.update_traces(text=np.round(patches_dct_ori[0,:, :, 0], 2),
                    hoverinfo='text',
                    texttemplate="%{text}",
                    textfont={"size": 8})
        st.subheader('Original Noise Image DCT')
        st.plotly_chart(fig_original)
        
        patches_th = np.where(np.abs(patches_dct_ori) < threshold, 0, patches_dct_ori)
        fig_original = px.imshow(patches_th[0,:, :, 0])
        fig_original.update_traces(text=np.round(patches_th[0,:, :, 0], 2),
                    hoverinfo='text',
                    texttemplate="%{text}",
                    textfont={"size": 8})
        st.subheader('Original Noise Image DCT After Threshold')
        st.plotly_chart(fig_original)
    else:      
        fig_1 = px.imshow(noisy_image)
        fig_1.update_layout(title=f'Image Noisy {noise}')
        
        fig_2 = px.imshow(img)
        fig_2.update_layout(title='Image Original')
        
        st.write("### Compare")
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_2, use_container_width=True)
        col2.plotly_chart(fig_1, use_container_width=True)
        
        image_patch = image2Patches(img, size_block, stride)
        patches_dct_ori = dct(dct(image_patch, axis=1, norm='ortho'), axis=2, norm='ortho')
        
        noise_patch = image2Patches(noisy_image, size_block, stride)
        patches_dct_noise = dct(dct(noise_patch, axis=1, norm='ortho'), axis=2, norm='ortho')
    
        fig_1 = px.imshow(patches_dct_ori[0,:, :, 0])
        fig_1.update_layout(title="Original Image DCT")
        fig_1.update_traces(text=np.round(patches_dct_ori[0,:, :, 0], 2),
                    hoverinfo='text',
                    texttemplate="%{text}",
                    textfont={"size": 8})
        
        fig_2 = px.imshow(patches_dct_noise[0,:, :, 0])
        fig_2.update_layout(title="Noisy Image DCT")
        fig_2.update_traces(text=np.round(patches_dct_noise[0,:, :, 0], 2),
                    hoverinfo='text',
                    texttemplate="%{text}",
                    textfont={"size": 8})
        
        st.write("### Compare DCT")
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_1, use_container_width=True)
        col2.plotly_chart(fig_2, use_container_width=True)
        
        patches_noise_th = np.where(np.abs(patches_dct_ori) < threshold, 0, patches_dct_ori)
        fig_1 = px.imshow(patches_dct_ori[0,:, :, 0])
        fig_1.update_layout(title="Original Image DCT")
        fig_1.update_traces(text=np.round(patches_dct_ori[0,:, :, 0], 2),
                    hoverinfo='text',
                    texttemplate="%{text}",
                    textfont={"size": 8})
        
        fig_2 = px.imshow(patches_noise_th[0,:, :, 0])
        fig_2.update_layout(title="Noisy Image DCT After Threshold")
        fig_2.update_traces(text=np.round(patches_noise_th[0,:, :, 0], 2),
                    hoverinfo='text',
                    texttemplate="%{text}",
                    textfont={"size": 8})
        st.write("### Compare DCT")
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_1, use_container_width=True)
        col2.plotly_chart(fig_2, use_container_width=True)
        
    # image_trans = ColorTransform(noisy_image, DCTbasis3x3, flag=1)
    image_trans = noisy_image
    newImg = sliding_window_denoising(image_trans, size_block, stride, threshold)
    # newImg = ColorTransform(newImg, DCTbasis3x3, flag=-1)
    newImg = np.clip(newImg, 0, 255).astype(np.uint8)
    
    if noise == "None":
        hist_ori = cv2.calcHist([noisy_image], [0], None, [256], [0, 256])
        df_ori = pd.DataFrame(hist_ori, columns=['Frequency'])
        hist_denoise = cv2.calcHist([newImg], [0], None, [256], [0, 256])
        df_denoise = pd.DataFrame(hist_denoise, columns=['Frequency'])
        fig_1 = px.bar(df_ori, x=df_ori.index, y='Frequency', labels={'x':'Pixel value', 'y':'Frequency'})
        fig_1.update_layout(title='Histogram Image Original')
        fig_2 = px.bar(df_denoise, x=df_denoise.index, y='Frequency', labels={'x':'Pixel value', 'y':'Frequency'})
        fig_2.update_layout(title='Histogram Image Denoising')
        st.write("### Compare Histogram")
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_1, use_container_width=True)
        col2.plotly_chart(fig_2, use_container_width=True)
    else:
        hist_ori = cv2.calcHist([img], [0], None, [256], [0, 256])
        df_ori = pd.DataFrame(hist_ori, columns=['Frequency'])
        hist_noise = cv2.calcHist([noisy_image], [0], None, [256], [0, 256])
        df_noise = pd.DataFrame(hist_noise, columns=['Frequency'])
        hist_denoise = cv2.calcHist([newImg], [0], None, [256], [0, 256])
        df_denoise = pd.DataFrame(hist_denoise, columns=['Frequency'])
        fig_1 = px.bar(df_ori, x=df_ori.index, y='Frequency', labels={'x':'Pixel value', 'y':'Frequency'})
        fig_1.update_layout(title='Histogram Image Original')
        
        fig_2 = px.bar(df_noise, x=df_noise.index, y='Frequency', labels={'x':'Pixel value', 'y':'Frequency'})
        fig_2.update_layout(title=f'Histogram Image Noisy {noise}')
        
        fig_3 = px.bar(df_denoise, x=df_denoise.index, y='Frequency', labels={'x':'Pixel value', 'y':'Frequency'})
        fig_3.update_layout(title='Histogram Image Denoising')
        st.write("### Compare Histogram")
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_1, use_container_width=True)
        col2.plotly_chart(fig_2, use_container_width=True)
        st.write("### Compare Histogram")
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_1, use_container_width=True)
        col2.plotly_chart(fig_3, use_container_width=True)
    
    fig_1 = px.imshow(noisy_image)
    fig_1.update_layout(title=f'Image Noisy {noise}')
    fig_2 = px.imshow(newImg)
    fig_2.update_layout(title='Image Denoising')
    
    st.write("### Compare")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_1, use_container_width=True)
    col2.plotly_chart(fig_2, use_container_width=True)
    
    fig_1 = px.imshow(img)
    fig_1.update_layout(title=f'Original Image')
    fig_2 = px.imshow(newImg)
    fig_2.update_layout(title='Image Denoising')
    
    st.write("### Compare Original")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_1, use_container_width=True)
    col2.plotly_chart(fig_2, use_container_width=True)
    
    if noise == "None":
        psnr_value = metrics.peak_signal_noise_ratio(noisy_image, newImg)
        mse_value = metrics.mean_squared_error(noisy_image, newImg)
        ssim_value = metrics.structural_similarity(noisy_image, newImg, win_size=3)
            
        st.write(f"### Image Quality Metrics  (Size Block={size_block}, Noise={noise}, Sigma={sigma}, Threshold={threshold}, Stride={stride})")
        table_data = {"Type Image":["Denoising Image"],"PSNR": [psnr_value], "MSE": [mse_value], "SSIM": [ssim_value]}
        table = st.table(table_data)
    
    else:
    
        psnr_value_noise = metrics.peak_signal_noise_ratio(img, noisy_image)
        mse_value_noise = metrics.mean_squared_error(img, noisy_image)
        ssim_value_noise = metrics.structural_similarity(img, noisy_image, win_size=3)
            
        # st.write(f"### Image Quality Metrics  (Size Block={size_block}, Noise={noise}, Sigma={sigma}, Threshold={threshold}, Stride={stride})")
        # table_data = {"PSNR": [psnr_value], "MSE": [mse_value], "SSIM": [ssim_value]}
        # table = st.table(table_data)
        
        psnr_value = metrics.peak_signal_noise_ratio(img, newImg)
        mse_value = metrics.mean_squared_error(img, newImg)
        ssim_value = metrics.structural_similarity(img, newImg, win_size=3)
            
        st.write(f"### Image Quality Metrics  (Size Block={size_block}, Noise={noise}, Sigma={sigma}, Threshold={threshold}, Stride={stride})")
        table_data = {"Type Image":["Denoising Image", "Noisy Image"], "PSNR": [psnr_value, psnr_value_noise], "MSE": [mse_value, mse_value_noise], "SSIM": [ssim_value,ssim_value_noise]}
        table = st.table(table_data)
    

    
    
    