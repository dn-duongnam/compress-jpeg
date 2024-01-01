import streamlit as st
import os
import rawpy
import numpy as np
from skimage import io
from io import BytesIO
import plotly.express as px
from skimage.color import rgb2ycbcr,ycbcr2rgb
from scipy.fftpack import dct,idct
from copy import copy
import plotly.figure_factory as ff
from scipy.fftpack import dct,idct
import plotly.graph_objects as go
import huffman
from collections import Counter
import time
import cv2

def toBlocks(img, xLen, yLen, h, w):
    blocks = np.zeros((yLen,xLen,h,w),dtype=np.int16)
    for y in range(yLen):
        for x in range(xLen):
            blocks[y][x]=img[y*h:(y+1)*h,x*w:(x+1)*w]
    return np.array(blocks)

def dctOrDedctAllBlocks(blocks, yLen, xLen, h, w,type="dct"):
    f=dct if type=="dct" else idct
    dedctBlocks = np.zeros((yLen,xLen,h,w))
    for y in range(yLen):
        for x in range(xLen):
            d      = np.zeros((h,w))
            block  = blocks[y][x][:,:]
            d[:,:] = f(f(block.T, norm = 'ortho').T, norm = 'ortho')
            if (type!="dct"):
                d=d.round().astype(np.int16)
            dedctBlocks[y][x]=d
    return dedctBlocks

def blocks2img(blocks, xLen, yLen, h, w):
    W   = xLen*w
    H   = yLen*h
    img = np.zeros((H,W))
    for y in range(yLen):
        for x in range(xLen):
            img[y*h:y*h+h,x*w:x*w+w]=blocks[y][x]
    return img

def zigZag(block, h, w):
    lines=[[] for i in range(h+w-1)]
    for y in range(h):
        for x in range(w):
            i=y+x
            if(i%2 ==0):
                lines[i].insert(0,block[y][x])
            else:
                lines[i].append(block[y][x])
    return np.array([coefficient for line in lines for coefficient in line])

def encodeRLE(x):
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int), 
                np.array([], dtype=int), 
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    
    return "".join([str(values[i]) + str(lengths[i]) for i in range(len(lengths))])

def runLength(zigZagArr, lastDC, bitBits, runBits, rbBits, hfm=None):
    run=0
    newDC=min(zigZagArr[0],2**(2**bitBits-1))
    DC=newDC-lastDC
    bitSize=max(0,min(int(np.ceil(np.log(abs(DC)+0.000000001)/np.log(2))),2**bitBits-1))
    code=format(bitSize, '0'+str(bitBits)+'b')
    if (bitSize>0):
        code+=(format(DC,"b") if DC>0 else ''.join([str((int(b)^1)) for b in format(abs(DC),"b")]))
    for AC in zigZagArr[1:]:
        if(AC!=0):
            AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
            if(run>2**runBits-1):
                runGap=2**runBits
                k=run//runGap
                for i in range(k):
                    code+=('1'*runBits+'0'*bitBits)if hfm == None else  hfm['1'*runBits+'0'*bitBits]#end
                run-=k*runGap
            run=min(run,2**runBits-1)
            bitSize=min(int(np.ceil(np.log(abs(AC)+0.000000001)/np.log(2))),2**bitBits-1)
            rb=format(run<<bitBits|bitSize,'0'+str(rbBits)+'b') if hfm == None else hfm[format(run<<bitBits|bitSize,'0'+str(rbBits)+'b')]
            code+=rb+(format(AC,"b") if AC>=0 else ''.join([str((int(b)^1)) for b in format(abs(AC),"b")]))
            run=0
        else:
            run+=1
    code+="0"*(rbBits) if hfm == None else  hfm["0"*(rbBits)]#end
    return code,newDC

def huffmanCounter(zigZagArr, bitBits, runBits, rbBits):
    rbCount = []
    run = 0
    for AC in zigZagArr[1:]:
        if(AC != 0):
            AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
            if (run>2**runBits-1):
                runGap=2**runBits
                k=run//runGap
                for i in range(k):
                    rbCount.append('1'*runBits+'0'*bitBits)
                run-=k*runGap
            run=min(run,2**runBits-1)
            bitSize=min(int(np.ceil(np.log(abs(AC)+0.000000001)/np.log(2))),2**bitBits-1)
            rbCount.append(format(run<<bitBits|bitSize,'0'+str(rbBits)+'b'))
            run=0
        else:
            run+=1
    rbCount.append("0"*(rbBits))
    return Counter(rbCount)

def runLength2bytes(code):
    return bytes([len(code)%8]+[int(code[i:i+8],2) for i in range(0, len(code), 8)])

def huffmanCounterWholeImg(blocks, xLen, yLen, h, w, bitBits, runBits, rbBits):
    rbCount=np.zeros(xLen*yLen,dtype=Counter)
    zz=np.zeros(xLen*yLen,dtype=object)
    for y in range(yLen):
        for x in range(xLen):
            zz[y*xLen+x]=zigZag(blocks[y, x,:,:], h, w)
            rbCount[y*xLen+x]=huffmanCounter(zz[y*xLen+x], bitBits, runBits, rbBits)
    return np.sum(rbCount),zz

def bytes2runLength(bytes):
    return "".join([format(i,'08b') for i in list(bytes)][1:-1 if list(bytes)[-1]!=0 else None])+(format(list(bytes)[-1],'0'+str(list(bytes)[0])+'b')if list(bytes)[-1]!=0 else"")

def savingQuantizedDctBlocks(blocks, xLen, yLen, h, w, img, useHuffman, bitBits, runBits, rbBits):
    rbCount,zigZag=huffmanCounterWholeImg(blocks, xLen, yLen, h, w, bitBits, runBits, rbBits)
    hfm=huffman.codebook(rbCount.items())
    sortedHfm=[[hfm[i[0]],i[0]] for i in rbCount.most_common()]
    code=""
    DC=0
    for y in range(yLen):
        for x in range(xLen):
            codeNew,DC=runLength(zigZag[y*xLen+x],DC, bitBits, runBits, rbBits, hfm if useHuffman else None)
            code+=codeNew
    savedImg=runLength2bytes(code)
    st.write("Image original size:    %.3f MB"%(img.size/(2**20)))
    st.write("Compression image size: %.3f MB"%(len(savedImg)/2**20))
    st.write("Compression ratio:      %.2f : 1"%(img.size/2**20/(len(savedImg)/2**20)))
    return bytes([int(format(xLen,'012b')[:8],2),int(format(xLen,'012b')[8:]+format(yLen,'012b')[:4],2),int(format(yLen,'012b')[4:],2)])+savedImg,sortedHfm

def deZigZag(zigZagArr, locations, h, w):
    zigZagArr=[zigZagArr[l[0]:l[1]] for l in locations]
    block=np.zeros((h,w),dtype=np.int16)
    for y in range(h):
        for x in range(w):
            i=y+x
            if(i%2 != 0):
                block[y][x]=zigZagArr[i][0]
                zigZagArr[i]=zigZagArr[i][1:]
            else:
                block[y][x]=zigZagArr[i][-1:][0]
                zigZagArr[i]=zigZagArr[i][:-1]
    return block

def deZigZag2(zigZagArr, h, w, zzLine):
    block=np.zeros((h,w),dtype=np.int16)
    for i in range(len(zzLine)):
        block[zzLine[i][1],zzLine[i][0]]=zigZagArr[i]
    return block

def loadingQuantizedDctBlocks(loadedbytes, runBits, bitBits, rbBits, h, w, zzLine, sortedHfm=None):
    runMax = 2**runBits - 1
    xLen = int(format(loadedbytes[0], 'b') + format(loadedbytes[1], '08b')[:4], 2)
    yLen = int(format(loadedbytes[1], '08b')[4:] + format(loadedbytes[2], '08b'), 2)
    code = bytes2runLength(loadedbytes[3:])
    blocks = np.zeros((yLen, xLen, h, w), dtype=np.int16)
    lastDC = 0
    rbBitsTmp = rbBits
    rbTmp = ""
    cursor = 0  # don't use code=code[index:] to remove readed strings when len(String) is large like 1,000,000. It will be extremely slow
    for y in range(yLen):
        for x in range(xLen):
            zz = np.zeros(64)
            bitSize = int(code[cursor:cursor+bitBits], 2)
            DC = code[cursor+bitBits:cursor+bitBits+bitSize]
            DC = (int(DC, 2) if DC[0] == "1" else -int(''.join([str((int(b)^1)) for b in DC]), 2)) if bitSize > 0 else 0
            cursor += (bitBits+bitSize)
            zz[0] = DC+lastDC
            lastDC = zz[0]
            r = 1
            while True:
                if sortedHfm is not None:
                    for ii in sortedHfm:
                        if ii[0] == code[cursor:cursor+len(ii[0])]:
                            rbTmp = ii[1]
                            rbBitsTmp = len(ii[0])
                            break
                    run=int(rbTmp[:runBits],2)
                    bitSize=int(rbTmp[runBits:],2)
                else:
                    run = int(code[cursor:cursor+runBits], 2)
                    bitSize = int(code[cursor+runBits:cursor+rbBitsTmp], 2)
                if bitSize == 0:
                    cursor += rbBitsTmp
                    if run == runMax:
                        r += (run+1)
                        continue
                    else:
                        break
                coefficient = code[cursor+rbBitsTmp:cursor+rbBitsTmp+bitSize]
                if coefficient[0] == "0":
                    coefficient = -int(''.join([str((int(b)^1)) for b in coefficient]), 2)
                else:
                    coefficient = int(coefficient, 2)
                if (r + run >= 64): break
                zz[r+run] = coefficient
                r += (run+1)
                cursor += rbBitsTmp+bitSize
            blocks[y, x, ...] = deZigZag2(zz, h, w, zzLine)
    return blocks

def gray_image_jpeg(args):
    st.title('Image Gray Display App')

    # upload and read image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "arw", "cr2", "png"])
    quality = st.text_input('Quality Compressed: ', '90')
    quality = int(quality.strip())
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
    
    # Crop image if too big
    height, width, _ = img_raw.shape
    if(height > 1024 and width > 1024):
        start_y = (height - 1024) // 2
        start_x = (width - 1024) // 2
        new_height, new_width = 1024, 1024
        img = img_raw[start_y:start_y + new_height, start_x:start_x + new_width, :]
    else:
        img = img_raw
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    originalImg = img.copy()
    
    # Config
    w                 = 8 # modify it if you want, maximal 8 due to default quantization table is 8*8
    w                 = max(2,min(8,w))
    h                 = w
    xLen              = img.shape[1]//w
    yLen              = img.shape[0]//h
    runBits           = 1 # modify it if you want
    bitBits           = 3  # modify it if you want
    rbBits            = runBits+bitBits ##(run,bitSize of coefficient)
    useHuffman        = True #modify it if you want
    quantizationRatio = 1 # modify it if you want, quantization table=default quantization table * quantizationRatio
    
    
    originalImg = copy(img) # copy image
        
    #Plot ảnh Original
    fig_original = px.imshow(originalImg, binary_string=True)
    fig_original.update_layout(title='Original Image')
    st.subheader('Original Image')
    st.plotly_chart(fig_original)
    
    #  Hiển thị vị trí 0,0 của kênh Y, CB (khối gốc)
    blocks = toBlocks(img, xLen, yLen, h, w)
    selected_block = blocks[0][0]
    Y_channel = selected_block[:, :]
    fig_Y = px.imshow(Y_channel, color_continuous_scale='gray')
    fig_Y.update_layout(title='Block 0 - 0')
    st.subheader('Block 0 - 0 in Blocks Image')
    st.plotly_chart(fig_Y)
    
    # DCT : Discrete Cosine Transform
    dctBlocks=dctOrDedctAllBlocks(blocks, yLen, xLen, h, w, "dct")
    # Hiển thị 3 kênh 2d block(0,0) sau khi dct
    block_Y_dct = dctBlocks[0][0][:, :]
    fig_Y_dct = px.imshow(block_Y_dct, color_continuous_scale='gray')
    fig_Y_dct.update_layout(title='Block 0 - 0 after DCT')
    st.subheader('Block 0 - 0 in Blocks Image after DCT')
    st.plotly_chart(fig_Y_dct)
    
    
    # Thêm widget slider để chọn vị trí
    x_pos = st.slider('Select x position', 0, dctBlocks.shape[1] - 1, 0)
    y_pos = st.slider('Select y position', 0, dctBlocks.shape[0] - 1, 0)
    dct_Y = dctBlocks[x_pos][y_pos][:, :] 
    # Tạo meshgrid cho biểu đồ 3D
    x = np.arange(dct_Y.shape[0])
    y = np.arange(dct_Y.shape[1])
    X, Y = np.meshgrid(x, y)
    # Tạo figure 3D của Plotly
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=dct_Y, colorscale='viridis')])
    fig.update_layout(scene=dict(xaxis_title='Frequency (X-axis)',
                                yaxis_title='Frequency (Y-axis)',
                                zaxis_title='DCT Coefficient'),
                    title=f'3D Surface of DCT at ({x_pos}, {y_pos})')
    st.plotly_chart(fig)
    
    # Lượng tử hóa
    # quantization table
    QY = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]])
    QC = np.array([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]])
    alpha = (100 - quality) / 50
    alpha = alpha
    QYquality = np.floor(QY * alpha + 0.5) if alpha != 0 else np.ceil(QY * alpha + 0.5)
    QYquality = QYquality[:w,:h]
    QC = QC[:w,:h]
    qDctBlocks = copy(dctBlocks)
    Q3 = np.dstack(QYquality*quantizationRatio)#all r-g-b/Y-Cb-Cr 3 channels need to be quantized
    Q3 = Q3*((11-w)/3)
    qDctBlocks = (qDctBlocks/Q3).round().astype('int16')
    
    
    Q_90 = np.floor(QY * alpha + 0.5) if alpha != 0 else np.ceil(QY * alpha + 0.5)
    QY_reordered = np.flipud(QY)
    Q_90_reordered = np.flipud(Q_90)
    min_value = min(np.min(Q_90), np.min(QY))
    max_value = max(np.max(Q_90), np.max(QY))
    fig1 = go.Figure()
    fig1.add_trace(go.Heatmap(z=QY_reordered, colorscale='gray', text=QY_reordered, hoverinfo='text', texttemplate="%{text}",
                textfont={"size":10}, zmin=min_value, zmax=max_value))
    fig1.update_layout(xaxis_title='X-axis', yaxis_title='Y-axis', title='Quantization Matrix Q_50', width=400, height=400)
    fig2 = go.Figure()
    fig2.add_trace(go.Heatmap(z=Q_90_reordered, colorscale='gray', text=Q_90_reordered, hoverinfo='text', texttemplate="%{text}",
                textfont={"size":10}, zmin=min_value, zmax=max_value))
    fig2.update_layout(xaxis_title='X-axis', yaxis_title='Y-axis', title=f'Quantization Matrix Q_{quality}', width=400, height=400)
    st.write("### Quantization Matrices")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    
    # Chia DCT / Quatization xong làm tròn thành số nguyên Hiển thị lên 
    coeff_dct = dctBlocks[0][0][:, :]
    coeff_quant = np.round(coeff_dct / QYquality)
    min_value = min(np.min(coeff_dct), np.min(coeff_quant))
    max_value = max(np.max(coeff_dct), np.max(coeff_quant))
    fig1 = go.Figure()
    fig1.add_trace(go.Heatmap(z=np.flipud(coeff_dct), colorscale='gray', text=np.round(np.flipud(coeff_dct),2), hoverinfo='text', texttemplate="%{text}",
                textfont={"size":10}, zmin=min_value, zmax=max_value))
    fig1.update_layout(xaxis_title='X-axis', yaxis_title='Y-axis', title='DCT Original', width=400, height=400)
    fig2 = go.Figure()
    fig2.add_trace(go.Heatmap(z=np.flipud(coeff_quant), colorscale='gray', text=np.round(np.flipud(coeff_quant),0), hoverinfo='text', texttemplate="%{text}",
                textfont={"size":10}, zmin=min_value, zmax=max_value))
    fig2.update_layout(xaxis_title='X-axis', yaxis_title='Y-axis', title='DCT After Quantization', width=400, height=400)
    st.write("### DCT and Quantization")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    
    # Chia DCT / Quatization xong làm tròn thành số nguyên Hiển thị lên 
    coeff_dct = dctBlocks[0][0][:, :]  
    coeff_quant = np.round(coeff_dct / QYquality)
    coeff_dct_conv = coeff_quant * QYquality
    no_quant = idct(idct(coeff_dct_conv.T, norm='ortho').T, norm='ortho')
    dct_origin = blocks[0][0][:, :]
    
    min_value = min(np.min(dct_origin), np.min(no_quant))
    max_value = max(np.max(dct_origin), np.max(no_quant))
    fig1 = go.Figure()
    fig1.add_trace(go.Heatmap(z=np.flipud(dct_origin), colorscale='gray', text=np.flipud(dct_origin), hoverinfo='text', texttemplate="%{text}",
                textfont={"size":10}, zmin=min_value, zmax=max_value))
    fig1.update_layout(xaxis_title='X-axis', yaxis_title='Y-axis', title='DCT Original', width=400, height=400)
    fig2 = go.Figure()
    fig2.add_trace(go.Heatmap(z=np.flipud(no_quant), colorscale='gray', text=np.round(np.flipud(no_quant),0), hoverinfo='text', texttemplate="%{text}",
                textfont={"size":10}, zmin=min_value, zmax=max_value))
    fig2.update_layout(xaxis_title='X-axis', yaxis_title='Y-axis', title='DCT After Quantization and Recovery', width=400, height=400)
    st.write("### DCT")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)
    
    # Zigzag
    st.write("### Zigzag")
    st.write(qDctBlocks[0][0][:,:])
    st.write(" ".join(list(zigZag(qDctBlocks[0][0][:,:], h, w).astype(np.str0))))
    
    # RunLength
    st.write("### RunLength")
    code1, _ = runLength(zigZag(qDctBlocks[0][0][:,:], h, w), 0, bitBits, runBits, rbBits)
    codeBlock = code1
    st.write(encodeRLE(zigZag(qDctBlocks[0][0][:,:], h, w).flatten()))
    st.write(codeBlock)
    st.write("Compresion size of this block: " + str(len(codeBlock)/8) + "KB")
    st.write("Original size of one block: " + str(w*h*3) + "KB")
    
    # Huffman
    rbCount=np.zeros(1,dtype=Counter)
    rbCount[0]=huffmanCounter(zigZag(qDctBlocks[0][0][:,:], h, w), bitBits, runBits, rbBits)
    rbCount=np.sum(rbCount)
    st.write("### Huffman")
    st.write("(run,bit) counter for Huffman Coding:\n"+str(rbCount))
    st.write("Huffman Coding:\n"+str(huffman.codebook(rbCount.items())))

    st.write("### Compress")
    t1=time.time()
    savedImg,sortedHfmForDecode=savingQuantizedDctBlocks(qDctBlocks, xLen, yLen, h, w, img, useHuffman, bitBits, runBits, rbBits)
    t2=time.time()
    st.write("Encoding: "+str(t2-t1)+" seconds")
    save = open("./imageOutput/img.bin", "wb")
    save.write(savedImg)
    save.close()
    
    #--------------------
    # Decompess
    gaps=[i for i in range(1,8)]+[8-i for i in range(8)]+[-1]
    locations=[[int(sum(range(gaps[i-1]+1))),sum(range(gaps[i]+1))] if gaps[i]>gaps[i-1]  else [64-sum(range(gaps[i-1])),64-sum(range(gaps[i]))] for i in range(len(gaps)-1)]
    
    #explaination of above section of codes
    z=zigZag(qDctBlocks[0][0][:,:], h, w)
    gaps=[ i for i in range(1,w)]+[w-i for i in range(w)]+[-1]
    locations=[[int(sum(range(gaps[i-1]+1))),sum(range(gaps[i]+1))] if gaps[i]>gaps[i-1]  else [w*h-sum(range(gaps[i-1])),w*h-sum(range(gaps[i]))] for i in range(len(gaps)-1)]
    zz1=[z[l[0]:l[1]] for l in locations]
    if w % 2 == 0:
        x = np.concatenate([np.append(np.arange(0, i), np.arange(0, i)[::-1][1:]) for i in range(2, w + 1, 2)])
        x = np.append(x, (w - 1 - x[::-1])[w:])
        y = np.concatenate([np.append(np.arange(0, i), np.arange(0, i)[::-1][1:]) for i in range(1, w + 1, 2)])
        y = np.append(np.append(y, np.arange(0, h)), (h - 1 - y[::-1]))
    else:
        x = np.concatenate([np.append(np.arange(0, i), np.arange(0, i)[::-1][1:]) for i in range(2, w, 2)])
        x = np.append(np.append(x, np.arange(0, w)), w - 1 - x[::-1])
        y = np.concatenate([np.append(np.arange(0, i), np.arange(0, i)[::-1][1:]) for i in range(1, w + 2, 2)])
        y = np.append(y[:-w], w - 1 - y[::-1])
        
    zzLine = np.dstack([x] + [y])[0]
    zzMatrix = np.zeros((w, h), dtype=np.int8)

    for i in range(len(zzLine)):
        zzMatrix[zzLine[i][1]][zzLine[i][0]] = i
    
    load = open("./imageOutput/img.bin", "rb")
    loadedbytes = load.read()
    code=bytes2runLength(loadedbytes[2:])
    
    # t1 = time.time()
    # # loadedBlocks = loadingQuantizedDctBlocks(loadedbytes, runBits, bitBits, rbBits, h, w, zzLine, sortedHfmForDecode if useHuffman else None)
    # t2 = time.time()
    # st.write("### Decompress")
    # st.write("Decoding: " + str(t2-t1) + " seconds")
    
    deDctLoadedBlocks=dctOrDedctAllBlocks(qDctBlocks*Q3, yLen, xLen, h, w ,"idct")
    loadedImg=blocks2img(deDctLoadedBlocks, xLen, yLen, h, w)
    fig_1 = px.imshow(originalImg, binary_string=True)
    fig_1.update_layout(title='Image Original')
    
    fig_2 = px.imshow(loadedImg.astype(np.int16), binary_string=True)
    fig_2.update_layout(title='Image Decompress')
    
    st.write("### Compare")
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_1, use_container_width=True)
    col2.plotly_chart(fig_2, use_container_width=True)